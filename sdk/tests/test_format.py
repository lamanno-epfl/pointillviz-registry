"""The on-disk readers: flat datasets, sharded atlases, and the two fixes.

The fixes are what the SDK changes about the app's readers: reading total
counts never writes into the dataset folder, and a dataset without a gene
matrix refuses with `no_genes` rather than crashing. Everything else must read
exactly what the app reads; PointillViz's own tests check that against its
readers on a real dataset.

Gene files are in the packed layout (`_genecodec`); files in any other layout
refuse with `gene_layout`, never read as a dataset without genes.
"""

import json
import os

import numpy as np
import pytest

from pointillviz_tool import Dataset, PluginError
from pointillviz_tool import _format as fmt
from pointillviz_tool import _genecodec


def _meta(path):
    with open(os.path.join(path, "meta.json")) as fh:
        return json.load(fh)


def _rewrite_meta(path, edit):
    meta = _meta(path)
    edit(meta)
    with open(os.path.join(path, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return meta


def _drop_totals_column(meta):
    """Rename the totals column, so the reader has to sum the genes itself."""
    for col in meta["record_layout"] + meta["extra_continuous"]:
        if col["name"] == "cont_total_counts":
            col["name"] = "cont_other"


def _dense(path, name):
    """One gene, densified straight from the files, for comparison."""
    gi, counts = fmt._gene_index(path)
    out = np.zeros(int(_meta(path)["n_cells"]))
    with open(counts, "rb") as fh:
        idx, val = fmt._read_gene(fh, gi, gi["genes"].index(name))
    out[idx] = val
    return out


# ---------------------------------------------------------------------------
# Flat datasets
# ---------------------------------------------------------------------------
def test_cells_follow_the_record_layout(fixture_dir):
    meta = fmt._load_meta(fixture_dir)
    fields = fmt._load_cells_bin(fixture_dir, meta)
    assert set(fields) == {c["name"] for c in meta["record_layout"]}
    assert all(arr.shape == (meta["n_cells"],) for arr in fields.values())
    assert fields["x"].dtype == np.float32 and fields["cell_type_idx"].dtype == np.uint8


def test_categorical_fields_lead_with_cell_types(fixture_dir):
    entries = fmt.categorical_fields(fmt._load_meta(fixture_dir))
    assert [e["field"] for e in entries] == ["cell_type_idx", "cat_region"]
    assert entries[0]["label"] == "cell_type" and entries[0]["missing_idx"] == 255


def test_totals_prefer_the_conversion_column(fixture_dir):
    meta = fmt._load_meta(fixture_dir)
    fields = fmt._load_cells_bin(fixture_dir, meta)
    totals = fmt._total_counts(fixture_dir, meta, fields)
    assert totals.dtype == np.float64
    assert np.array_equal(totals, np.asarray(fields["cont_total_counts"], dtype=np.float64))


def test_totals_without_a_column_are_summed_in_memory(fresh_fixture, snapshot):
    meta = _rewrite_meta(fresh_fixture, _drop_totals_column)
    changed = snapshot(fresh_fixture)
    fields = fmt._load_cells_bin(fresh_fixture, meta)
    totals = fmt._total_counts(fresh_fixture, meta, fields)
    expected = sum(_dense(fresh_fixture, g) for g in fmt._gene_index(fresh_fixture)[0]["genes"])
    assert np.array_equal(totals, expected)
    assert changed() == [] and not os.path.exists(os.path.join(fresh_fixture, "de_cache")), \
        "reading total counts must never write into the dataset folder"


def test_totals_work_on_a_read_only_dataset(fresh_fixture):
    meta = _rewrite_meta(fresh_fixture, _drop_totals_column)
    os.chmod(fresh_fixture, 0o555)
    try:
        if os.access(fresh_fixture, os.W_OK):
            pytest.skip("this user can write into a read-only folder")
        ds = Dataset(fresh_fixture)
        assert ds.total_counts().shape == (meta["n_cells"],)
        assert np.isfinite(ds.gene("Ubiquitous", "cp10k")).all()
    finally:
        os.chmod(fresh_fixture, 0o755)


def test_totals_reuse_the_apps_cache_without_writing(fresh_fixture, snapshot):
    meta = _rewrite_meta(fresh_fixture, _drop_totals_column)
    os.makedirs(os.path.join(fresh_fixture, "de_cache"))
    cached = np.arange(meta["n_cells"], dtype=np.float64) + 1
    np.save(os.path.join(fresh_fixture, "de_cache", "total_counts.npy"), cached)
    changed = snapshot(fresh_fixture)
    totals = fmt._total_counts(fresh_fixture, meta, fmt._load_cells_bin(fresh_fixture, meta))
    assert np.array_equal(totals, cached) and changed() == []


def test_a_missing_gene_matrix_is_a_refusal(fresh_fixture):
    os.remove(os.path.join(fresh_fixture, "gene_index.json"))
    with pytest.raises(PluginError) as exc:
        fmt._gene_index(fresh_fixture)
    assert exc.value.code == "no_genes" and "gene" in exc.value.message


def _edit_index(path, edit):
    with open(path) as fh:
        gi = json.load(fh)
    edit(gi)
    with open(path, "w") as fh:
        json.dump(gi, fh)


def test_gene_files_in_the_old_layout_are_a_refusal(fresh_fixture):
    _edit_index(os.path.join(fresh_fixture, "gene_index.json"), lambda gi: gi.pop("layout"))
    with pytest.raises(PluginError) as exc:
        fmt._gene_index(fresh_fixture)
    assert exc.value.code == "gene_layout"
    assert "convert it again" in exc.value.message, "the message says what to do"
    ds = Dataset(fresh_fixture)
    for ask in (lambda: ds.gene("Silent"), lambda: ds.gene_names):
        with pytest.raises(PluginError) as exc:
            ask()
        assert exc.value.code == "gene_layout", "never an empty gene"


def test_a_value_dtype_the_layout_does_not_know_is_a_refusal(fresh_fixture):
    _edit_index(os.path.join(fresh_fixture, "gene_index.json"),
                lambda gi: gi.update(value_dtype="u32"))
    with pytest.raises(PluginError) as exc:
        fmt._gene_index(fresh_fixture)
    assert exc.value.code == "gene_layout"


@pytest.mark.parametrize("gene, key, value", [("Ubiquitous", "lengths", 0), ("Silent", "lengths", 8),
                                              ("Ubiquitous", "nnz", 0)])
def test_an_entry_whose_nnz_and_length_disagree_is_never_an_empty_gene(fresh_fixture, gene, key, value):
    path = os.path.join(fresh_fixture, "gene_index.json")
    _edit_index(path, lambda gi: gi[key].__setitem__(gi["genes"].index(gene), value))
    with pytest.raises(_genecodec.GeneLayoutError):
        _dense(fresh_fixture, gene)
    # The whole-matrix pass (no totals column) meets it as well.
    meta = _rewrite_meta(fresh_fixture, _drop_totals_column)
    with pytest.raises(_genecodec.GeneLayoutError):
        fmt._total_counts(fresh_fixture, meta, fmt._load_cells_bin(fresh_fixture, meta))


def test_escaped_gaps_and_values_read_back_exactly(fresh_fixture):
    # Gaps of 255 or more and u16 values of 255 or more are escaped in a block;
    # the fixture's own genes have neither, so one gene here has both.
    n = int(_meta(fresh_fixture)["n_cells"])
    dense = np.zeros(n)
    rows = np.array([0, 254, 255, 256, 511, 599])
    dense[rows] = [255, 1, 65535, 254, 300, 7]
    idx = np.flatnonzero(dense).astype(np.uint32)
    block, ne, nv = _genecodec.encode_block(idx, dense[idx].astype(np.uint16), "u16")
    assert ne > 0 and nv > 0, "the gene must exercise both escapes"
    with open(os.path.join(fresh_fixture, "gene_counts.bin"), "ab") as fh:
        offset = fh.tell()
        fh.write(block)

    def put(gi):
        g = gi["genes"].index("Silent")
        gi["offsets"][g], gi["lengths"][g], gi["nnz"][g] = offset, len(block), int(idx.size)
        gi["n_esc_idx"][g], gi["n_esc_val"][g] = ne, nv
    _edit_index(os.path.join(fresh_fixture, "gene_index.json"), put)
    assert np.array_equal(_dense(fresh_fixture, "Silent"), dense)
    assert np.array_equal(Dataset(fresh_fixture).gene("Silent", "raw"), dense.astype(np.float32))


# ---------------------------------------------------------------------------
# Sharded atlases
# ---------------------------------------------------------------------------
GENES = ["G0", "G1", "G2"]
SHARDS = {
    # rows are cells, columns GENES; zeros are left out of the sparse file
    "A": np.array([[1, 0, 2], [0, 0, 0], [3, 1, 0], [0, 4, 0], [5, 0, 1]]),
    "B": np.array([[0, 2, 0], [7, 0, 0], [0, 0, 3], [1, 1, 1]]),
}
_REC = np.dtype([("x", "<f4"), ("y", "<f4"), ("cat_tissue", "u1"),
                 ("_pad0", "u1"), ("_pad1", "u1"), ("_pad2", "u1")])


def _write_shard(sec_dir, si, counts, lod=None, value_dtype="u16"):
    os.makedirs(sec_dir)
    n = counts.shape[0]
    rec = np.zeros(n, dtype=_REC)
    rec["x"] = np.arange(n, dtype=np.float32)
    rec["y"] = 10.0 * si
    rec["cat_tissue"] = si
    rec.tofile(os.path.join(sec_dir, "cells.bin"))
    offsets, lengths, nnz, n_esc_idx, n_esc_val = [], [], [], [], []
    with open(os.path.join(sec_dir, "gene_counts.bin"), "wb") as fh:
        for g in range(counts.shape[1]):
            idx = np.flatnonzero(counts[:, g]).astype("<u4")
            vals = counts[idx, g].astype("<u2" if value_dtype == "u16" else "<f4")
            block, ne, nv = _genecodec.encode_block(idx, vals, value_dtype)
            offsets.append(fh.tell())
            lengths.append(len(block))
            nnz.append(int(idx.size))
            n_esc_idx.append(ne)
            n_esc_val.append(nv)
            fh.write(block)
    gi = {"n_cells": n, "n_genes": len(GENES), "layout": "packed", "value_dtype": value_dtype,
          "genes": GENES, "offsets": offsets, "lengths": lengths, "nnz": nnz,
          "n_esc_idx": n_esc_idx, "n_esc_val": n_esc_val,
          "max_counts": counts.max(axis=0).astype(float).tolist()}
    if lod is not None:
        gi["lod_cells"] = [lod]
        gi["lod_cuts"] = [[int((counts[:lod, g] > 0).sum()) for g in range(len(GENES))]]
    with open(os.path.join(sec_dir, "gene_index.json"), "w") as fh:
        json.dump(gi, fh)


def _write_atlas(root, shards, lods, value_dtype="u16", genes_extra=None):
    for si, (sid, counts) in enumerate(shards.items()):
        _write_shard(os.path.join(root, "sections", sid), si, counts, lod=lods.get(sid),
                     value_dtype=value_dtype)
    meta = {
        "atlas": True, "name": "atlas_probe",
        "n_cells": int(sum(c.shape[0] for c in shards.values())),
        "record_layout": [
            {"name": "x", "dtype": "f32", "role": "x", "offset": 0},
            {"name": "y", "dtype": "f32", "role": "y", "offset": 4},
            {"name": "cat_tissue", "dtype": "u8", "role": "categorical", "offset": 8},
            {"name": "_pad0", "dtype": "u8", "role": "padding", "offset": 9},
            {"name": "_pad1", "dtype": "u8", "role": "padding", "offset": 10},
            {"name": "_pad2", "dtype": "u8", "role": "padding", "offset": 11}],
        "record_size": 12,
        "sections": [{"section_id": sid, "n_cells": int(c.shape[0])}
                     for sid, c in shards.items()],
        "genes": {"n_genes": len(GENES), "genes": GENES, "value_dtype": value_dtype,
                  **(genes_extra or {})},
        "extra_categorical": [{"name": "cat_tissue", "source_obs": "tissue",
                               "levels": list(shards), "missing_idx": 255}],
        "spatial": {"mode": "2d"},
    }
    with open(os.path.join(root, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return root


@pytest.fixture
def atlas(tmp_path):
    return _write_atlas(str(tmp_path / "atlas"), SHARDS, {"B": 3})


def _atlas_gene(root, meta, name):
    gi, counts = fmt._gene_index(root, meta)
    out = np.zeros(gi["n_cells"])
    with open(counts, "rb") as fh:
        idx, val = fmt._read_gene(fh, gi, GENES.index(name))
    out[idx] = val
    return out


def test_compose_cells_lays_shards_end_to_end(atlas):
    meta = fmt._load_meta(atlas)
    comp = fmt.default_composition(meta)
    comp[1]["offset_x"] = 100.0
    fields = fmt.compose_cells(atlas, meta, comp)
    assert "_pad0" not in fields, "padding is part of the record, never a field"
    assert fields["section_idx"].tolist() == [0] * 5 + [1] * 4
    assert fields["x"].tolist() == [0, 1, 2, 3, 4, 100, 101, 102, 103]
    assert fields["cat_tissue"].tolist() == [0] * 5 + [1] * 4


def test_atlas_genes_live_in_the_composed_index_space(atlas):
    meta = fmt._load_meta(atlas)
    full = np.vstack(list(SHARDS.values()))
    for g, name in enumerate(GENES):
        assert np.array_equal(_atlas_gene(atlas, meta, name), full[:, g])
    assert fmt._gene_index(atlas, meta)[0]["nnz"] == (full > 0).sum(axis=0).tolist()


def test_a_partial_composition_cuts_every_gene_to_its_prefix(atlas):
    # A at 2 of 5 rows (no precomputed cut: a binary search per gene), B at
    # 3 of 4 rows (its precomputed LOD cut). Both must keep the same cells.
    meta = fmt._load_meta(atlas)
    meta["_composition"] = [{"section_id": "A", "count": 2}, {"section_id": "B", "count": 3}]
    expected = np.vstack([SHARDS["A"][:2], SHARDS["B"][:3]])
    fields = fmt._load_cells_bin(atlas, meta)
    assert fields["x"].shape == (5,)
    for g, name in enumerate(GENES):
        assert np.array_equal(_atlas_gene(atlas, meta, name), expected[:, g])
    totals = fmt._total_counts(atlas, meta, fields)
    assert np.array_equal(totals, expected.sum(axis=1))


def test_the_handle_reads_through_a_composition(atlas):
    # The sections on view, in the viewer's order, one of them at a level of detail.
    comp = [{"section_id": "B", "count": 3, "offset_x": 10.0, "offset_y": 0.0},
            {"section_id": "A", "count": 0, "offset_x": 0.0, "offset_y": 0.0}]
    ds = Dataset(atlas, composition=comp)
    assert ds.n_cells == 8
    assert ds.xy.shape == (8, 2) and ds.total_counts().shape == (8,)
    assert ds.section.tolist() == [0] * 3 + [1] * 5
    assert [(s["section_id"], s["index"]) for s in ds.sections] == [("B", 0), ("A", 1)]
    expected = np.vstack([SHARDS["B"][:3], SHARDS["A"]])
    assert np.array_equal(ds.gene("G1", "raw"), expected[:, 1].astype(np.float32))
    assert Dataset(atlas).n_cells == 9, "the handle without one still holds every section"


def test_a_composition_of_a_section_the_atlas_lacks_is_refused(atlas):
    with pytest.raises(PluginError) as caught:
        Dataset(atlas, composition=[{"section_id": "Z", "count": 1}])
    assert caught.value.code == "bad_composition"


def test_compositions_keep_their_own_neighbour_graphs(atlas):
    # Two of one size in another order: the same file would hand each the other's cell indices.
    a = {"section_id": "A", "count": 4, "offset_x": 0.0, "offset_y": 0.0}
    b = {"section_id": "B", "count": 4, "offset_x": 0.0, "offset_y": 0.0}
    first, second = Dataset(atlas, composition=[a, b]), Dataset(atlas, composition=[b, a])
    for ds in (first, second):
        idx, _ = ds.knn(2)
        for row in range(ds.n_cells):
            assert (ds.section[idx[row]] == ds.section[row]).all()
    assert first.knn(2)[0][0].max() < 4 and second.knn(2)[0][0].max() < 4
    assert first._composition_tag() != second._composition_tag()
    whole = fmt.default_composition(fmt._load_meta(atlas))
    assert Dataset(atlas, composition=whole)._composition_tag() == ""


def test_a_gene_a_section_did_not_measure_is_nan_there_not_zero(atlas):
    # What the atlas builder records for sections of different panels: an empty
    # column, and the gene's index in the section's `unmeasured_genes`.
    path = os.path.join(atlas, "meta.json")
    with open(path) as fh:
        meta = json.load(fh)
    meta["sections"][0]["unmeasured_genes"] = [GENES.index("G1")]
    with open(path, "w") as fh:
        json.dump(meta, fh)
    ds = Dataset(atlas)
    got = ds.gene("G1", "raw")
    assert np.isnan(got[:5]).all(), "A did not measure G1"
    assert np.array_equal(got[5:], SHARDS["B"][:, 1].astype(np.float32))
    assert (got[5:] == 0).any(), "a zero stays a zero where the gene was measured"
    assert ds.gene_measured("G1").tolist() == [False] * 5 + [True] * 4
    assert ds.gene_measured("G0").all() and not np.isnan(ds.gene("G0", "raw")).any()
    assert np.isnan(ds.genes(["G0", "G1"], "raw")[:5, 1]).all()
    # With only the section that measured it on view, nothing is missing.
    bound = Dataset(atlas, composition=[{"section_id": "B", "count": 4}])
    assert not np.isnan(bound.gene("G1", "raw")).any() and bound.gene_measured("G1").all()


def test_atlas_totals_are_summed_without_writing(atlas, snapshot):
    meta = fmt._load_meta(atlas)
    changed = snapshot(atlas)
    totals = fmt._total_counts(atlas, meta, fmt._load_cells_bin(atlas, meta))
    assert np.array_equal(totals, np.vstack(list(SHARDS.values())).sum(axis=1))
    assert changed() == [] and not os.path.exists(os.path.join(atlas, "de_cache"))


def test_the_handle_reads_an_atlas(atlas):
    ds = Dataset(atlas)
    full = np.vstack(list(SHARDS.values()))
    assert ds.n_cells == 9 and ds.gene_names == GENES
    assert np.array_equal(ds.gene("G1", "raw"), full[:, 1].astype(np.float32))
    assert ds.section.tolist() == [0] * 5 + [1] * 4
    levels, codes = ds.categorical("tissue")
    assert levels == ["A", "B"] and codes.tolist() == [0] * 5 + [1] * 4


def test_an_atlas_of_logged_counts_normalises_as_its_counts_do(tmp_path):
    # An atlas whose shards hold log1p of the counts (the transform record in
    # meta.genes, as the app writes it for an atlas): undone before the totals
    # and before CP10K, never logged a second time; "raw" stays the logs.
    plain = Dataset(_write_atlas(str(tmp_path / "counts"), SHARDS, {}))
    logged = Dataset(_write_atlas(
        str(tmp_path / "logged"), {k: np.log1p(v.astype(np.float64)) for k, v in SHARDS.items()}, {},
        value_dtype="f32",
        genes_extra={"values": {"kind": "log1p_counts", "log_base": "e", "origin": "file"},
                     "transform": {"cp10k": False, "log1p": True, "in_file": True}}))
    assert logged.gene_values == {"kind": "log1p_counts", "origin": "file", "log_base": "e"}
    assert np.allclose(logged.total_counts(), plain.total_counts())
    for g in GENES:
        assert np.allclose(logged.gene(g, "cp10k"), plain.gene(g, "cp10k"), rtol=1e-5)
        assert np.allclose(logged.gene(g, "log1p"), plain.gene(g, "log1p"), atol=1e-6)
        assert np.allclose(logged.gene(g, "raw"), np.log1p(plain.gene(g, "raw")), atol=1e-6)


def test_atlas_prefixes_read_escaped_entries_exactly(tmp_path):
    # Gaps and values a block escapes, cut at an LOD level (a prefix read) and
    # at a count that is not one (a binary search over the decoded indices),
    # behind a shard that shifts every index.
    wide = np.zeros((700, len(GENES)), dtype=np.int64)
    wide[[0, 300, 301, 399, 400, 650], 0] = [1, 1000, 254, 255, 2, 65535]
    wide[[5, 260, 699], 1] = [300, 3, 4]
    root = _write_atlas(str(tmp_path / "wide"), {"A": SHARDS["A"], "W": wide}, {"W": 400})
    meta = fmt._load_meta(root)
    for count in (0, 400, 350, 300, 1):
        bound = dict(meta, _composition=[{"section_id": "A", "count": 0},
                                         {"section_id": "W", "count": count}])
        expected = np.vstack([SHARDS["A"], wide[:count or 700]])
        for g, name in enumerate(GENES):
            assert np.array_equal(_atlas_gene(root, bound, name), expected[:, g]), (count, name)
        assert fmt._gene_index(root, bound)[0]["nnz"] == (expected > 0).sum(axis=0).tolist()


def test_an_atlas_shard_in_the_old_layout_is_a_refusal(atlas):
    _edit_index(os.path.join(atlas, "sections", "B", "gene_index.json"),
                lambda gi: gi.pop("layout"))
    with pytest.raises(PluginError) as exc:
        fmt._gene_index(atlas, fmt._load_meta(atlas))
    assert exc.value.code == "gene_layout" and "sections/B/gene_index.json" in exc.value.message
    with pytest.raises(PluginError) as exc:
        Dataset(atlas).gene("G0", "raw")
    assert exc.value.code == "gene_layout", "never a section without genes"


@pytest.mark.parametrize("order", ["sequential", "skips", "reverse", "random"])
def test_a_pass_over_the_genes_reads_what_single_reads_do(tmp_path, monkeypatch, order):
    # A gene-by-gene pass decodes runs of genes ahead per shard; tiny runs here
    # cross every run boundary, over escaped entries and LOD cuts.
    monkeypatch.setattr(fmt, "READ_AHEAD_ENTRIES", 2)
    wide = np.zeros((700, len(GENES)), dtype=np.int64)
    wide[[0, 300, 301, 399, 400, 650], 0] = [1, 1000, 254, 255, 2, 65535]
    wide[[5, 260, 699], 1] = [300, 3, 4]
    root = _write_atlas(str(tmp_path / "wide"), {"A": SHARDS["A"], "W": wide}, {"W": 400})
    meta = fmt._load_meta(root)
    genes = {"sequential": [0, 1, 2] * 3, "skips": [0, 2, 0, 1, 2],
             "reverse": [2, 1, 0, 2, 1, 0], "random": [1, 0, 2, 2, 0, 1, 1]}[order]
    for count in (0, 400, 350, 1):
        bound = dict(meta, _composition=[{"section_id": "A", "count": 0},
                                         {"section_id": "W", "count": count}])
        direct, path = fmt._gene_index(root, bound)
        direct["_atlas_reader"]._direct = [True] * len(direct["_atlas_reader"]._parts)
        gi, _ = fmt._gene_index(root, bound)
        with open(path, "rb") as fh:
            for g in genes:
                i, v = fmt._read_gene(fh, gi, g)
                ri, rv = fmt._read_gene(fh, direct, g)
                assert (i.dtype, v.dtype) == (ri.dtype, rv.dtype) == (np.uint32, np.float64)
                assert i.tobytes() == ri.tobytes() and v.tobytes() == rv.tobytes(), (count, g)


def test_a_broken_block_ahead_fails_the_pass_at_that_gene(tmp_path, monkeypatch):
    # A run decoded ahead that holds a broken block falls back to direct reads,
    # so the genes before it still read and the broken one raises when read.
    monkeypatch.setattr(fmt, "READ_AHEAD_ENTRIES", 1 << 20)
    root = _write_atlas(str(tmp_path / "broken"), SHARDS, {})
    _edit_index(os.path.join(root, "sections", "B", "gene_index.json"),
                lambda gi: gi["lengths"].__setitem__(2, 0))
    gi, path = fmt._gene_index(root, fmt._load_meta(root))
    with open(path, "rb") as fh:
        for g in (0, 1):
            fmt._read_gene(fh, gi, g)
        with pytest.raises(_genecodec.GeneLayoutError):
            fmt._read_gene(fh, gi, 2)


@pytest.mark.parametrize("key", ["lengths", "nnz"])
def test_a_shard_entry_with_one_of_nnz_and_length_0_raises_in_every_reader(atlas, key):
    # Exactly one of the two 0 is a broken index entry. A shard's cut skips a
    # gene with nnz 0, so the readers check the entry itself, never reading
    # it as a gene without entries in that section.
    _edit_index(os.path.join(atlas, "sections", "B", "gene_index.json"),
                lambda gi: gi[key].__setitem__(2, 0))
    meta = fmt._load_meta(atlas)

    def read(order):
        gi, path = fmt._gene_index(atlas, meta)
        with open(path, "rb") as fh:
            for g in order:
                fmt._read_gene(fh, gi, g)
    readers = {"direct read": lambda: read([2]),
               "pass over the genes (read-ahead)": lambda: read([0, 1, 2]),
               "_total_counts (whole-matrix pass)":
                   lambda: fmt._total_counts(atlas, meta, fmt._load_cells_bin(atlas, meta)),
               "ds.gene": lambda: Dataset(atlas).gene("G2", "raw")}
    for name, fn in readers.items():
        with pytest.raises(_genecodec.GeneLayoutError):
            fn()
