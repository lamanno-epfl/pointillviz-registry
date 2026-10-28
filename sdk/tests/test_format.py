"""The on-disk readers: flat datasets, sharded atlases, and the two fixes.

The fixes are what the SDK changes about the app's readers: reading total
counts never writes into the dataset folder, and a dataset without a gene
matrix refuses with `no_genes` rather than crashing. Everything else must read
exactly what the app reads; PointillViz's own tests check that against its
readers on a real dataset.
"""

import json
import os

import numpy as np
import pytest

from pointillviz_tool import Dataset, PluginError
from pointillviz_tool import _format as fmt


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


def _write_shard(sec_dir, si, counts, lod=None):
    os.makedirs(sec_dir)
    n = counts.shape[0]
    rec = np.zeros(n, dtype=_REC)
    rec["x"] = np.arange(n, dtype=np.float32)
    rec["y"] = 10.0 * si
    rec["cat_tissue"] = si
    rec.tofile(os.path.join(sec_dir, "cells.bin"))
    offsets, lengths, nnz = [], [], []
    with open(os.path.join(sec_dir, "gene_counts.bin"), "wb") as fh:
        for g in range(counts.shape[1]):
            idx = np.flatnonzero(counts[:, g]).astype("<u4")
            blob = idx.tobytes() + counts[idx, g].astype("<u2").tobytes()
            offsets.append(fh.tell())
            lengths.append(len(blob))
            nnz.append(int(idx.size))
            fh.write(blob)
    gi = {"n_cells": n, "n_genes": len(GENES), "value_dtype": "u16", "genes": GENES,
          "offsets": offsets, "lengths": lengths, "nnz": nnz,
          "max_counts": counts.max(axis=0).astype(float).tolist()}
    if lod is not None:
        gi["lod_cells"] = [lod]
        gi["lod_cuts"] = [[int((counts[:lod, g] > 0).sum()) for g in range(len(GENES))]]
    with open(os.path.join(sec_dir, "gene_index.json"), "w") as fh:
        json.dump(gi, fh)


@pytest.fixture
def atlas(tmp_path):
    root = str(tmp_path / "atlas")
    for si, (sid, counts) in enumerate(SHARDS.items()):
        _write_shard(os.path.join(root, "sections", sid), si, counts,
                     lod=3 if sid == "B" else None)
    meta = {
        "atlas": True, "name": "atlas_probe", "n_cells": 9,
        "record_layout": [
            {"name": "x", "dtype": "f32", "role": "x", "offset": 0},
            {"name": "y", "dtype": "f32", "role": "y", "offset": 4},
            {"name": "cat_tissue", "dtype": "u8", "role": "categorical", "offset": 8},
            {"name": "_pad0", "dtype": "u8", "role": "padding", "offset": 9},
            {"name": "_pad1", "dtype": "u8", "role": "padding", "offset": 10},
            {"name": "_pad2", "dtype": "u8", "role": "padding", "offset": 11}],
        "record_size": 12,
        "sections": [{"section_id": sid, "n_cells": int(c.shape[0])}
                     for sid, c in SHARDS.items()],
        "genes": {"n_genes": len(GENES), "genes": GENES, "value_dtype": "u16"},
        "extra_categorical": [{"name": "cat_tissue", "source_obs": "tissue",
                               "levels": list(SHARDS), "missing_idx": 255}],
        "spatial": {"mode": "2d"},
    }
    with open(os.path.join(root, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return root


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
