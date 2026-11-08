"""What a dataset's gene matrix holds (meta.genes.values), and what a tool reads.

PointillViz records what each dataset stores: raw counts, values the import or
the file already normalised (CP10K, CPM, logged or not), or scaled values.
Every statistic in the app reads expression as log1p(CP10K) whatever that is.
A tool reads through `ds.gene(name, "cp10k" | "log1p")`, so the SDK must give
those the app's numbers on every kind:

  * a stored log (log1p, or log2(x + 1)) is undone first, never logged again;
  * values normalised per cell keep their own normalisation: CPM is rescaled to
    CP10K, and a file normalised over the whole transcriptome and then cut to
    its panel is not normalised again over the genes that are left;
  * scaled values cannot be taken back to expression: normalising refuses in a
    sentence, and `"raw"` still reads them as stored.

Each case rewrites the conformance fixture's counts as that kind, written the
way the app writes it (f32 blocks, the record in meta.json, the transform
record in meta.json and gene_index.json), and checks the definitions.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from pointillviz_tool import Dataset, PluginError
from pointillviz_tool import _format as fmt
from pointillviz_tool import _genecodec

GENES = ("Ubiquitous", "Marker0", "Marker1", "Gene010", "Silent")


def _counts(dataset_dir):
    """{gene: dense counts} and the fixture's totals column."""
    ds = Dataset(dataset_dir)
    out = {g: ds.gene(g, "raw").astype(np.float64) for g in ds.gene_names}
    return out, ds.total_counts().copy()


def _rewrite(dataset_dir, stored, values=None, transform=None):
    """Write `stored` ({gene: dense values}) as the dataset's gene matrix, f32
    in the packed layout, with the value record and the transform record the
    app writes for it (None leaves either out)."""
    gi_path = os.path.join(dataset_dir, "gene_index.json")
    meta_path = os.path.join(dataset_dir, "meta.json")
    with open(gi_path, encoding="utf-8") as fh:
        gi = json.load(fh)
    offsets, lengths, nnz, n_esc_idx, n_esc_val, max_vals = [], [], [], [], [], []
    sha1 = hashlib.sha1()
    offset = 0
    with open(os.path.join(dataset_dir, "gene_counts.bin"), "wb") as fh:
        for name in gi["genes"]:
            dense = stored[name]
            idx = np.flatnonzero(dense != 0).astype(np.uint32)
            blob, ne, nv = _genecodec.encode_block(idx, dense[idx].astype(np.float32), "f32")
            fh.write(blob)
            sha1.update(blob)
            offsets.append(offset)
            lengths.append(len(blob))
            nnz.append(int(idx.size))
            n_esc_idx.append(ne)
            n_esc_val.append(nv)
            max_vals.append(float(dense.max()) if dense.size else 0.0)
            offset += len(blob)
    gi.update(value_dtype="f32", offsets=offsets, lengths=lengths, nnz=nnz,
              n_esc_idx=n_esc_idx, n_esc_val=n_esc_val, max_counts=max_vals,
              total_nnz=int(sum(nnz)), file_size=offset, checksum=sha1.hexdigest())
    gi.pop("transform", None)
    if transform:
        gi["transform"] = transform
    with open(gi_path, "w", encoding="utf-8") as fh:
        json.dump(gi, fh)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["genes"]["value_dtype"] = "f32"
    meta["genes"].pop("transform", None)
    meta["genes"].pop("values", None)
    if transform:
        meta["genes"]["transform"] = transform
    if values:
        meta["genes"]["values"] = values
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def _row_sums(counts):
    return np.sum([v for v in counts.values()], axis=0)


# kind -> (stored value of count c in a cell of totals T, row sums R; the
#          record; the transform record; the per-cell total CP10K divides by)
CASES = {
    # Normalised in the file over its whole transcriptome (the totals column),
    # then cut to the panel: the stored genes no longer sum to the target.
    "log1p_cp10k": (lambda c, T, R: np.log1p(c * 1e4 / T),
                    {"kind": "log1p_cp10k", "target_sum": 10000, "log_base": "e", "origin": "file"},
                    {"cp10k": True, "log1p": True, "target_sum": 10000.0, "in_file": True},
                    "column"),
    "cpm": (lambda c, T, R: c * 1e6 / T,
            {"kind": "cp10k", "target_sum": 1000000, "origin": "file"},
            {"cp10k": True, "log1p": False, "target_sum": 1000000.0, "in_file": True},
            "column"),
    "log2_cp10k": (lambda c, T, R: np.log2(c * 1e4 / T + 1),
                   {"kind": "log1p_cp10k", "target_sum": 10000, "log_base": 2, "origin": "file"},
                   {"cp10k": True, "log1p": True, "target_sum": 10000.0, "log_base": 2, "in_file": True},
                   "column"),
    # log1p of the counts themselves: the totals column still applies.
    "log1p_counts": (lambda c, T, R: np.log1p(c),
                     {"kind": "log1p_counts", "log_base": "e", "origin": "file"},
                     {"cp10k": False, "log1p": True, "in_file": True},
                     "column"),
    # The app's own CP10K + log1p, over the imported matrix's own row sums.
    "import": (lambda c, T, R: np.log1p(c * 1e4 / R),
               {"kind": "log1p_cp10k", "target_sum": 10000, "log_base": "e", "origin": "import"},
               {"cp10k": True, "log1p": True},
               "rows"),
    # The same, converted before the record existed: the transform alone says so.
    "import_no_record": (lambda c, T, R: np.log1p(c * 1e4 / R),
                         None,
                         {"cp10k": True, "log1p": True},
                         "rows"),
}


@pytest.mark.parametrize("kind", sorted(CASES))
def test_every_kind_reads_as_the_app_reads_it(fresh_fixture, kind):
    counts, T = _counts(fresh_fixture)
    R = _row_sums(counts)
    assert not np.allclose(T, R), "the panel is a cut of the transcriptome: its rows are not the totals"
    fn, values, transform, per = CASES[kind]
    stored = {g: np.where(v > 0, fn(v, T, R), 0.0) for g, v in counts.items()}
    _rewrite(fresh_fixture, stored, values, transform)
    ds = Dataset(fresh_fixture)
    total = T if per == "column" else R
    for g in GENES:
        c = counts[g]
        want = c * 1e4 / total
        got = ds.gene(g, "cp10k")
        assert np.allclose(got, want, rtol=1e-4, atol=1e-3), (kind, g, np.abs(got - want).max())
        assert np.allclose(ds.gene(g, "log1p"), np.log1p(want), atol=1e-4), (kind, g)
        assert np.array_equal(ds.gene(g, "raw"), stored[g].astype(np.float32)), \
            "raw is the stored values, untouched"
    if values:
        assert ds.gene_values["kind"] == values["kind"]
        assert ds.gene_values["origin"] == values["origin"]
    else:
        assert ds.gene_values == {"kind": "log1p_cp10k", "origin": "import",
                                  "target_sum": 10000.0, "log_base": "e"}


def test_gene_values_of_a_dataset_without_a_record_is_counts(fixture_dir):
    assert Dataset(fixture_dir).gene_values == {"kind": "counts", "origin": "file"}


def test_normalised_totals_are_the_target_sum(fresh_fixture):
    counts, T = _counts(fresh_fixture)
    stored = {g: np.where(v > 0, v * 1e6 / T, 0.0) for g, v in counts.items()}
    _rewrite(fresh_fixture, stored, CASES["cpm"][1], CASES["cpm"][2])
    totals = Dataset(fresh_fixture).total_counts()
    expressing = _row_sums(counts) > 0
    assert np.all(totals[expressing] == 1e6) and np.all(totals[~expressing] == 0), \
        "CPM values total 1,000,000 per cell: the column of true totals is not applied again"


def test_scaled_values_refuse_to_normalise_but_read_raw(fresh_fixture):
    counts, _ = _counts(fresh_fixture)
    stored = {}
    for g, v in counts.items():
        lv = np.log1p(v)
        sd = lv.std()
        stored[g] = (lv - lv.mean()) / sd if sd > 0 else np.zeros_like(lv)
    _rewrite(fresh_fixture, stored, {"kind": "scaled", "origin": "file"}, None)
    ds = Dataset(fresh_fixture)
    assert np.array_equal(ds.gene("Marker0", "raw"), stored["Marker0"].astype(np.float32))
    assert ds.gene_values["kind"] == "scaled"
    for ask in (lambda: ds.gene("Marker0", "cp10k"), lambda: ds.gene("Marker0", "log1p"),
                lambda: ds.genes(["Marker0"], "log1p"), lambda: ds.total_counts()):
        with pytest.raises(PluginError) as exc:
            ask()
        assert exc.value.code == "scaled_values"
        assert "scaled values" in str(exc.value) and "counts" in str(exc.value), \
            "the user reads why, and what to import instead"


def test_values_are_read_from_the_record_not_guessed(fresh_fixture):
    # Integers stored as f32 with no record and no transform: counts, as ever.
    counts, T = _counts(fresh_fixture)
    _rewrite(fresh_fixture, counts, None, None)
    ds = Dataset(fresh_fixture)
    assert ds.gene_values == {"kind": "counts", "origin": "file"}
    assert np.allclose(ds.gene("Marker0", "cp10k"), counts["Marker0"] * 1e4 / T, rtol=1e-4)
    assert fmt.values_of({}) == {"kind": "counts", "origin": "file"}
