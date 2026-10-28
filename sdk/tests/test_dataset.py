"""The `ds` handle a tool is given.

What is defended is not "the code runs" but the promises a tool is written
against: every per-cell array is length n_cells with NaN (or -1) for "no
value"; the k-NN graph never crosses a section and marks what it could not
fill rather than quietly substituting cell 0; a dataset without genes refuses
in a sentence; and the k-NN cache under plugin_cache/ is the only thing the
handle ever writes.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pointillviz_tool
from pointillviz_tool import Dataset, PluginError
from pointillviz_tool.dataset import KNN_CACHE_BUDGET_BYTES, _prune_cache
from pointillviz_tool.fixture import N_CELLS


def refusal(fn):
    with pytest.raises(PluginError) as exc:
        fn()
    return exc.value


def test_the_public_names():
    for name in ("PluginError", "Dataset", "check_result", "html_view",
                 "download_file", "image"):
        assert hasattr(pointillviz_tool, name), name
    assert pointillviz_tool.__version__ == "1.0.0"
    from pointillviz_tool import dataset, results
    assert dataset.PluginError is results.PluginError is PluginError
    assert issubclass(PluginError, ValueError)


def test_importing_the_sdk_does_not_import_scipy():
    code = ("import sys, pointillviz_tool, pointillviz_tool.manifest, "
            "pointillviz_tool.bundle, pointillviz_tool.fixture; "
            "print('scipy' in sys.modules)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(pointillviz_tool.__file__)))
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"


def test_device_is_cpu_unless_the_run_chose_a_gpu(fixture_dir):
    assert Dataset(fixture_dir).device == "cpu"
    assert Dataset(fixture_dir, device="cuda").device == "cuda"
    with pytest.raises(ValueError):
        Dataset(fixture_dir, device="gpu")


def test_arrays_are_full_length(fixture_dir):
    ds = Dataset(fixture_dir)
    for name, arr in (("gene", ds.gene("Ubiquitous")), ("continuous", ds.continuous("cont_cell_area")),
                      ("codes", ds.cell_types()[1]), ("section", ds.section),
                      ("totals", ds.total_counts())):
        assert arr.shape[0] == ds.n_cells, f"{name} is not n_cells long"
    assert ds.coords.shape == (ds.n_cells, 2)
    assert ds.xy.shape == (ds.n_cells, 2)
    assert ds.genes(["Silent", "Marker0"]).shape == (ds.n_cells, 2)


def test_missing_is_minus_one_not_the_sentinel(fixture_dir):
    ds = Dataset(fixture_dir)
    _, codes = ds.cell_types()
    assert codes.min() == -1 and 255 not in np.unique(codes), \
        "the record's 255 sentinel must be translated; a tool writing `codes >= 0` " \
        "is the documented idiom and it has to work"
    levels, region = ds.categorical("cat_region")
    assert region.max() < len(levels) and region.min() == -1


def test_normalisation_modes_differ_correctly(fixture_dir):
    ds = Dataset(fixture_dir)
    raw = ds.gene("Ubiquitous", "raw")
    cp10k = ds.gene("Ubiquitous", "cp10k")
    log1p = ds.gene("Ubiquitous", "log1p")
    # The definition, not just "different from raw": a handle that divided by
    # the size factor returned count * total / 1e4 and still looked rescaled.
    totals = ds.total_counts()
    ok = totals > 0
    assert np.allclose(cp10k[ok], raw[ok] * 1e4 / totals[ok], rtol=1e-4), \
        "cp10k is count * 10,000 / total transcripts"
    assert np.allclose(log1p, np.log1p(cp10k), atol=1e-5), "log1p is log1p OF cp10k"
    assert raw.dtype == np.float32
    assert refusal(lambda: ds.gene("Ubiquitous", "zscore")).code == "bad_normalize"


def test_unknown_names_are_sentences(fixture_dir):
    ds = Dataset(fixture_dir)
    exc = refusal(lambda: ds.gene("NotAGene"))
    assert exc.code == "unknown_gene"
    assert "NotAGene" in str(exc) and "panel" in str(exc), \
        "the message must name what was asked for and what was available"
    assert refusal(lambda: ds.categorical("nope")).code == "unknown_field"
    assert refusal(lambda: ds.continuous("nope")).code == "unknown_field"


def test_a_dataset_without_genes_refuses_in_a_sentence(fresh_fixture):
    os.remove(os.path.join(fresh_fixture, "gene_index.json"))
    ds = Dataset(fresh_fixture)
    for ask in (lambda: ds.gene_names, lambda: ds.has_gene("Silent"),
                lambda: ds.gene("Silent")):
        assert refusal(ask).code == "no_genes"
    assert ds.n_cells == N_CELLS and ds.cell_types()[0], \
        "everything that is not expression still works"


def test_totals_without_genes_or_a_column_refuse(fresh_fixture):
    os.remove(os.path.join(fresh_fixture, "gene_index.json"))
    meta = json.load(open(os.path.join(fresh_fixture, "meta.json")))
    for col in meta["record_layout"]:
        if col["name"] == "cont_total_counts":
            col["name"] = "cont_other"
    json.dump(meta, open(os.path.join(fresh_fixture, "meta.json"), "w"))
    assert refusal(lambda: Dataset(fresh_fixture).total_counts()).code == "no_genes"


def test_knn_marks_what_it_could_not_fill(fixture_dir):
    ds = Dataset(fixture_dir)
    idx, dist = ds.knn(8)
    assert idx.shape == (ds.n_cells, 8) and dist.shape == idx.shape
    no_coords = ~np.isfinite(ds.coords).all(axis=1)
    assert (idx[no_coords] < 0).all(), \
        "a cell with no position must get -1 neighbours, not cell 0 -- the latter " \
        "silently averages one arbitrary cell into every isolated neighbourhood"
    assert np.isinf(dist[no_coords]).all()
    assert (idx[~no_coords] >= 0).all(), "every positioned cell has 8 neighbours here"
    assert (idx[~no_coords] != np.arange(ds.n_cells)[~no_coords, None]).all(), \
        "a cell must never be its own neighbour"
    assert refusal(lambda: ds.knn(0)).code == "bad_k"


def test_knn_never_crosses_a_section(fixture_dir):
    ds = Dataset(fixture_dir)
    idx, _ = ds.knn(8)
    sect = ds.section
    valid = idx >= 0
    rows = np.repeat(np.arange(ds.n_cells)[:, None], 8, axis=1)[valid]
    assert (sect[idx[valid]] == sect[rows]).all(), \
        "sections are laid out on a shared canvas and can overlap in coordinate " \
        "space; one tree over everything hands a cell a neighbour from another specimen"


def test_section_filter_isolates(fixture_dir):
    ds = Dataset(fixture_dir, section_filter=1)
    idx, _ = ds.knn(6)
    sect = ds.section
    assert (idx[sect == 0] < 0).all(), "a filtered-out section must have no graph at all"
    assert (idx[(sect == 1) & np.isfinite(ds.coords).all(axis=1)] >= 0).any()


def test_knn_caches_on_disk_and_a_fresh_handle_reuses_it(fresh_fixture, snapshot,
                                                         monkeypatch):
    changed = snapshot(fresh_fixture)
    first, _ = Dataset(fresh_fixture).knn(8)
    cache = os.path.join(fresh_fixture, "plugin_cache")
    assert os.listdir(cache) == ["knn_k8_sNone.npz"], "the graph must be cached on disk"
    assert all(p.startswith("plugin_cache") for p in changed()), \
        "plugin_cache/ is the only place the handle writes"

    def rebuilt(self):
        raise AssertionError("a fresh handle rebuilt a graph that was cached on disk")
    monkeypatch.setattr(Dataset, "kdtrees", rebuilt)
    again, _ = Dataset(fresh_fixture).knn(8)     # a FRESH handle: only the disk can help
    assert np.array_equal(first, again)


def test_a_corrupt_cache_rebuilds(fresh_fixture):
    Dataset(fresh_fixture).knn(8)
    with open(os.path.join(fresh_fixture, "plugin_cache", "knn_k8_sNone.npz"), "wb") as fh:
        fh.write(b"not an npz")
    idx, _ = Dataset(fresh_fixture).knn(8)
    assert idx.shape == (N_CELLS, 8), "a corrupt cache must mean a rebuild, not a crash"


def test_the_cache_is_bounded_and_evicts_by_use(fresh_fixture):
    # One afternoon of conformance runs once left 194 MB in a single dataset
    # directory (k=100 alone is a 162 MB file at 135k cells) with nothing ever
    # removing it, on a shared volume that later hit zero bytes free.
    cache = os.path.join(fresh_fixture, "plugin_cache")
    os.makedirs(cache)
    names = []
    for i, mb in enumerate((200, 100, 100)):
        f = os.path.join(cache, f"knn_k{i}_sNone.npz")
        with open(f, "wb") as fh:
            fh.truncate(mb * 1024 * 1024)
        os.utime(f, (1000 + i, 1000 + i))       # atime ascending: file 0 oldest
        names.append(f)
    keep = os.path.join(cache, "notes.txt")     # not ours; must survive
    with open(keep, "w") as fh:
        fh.write("x")
    _prune_cache(cache)
    total = sum(os.path.getsize(os.path.join(cache, n))
                for n in os.listdir(cache) if n.endswith(".npz"))
    assert total <= KNN_CACHE_BUDGET_BYTES, f"{total} bytes left, over budget"
    assert not os.path.exists(names[0]), "the least recently used file must go first"
    assert os.path.exists(names[2]), "the most recently used file must survive"
    assert os.path.exists(keep), "pruning must never touch a file this module did not write"


def test_a_full_disk_cannot_leave_a_truncated_or_stray_cache(fresh_fixture, monkeypatch):
    # A write that runs out of room truncates its target and THEN fails, so a
    # direct write leaves a 0-byte cache behind. The temp write is made to fail
    # after the real graph is computed; the destination must be untouched.
    Dataset(fresh_fixture).knn(6)
    path = os.path.join(fresh_fixture, "plugin_cache", "knn_k6_sNone.npz")
    good = os.path.getsize(path)

    def boom(file, **kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(np, "savez", boom)
    idx, _ = Dataset(fresh_fixture).knn(9)      # a NEW k, so it tries to write
    assert idx.shape == (N_CELLS, 9), "the graph must still be returned"
    assert os.path.getsize(path) == good, \
        "an ENOSPC while caching one graph must not damage another"
    stray = [n for n in os.listdir(os.path.dirname(path)) if ".tmp" in n]
    assert not stray, f"a failed cache write left temp files behind: {stray}"


def test_field_lists_match_the_viewers_menus(fixture_dir):
    ds = Dataset(fixture_dir)
    names = [f["name"] for f in ds.categorical_fields()]
    assert names[0] == "cell_type_idx", "the cell-type column leads, as in the viewer"
    assert "cat_region" in names
    assert [f["name"] for f in ds.continuous_fields()] == ["cont_cell_area", "cont_total_counts"]
    assert ds.cell_type_colors["Ghost"] == "#911eb4"


def test_selection_and_section_are_carried_not_applied(fixture_dir):
    ds = Dataset(fixture_dir, subset=[3, 1, 2], section_filter="1")
    assert ds.subset.dtype == np.int64 and ds.subset.tolist() == [3, 1, 2]
    assert ds.section_filter == 1
    assert ds.gene("Ubiquitous").shape == (N_CELLS,), "the handle never applies the subset"


def test_log_and_progress_reach_the_host(fixture_dir):
    seen = []
    ds = Dataset(fixture_dir, log=lambda m: seen.append(("log", m)),
                 progress=lambda f, m=None: seen.append(("progress", f, m)))
    ds.log(3)
    ds.progress(1, "done")
    assert seen == [("log", "3"), ("progress", 1.0, "done")]
    Dataset(fixture_dir).log("no host attached is fine")
