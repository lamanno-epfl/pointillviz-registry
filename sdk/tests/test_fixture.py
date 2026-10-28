"""The fixture: the dataset every gate validates a tool against."""

import json
import os

import numpy as np

from pointillviz_tool import Dataset
from pointillviz_tool.fixture import (CELL_TYPES, FIXTURE_SEED, N_CELLS, N_GENES,
                                      N_SECTIONS, make_fixture, write_fixture)
from pointillviz_tool.manifest import check_requires, validate


def test_two_builds_are_byte_identical(tmp_path):
    a = make_fixture(str(tmp_path / "a"))
    b = write_fixture(str(tmp_path / "b"))["path"]
    for fn in ("cells.bin", "gene_counts.bin", "gene_index.json", "meta.json"):
        with open(os.path.join(a, fn), "rb") as fa, open(os.path.join(b, fn), "rb") as fb:
            assert fa.read() == fb.read(), f"{fn} differs between two builds"


def test_write_fixture_describes_what_it_wrote(fixture_info):
    assert fixture_info["seed"] == FIXTURE_SEED
    with open(os.path.join(fixture_info["path"], "meta.json")) as fh:
        meta = json.load(fh)
    assert fixture_info["n_cells"] == meta["n_cells"] == N_CELLS
    assert fixture_info["n_genes"] == meta["genes"]["n_genes"] == N_GENES
    assert fixture_info["n_sections"] == len(meta["sections"]) == N_SECTIONS
    assert fixture_info["cell_types"] == meta["cell_types"] == CELL_TYPES
    assert fixture_info["categoricals"] == [c["name"] for c in meta["extra_categorical"]]
    assert fixture_info["continuous"] == [c["name"] for c in meta["extra_continuous"]]


def test_fixture_is_awkward_on_purpose(fixture_dir):
    ds = Dataset(fixture_dir)
    assert ds.n_cells == N_CELLS
    assert int((~np.isfinite(ds.coords).all(axis=1)).sum()) == 8, \
        "the fixture must carry cells with no coordinates"
    labels, codes = ds.cell_types()
    assert int((codes < 0).sum()) > 0, "must carry unlabelled cells"
    counts = np.bincount(codes[codes >= 0], minlength=len(labels))
    assert counts[labels.index("Ghost")] == 0, \
        "the empty cell type is the point of Ghost; a tool must survive it"
    assert len(np.unique(ds.section)) == 2, "must carry two sections"
    assert float(ds.gene("Silent", "raw").max()) == 0.0
    assert int((ds.gene("Ubiquitous", "raw") > 0).sum()) == N_CELLS


def test_fixture_satisfies_an_ordinary_manifest(fixture_dir):
    with open(os.path.join(fixture_dir, "meta.json")) as fh:
        meta = json.load(fh)
    m = validate({"id": "demo-tool", "name": "Demo", "version": "1.0.0", "author": "a",
                  "description": "d", "outputs": ["summary"],
                  "requires": {"genes": True, "categoricals": True, "spatial": "2d",
                               "min_cells": 100}})
    assert check_requires(m, meta) == "", \
        "the fixture must satisfy an ordinary manifest, or every gate is testing " \
        "its own refusal path instead of the tool"
