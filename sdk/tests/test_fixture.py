"""The fixture: the dataset every gate validates a tool against."""

import hashlib
import json
import os
import re

import numpy as np

from pointillviz_tool import Dataset, _genecodec
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


def test_the_genes_are_in_the_packed_layout_the_converter_writes(fixture_dir):
    with open(os.path.join(fixture_dir, "gene_index.json")) as fh:
        gi = json.load(fh)
    with open(os.path.join(fixture_dir, "meta.json")) as fh:
        meta = json.load(fh)
    size = os.path.getsize(os.path.join(fixture_dir, "gene_counts.bin"))
    assert gi["layout"] == "packed" and gi["value_dtype"] == "u16"
    assert gi["file_size"] == meta["genes"]["file_size"] == size == sum(gi["lengths"])
    assert gi["offsets"] == [0] + np.cumsum(gi["lengths"])[:-1].tolist(), \
        "one block per gene, back to back, in gene order"
    assert len(gi["n_esc_idx"]) == len(gi["n_esc_val"]) == N_GENES
    assert gi["lengths"][gi["genes"].index("Silent")] == 0, "an empty gene is an empty block"
    with open(os.path.join(fixture_dir, "gene_counts.bin"), "rb") as fh:
        for g in range(N_GENES):
            idx, val = _genecodec.read_gene(fh, gi, g)
            assert idx.size == val.size == gi["nnz"][g]
            assert (np.diff(idx.astype(np.int64)) > 0).all() and (val > 0).all()
    _genecodec.check_index(gi)


def test_the_index_carries_the_counts_files_checksum(fixture_dir):
    # The browser's cache key and a requirement of its index check: the
    # lowercase hex SHA-1 of the whole counts file, right after file_size.
    with open(os.path.join(fixture_dir, "gene_index.json")) as fh:
        gi = json.load(fh)
    with open(os.path.join(fixture_dir, "gene_counts.bin"), "rb") as fh:
        want = hashlib.sha1(fh.read()).hexdigest()
    assert gi["checksum"] == want
    assert re.fullmatch(r"[0-9a-f]{40}", gi["checksum"])
    keys = list(gi)
    assert keys[keys.index("file_size") + 1] == "checksum"


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
