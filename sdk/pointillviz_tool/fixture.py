"""A tiny synthetic dataset in the real on-disk format, for validating tools.

Every gate runs a candidate tool against this, never against a user's data. It
has to be a REAL dataset — the same `cells.bin` record layout, the same
gene-major `gene_counts.bin`, the same `meta.json` keys — because the point of
the gate is "does this tool work on the thing it will be given", and a mock
handle would validate against the mock.

It is also deliberately awkward. Real datasets have cells with no coordinates,
categories with no members, genes expressed in nothing, and more than one
section; a tool that only ever saw a clean rectangle would pass validation and
fail on the first slide someone opened. So the fixture carries all four:

  * 600 cells over 2 sections (so per-section graph handling is exercised),
  * 8 cells with NaN coordinates,
  * 6 cell types, one of which has zero cells,
  * 40 genes, one expressed nowhere and one expressed everywhere,
  * one extra categorical with a missing level, two continuous columns.

Deterministic: the same seed gives byte-identical files with the same numpy,
so a gate's verdict on one machine can be reproduced on another. (numpy does
not promise the same random streams across its own versions.)
"""

from __future__ import annotations

import json
import os

import numpy as np

FIXTURE_SEED = 20260820
N_CELLS = 600
N_GENES = 40
N_SECTIONS = 2
CELL_TYPES = ["Epithelial", "Fibroblast", "T cell", "Macrophage", "Endothelial",
              "Ghost"]                       # Ghost is real and has zero members
REGIONS = ["core", "rim", "margin"]


def write_fixture(dest_dir: str) -> dict:
    """Write the fixture into `dest_dir` and describe what it holds.

    Returns {path, seed, n_cells, n_sections, n_genes, cell_types,
    categoricals, continuous}: the facts a gate quotes when it explains why a
    tool refusing the fixture would refuse almost any dataset.
    """
    make_fixture(dest_dir)
    return {
        "path": dest_dir,
        "seed": FIXTURE_SEED,
        "n_cells": N_CELLS,
        "n_sections": N_SECTIONS,
        "n_genes": N_GENES,
        "cell_types": list(CELL_TYPES),
        "categoricals": ["cat_region"],
        "continuous": ["cont_cell_area", "cont_total_counts"],
    }


def make_fixture(out_dir: str, seed: int = FIXTURE_SEED) -> str:
    """Write a complete converted dataset into `out_dir`. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- geometry: two square sections, laid out side by side -------------
    per = N_CELLS // N_SECTIONS
    section_idx = np.repeat(np.arange(N_SECTIONS, dtype=np.uint8), per)
    xy = rng.uniform(0.0, 1000.0, size=(N_CELLS, 2)).astype(np.float32)
    xy[section_idx == 1, 0] += 1200.0          # section 2 offset on the canvas

    # ---- cell types: spatially structured, so a mixing statistic is not
    #      measuring pure noise and a tool's output can be sanity-checked ----
    codes = np.zeros(N_CELLS, dtype=np.uint8)
    cx, cy = 500.0, 500.0
    for i in range(N_CELLS):
        x = xy[i, 0] - (1200.0 if section_idx[i] else 0.0)
        r = np.hypot(x - cx, xy[i, 1] - cy)
        # Concentric shells: epithelium in the middle, stroma outside, immune
        # cells sprinkled through both. Real enough to have a boundary.
        if r < 220:
            codes[i] = 0 if rng.random() < 0.85 else 2
        elif r < 400:
            codes[i] = 1 if rng.random() < 0.7 else 3
        else:
            codes[i] = 4 if rng.random() < 0.6 else 1
    # 12 cells with no annotation at all — the 255 sentinel a real .obs gap
    # becomes at conversion time.
    codes[rng.choice(N_CELLS, 12, replace=False)] = 255

    # ---- 8 cells with no coordinates --------------------------------------
    nan_rows = rng.choice(N_CELLS, 8, replace=False)
    xy[nan_rows, :] = np.nan

    # ---- an extra categorical, with its own missing level ------------------
    region = rng.integers(0, len(REGIONS), size=N_CELLS).astype(np.uint8)
    region[rng.choice(N_CELLS, 20, replace=False)] = 255

    # ---- two continuous columns -------------------------------------------
    area = rng.gamma(shape=4.0, scale=25.0, size=N_CELLS).astype(np.float32)
    totals = rng.integers(150, 4000, size=N_CELLS).astype(np.float32)

    record = np.zeros(N_CELLS, dtype=np.dtype([
        ("x", np.float32), ("y", np.float32),
        ("section_idx", np.uint8), ("cell_type_idx", np.uint8),
        ("cat_region", np.uint8), ("_pad0", np.uint8),
        ("cont_cell_area", np.float32), ("cont_total_counts", np.float32),
    ]))
    record["x"] = xy[:, 0]
    record["y"] = xy[:, 1]
    record["section_idx"] = section_idx
    record["cell_type_idx"] = codes
    record["cat_region"] = region
    record["cont_cell_area"] = area
    record["cont_total_counts"] = totals
    record.tofile(os.path.join(out_dir, "cells.bin"))

    genes, gi = _write_genes(out_dir, rng, codes, totals)
    meta = _meta(out_dir, record, genes, gi)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return out_dir


def _write_genes(out_dir: str, rng, codes: np.ndarray, totals: np.ndarray):
    """Gene-major sparse counts, in the layout PointillViz's converter writes."""
    names = [f"Gene{i:03d}" for i in range(N_GENES)]
    names[0] = "Silent"                        # expressed in nothing
    names[1] = "Ubiquitous"                    # expressed in everything
    names[2] = "Marker0"                       # tracks cell type 0
    names[3] = "Marker1"                       # tracks cell type 1

    offsets, lengths, nnz_list, max_vals = [], [], [], []
    offset = 0
    with open(os.path.join(out_dir, "gene_counts.bin"), "wb") as fh:
        for g, name in enumerate(names):
            if name == "Silent":
                dense = np.zeros(N_CELLS)
            elif name == "Ubiquitous":
                dense = rng.integers(1, 40, size=N_CELLS).astype(np.float64)
            elif name in ("Marker0", "Marker1"):
                target = 0 if name == "Marker0" else 1
                dense = np.where(codes == target,
                                 rng.integers(5, 60, size=N_CELLS),
                                 rng.integers(0, 3, size=N_CELLS)).astype(np.float64)
            else:
                dense = rng.binomial(1, 0.25, size=N_CELLS) * rng.integers(1, 25, size=N_CELLS)
                dense = dense.astype(np.float64)
            idx = np.flatnonzero(dense > 0).astype(np.uint32)
            val = dense[idx].astype(np.uint16)
            blob = idx.tobytes() + val.tobytes()
            fh.write(blob)
            offsets.append(offset)
            lengths.append(len(blob))
            nnz_list.append(int(idx.size))
            max_vals.append(float(dense.max()))
            offset += len(blob)

    gi = {
        "n_cells": N_CELLS, "n_genes": N_GENES,
        "value_dtype": "u16", "index_dtype": "u4",
        "genes": names, "offsets": offsets, "lengths": lengths,
        "nnz": nnz_list, "max_counts": max_vals,
        "total_nnz": int(sum(nnz_list)), "file_size": offset,
    }
    with open(os.path.join(out_dir, "gene_index.json"), "w", encoding="utf-8") as fh:
        json.dump(gi, fh)
    return names, gi


def _meta(out_dir: str, record: np.ndarray, genes: list, gi: dict) -> dict:
    x, y = record["x"], record["y"]
    fin = np.isfinite(x) & np.isfinite(y)
    sections = []
    for s in range(N_SECTIONS):
        m = fin & (record["section_idx"] == s)
        sections.append({
            "index": s, "section_id": f"fixture_s{s}", "n_cells": int(m.sum()),
            "x_min": float(x[m].min()), "x_max": float(x[m].max()),
            "y_min": float(y[m].min()), "y_max": float(y[m].max()),
            "width": float(x[m].max() - x[m].min()),
            "height": float(y[m].max() - y[m].min()),
            "offset_x": 0.0, "offset_y": 0.0,
        })
    palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
    return {
        "name": "plugin_fixture",
        "title": "Plugin conformance fixture",
        "provenance": {"organism": "synthetic", "tissue": "synthetic",
                       "modality": "synthetic", "n_cells_source": N_CELLS,
                       "n_genes_source": N_GENES},
        "spatial": {"mode": "2d", "units": "px", "obsm_key": "spatial",
                    "ranges": {"x": [float(x[fin].min()), float(x[fin].max())],
                               "y": [float(y[fin].min()), float(y[fin].max())],
                               "z": None}},
        "n_cells": N_CELLS,
        "record_size": int(record.dtype.itemsize),
        "record_layout": [
            {"name": "x", "dtype": "f32", "role": "coord", "offset": 0},
            {"name": "y", "dtype": "f32", "role": "coord", "offset": 4},
            {"name": "section_idx", "dtype": "u8", "role": "section", "offset": 8},
            {"name": "cell_type_idx", "dtype": "u8", "role": "celltype", "offset": 9},
            {"name": "cat_region", "dtype": "u8", "role": "categorical", "offset": 10},
            {"name": "_pad0", "dtype": "u8", "role": "pad", "offset": 11},
            {"name": "cont_cell_area", "dtype": "f32", "role": "continuous", "offset": 12},
            {"name": "cont_total_counts", "dtype": "f32", "role": "continuous", "offset": 16},
        ],
        "sections": sections,
        "section_obs": "section",
        "section_levels": [s["section_id"] for s in sections],
        "celltype_obs": "cell_type",
        "cell_types": CELL_TYPES,
        "cell_type_colors": dict(zip(CELL_TYPES, palette)),
        "extra_categorical": [{
            "name": "cat_region", "source_obs": "region", "levels": REGIONS,
            "colors": {"core": "#1abc9c", "rim": "#e67e22", "margin": "#8e44ad"},
            "dtype": "u8", "missing_idx": 255,
        }],
        "extra_continuous": [
            {"name": "cont_cell_area", "source_obs": "cell_area", "dtype": "f32",
             "min": float(record["cont_cell_area"].min()),
             "max": float(record["cont_cell_area"].max()),
             "p1": float(np.percentile(record["cont_cell_area"], 1)),
             "p99": float(np.percentile(record["cont_cell_area"], 99))},
            {"name": "cont_total_counts", "source_obs": "total_counts", "dtype": "f32",
             "min": float(record["cont_total_counts"].min()),
             "max": float(record["cont_total_counts"].max()),
             "p1": float(np.percentile(record["cont_total_counts"], 1)),
             "p99": float(np.percentile(record["cont_total_counts"], 99))},
        ],
        "layout": {"total_width": 2200.0, "total_height": 1000.0,
                   "grid_cols": 2, "grid_rows": 1},
        "genes": {"n_genes": N_GENES, "total_nnz": gi["total_nnz"],
                  "file_size": gi["file_size"], "value_dtype": "u16", "source": "X"},
        "obsm_matrices": [], "embeddings": [], "boundaries": None, "images": None,
    }


if __name__ == "__main__":
    import sys
    import tempfile
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(),
                                                             "plugin_fixture")
    print(make_fixture(out))
