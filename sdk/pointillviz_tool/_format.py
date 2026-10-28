"""The converted-dataset format, read with nothing but numpy.

A converted dataset is a folder: `meta.json` says what the dataset holds,
`cells.bin` has one fixed-width record per cell in the layout `meta.json`
declares, and, when there is expression, `gene_counts.bin` holds gene-major
sparse counts indexed by `gene_index.json`. A sharded atlas has no root
`cells.bin` or gene matrix. It keeps one of each per section under
`sections/<id>/`, and the cell order a reader sees is the order a
*composition* lays those shards out in.

These readers are the part of PointillViz's own (`backend/diffexp.py`,
`backend/atlas.py`, `backend/convert.py`) that the `ds` handle needs, moved
here so that a tool's environment needs only numpy. The app keeps its copies
for its own features, and an app test holds the two to identical genes,
totals and categories.

Two things differ from the app on purpose:

  * Nothing here writes into the dataset folder. Total counts come from the
    per-cell column written at conversion time when there is one, and are
    otherwise summed in memory for the run. The app caches that sum under
    `de_cache/`, which a tool must not do: the folder may be read-only or
    shared, and a tool's run must not change the data it was handed.
  * A dataset with no gene matrix refuses with `PluginError("no_genes")`, a
    sentence the user can act on, rather than a FileNotFoundError that the
    runner can only report as a crash.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np


class PluginError(ValueError):
    """A tool refusing an input, in words a user can act on.

    Raised by tools themselves and by this package. `code` is a stable slug the
    frontend may branch on; `message` is the sentence shown to the user. A tool
    that raises anything else is reported as a crash with its traceback, which
    is the right outcome for a bug and the wrong one for "this dataset has no
    cell types" — say which one it is.

    Defined in this module, the lowest layer, because the readers raise it
    too. Import it from `pointillviz_tool`.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


TARGET_SUM = 1e4           # CP10K, the scanpy default

_NP_DTYPE = {
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32,
    "i8": np.int8, "i16": np.int16, "i32": np.int32,
    "f32": np.float32, "f64": np.float64,
}

#: Record dtypes of atlas shards. Explicitly little-endian and limited to the
#: four types the atlas builder writes: `compose_cells` keeps only the layout
#: columns named here, so widening this would change which columns it reads.
_ATLAS_DTYPE = {"f32": "<f4", "u8": "<u1", "u16": "<u2", "u32": "<u4"}


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "dataset"


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
def _load_meta(dataset_dir: str) -> dict:
    with open(os.path.join(dataset_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cells_bin(dataset_dir: str, meta: dict) -> dict[str, np.ndarray]:
    if meta.get("atlas"):
        comp = meta.get("_composition") or default_composition(meta)
        return compose_cells(dataset_dir, meta, comp)
    n_cells = int(meta["n_cells"])
    layout = meta["record_layout"]
    np_fields = [(c["name"], _NP_DTYPE[c["dtype"]]) for c in layout]
    arr = np.memmap(os.path.join(dataset_dir, "cells.bin"),
                    dtype=np.dtype(np_fields), mode="r", shape=(n_cells,))
    return {name: arr[name] for name, _ in np_fields}


def default_composition(meta: dict) -> list:
    """Every section at full detail, in atlas order, with the import-time grid."""
    return [{"section_id": s["section_id"], "count": s["n_cells"],
             "offset_x": 0.0, "offset_y": 0.0} for s in meta.get("sections", [])]


def compose_cells(dataset_dir: str, meta: dict, composition) -> dict:
    """
    Reproduce the browser's composed cell arrays for an atlas dataset.

    An atlas has no root `cells.bin`, so its cell arrays are built here from
    the shards, in the order of the composition the caller binds, so that the
    reader and the viewer agree on the index space exactly: cell index *i*
    means the same cell on both sides.

    composition = [{"section_id", "count", "offset_x", "offset_y"}, ...] in
    composition order. `count` is the LOD prefix currently loaded, which is a
    prefix of the shard because shard rows are a random permutation. The
    offsets are the runtime grid layout; applying them here is what makes a
    cross-section distance measure what the user is actually looking at.
    """
    layout = [c for c in meta["record_layout"] if c["dtype"] in _ATLAS_DTYPE]
    disk_layout = [c for c in layout if c.get("offset", 0) >= 0
                   and c["name"] != "section_idx"]
    rec = np.dtype([(c["name"], _ATLAS_DTYPE[c["dtype"]]) for c in disk_layout])

    parts = []
    total = 0
    for entry in composition:
        sid = entry["section_id"]
        sec_dir = os.path.join(dataset_dir, "sections", _safe_name(sid))
        path = os.path.join(sec_dir, "cells.bin")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"atlas section {sid!r} has no cells.bin")
        on_disk = os.path.getsize(path) // rec.itemsize
        count = int(entry.get("count") or on_disk)
        count = max(0, min(count, on_disk))
        arr = np.memmap(path, dtype=rec, mode="r", shape=(on_disk,))[:count]
        parts.append((entry, arr, count))
        total += count

    # Alignment padding is part of the record (so it must stay in `rec`), but
    # never a field anyone reads — materializing it would cost one array per
    # pad byte over the whole composition.
    real = [c for c in disk_layout if c.get("role") != "padding"]
    out: dict[str, np.ndarray] = {
        c["name"]: np.empty(total, dtype=_ATLAS_DTYPE[c["dtype"]]) for c in real}
    out["section_idx"] = np.empty(total, dtype="<u2")

    pos = 0
    for si, (entry, arr, count) in enumerate(parts):
        for c in real:
            out[c["name"]][pos:pos + count] = arr[c["name"]]
        out["x"][pos:pos + count] += np.float32(entry.get("offset_x") or 0.0)
        out["y"][pos:pos + count] += np.float32(entry.get("offset_y") or 0.0)
        out["section_idx"][pos:pos + count] = si
        pos += count
    return out


def _is_full_composition(dataset_dir: str, meta: dict) -> bool:
    """Does the bound composition cover the whole dataset, in dataset order?

    Judged against the PRISTINE meta.json on disk — not against the caller's
    meta dict, because every field of it that could answer the question is
    rewritten by some composition binder in the app: one overwrites
    meta["n_cells"], another also replaces meta["sections"] with the
    composition-filtered subset, so a section-subset composition compared
    against the caller's own sections list looks "full". Flat datasets and
    unbound atlases are always full; the disk read costs ~3 ms and only the
    composition-bound atlas path pays it.
    """
    comp = meta.get("_composition")
    if not meta.get("atlas") or comp is None:
        return True
    try:
        sections = _load_meta(dataset_dir).get("sections") or []
    except (OSError, ValueError):
        return False
    if len(comp) != len(sections):
        return False
    for entry, sec in zip(comp, sections):
        if entry.get("section_id") != sec.get("section_id"):
            return False
        count = entry.get("count")
        if count is not None and int(count) < int(sec.get("n_cells") or 0):
            return False
    return True


def categorical_fields(meta: dict) -> list[dict]:
    """Every categorical .obs column usable as a grouping.

    Deliberately the same set the viewer's own grouping menus offer: the
    cell-type column plus every entry of meta.extra_categorical. Continuous
    columns are never included.
    """
    out = []
    if meta.get("cell_types"):
        out.append({
            "field": "cell_type_idx",
            "label": meta.get("celltype_obs") or "Cell Types",
            "levels": list(meta["cell_types"]),
            "colors": meta.get("cell_type_colors") or {},
            # 65535 on >254-type datasets, where cell_type_idx is u16 and
            # index 255 is a real type. 255 for everything converted before.
            "missing_idx": int(meta.get("cell_type_missing_idx", 255)),
        })
    for c in (meta.get("extra_categorical") or []):
        if c.get("name") == "cell_type_idx" or not c.get("levels"):
            continue
        out.append({
            "field": c["name"],
            "label": c.get("source_obs") or c["name"],
            "levels": list(c["levels"]),
            "colors": c.get("colors") or {},
            "missing_idx": int(c.get("missing_idx", 255)),
        })
    return out


# ---------------------------------------------------------------------------
# Size factors for CP10K
# ---------------------------------------------------------------------------
def _total_counts(dataset_dir: str, meta: dict,
                  fields: dict[str, np.ndarray]) -> np.ndarray:
    """Per-cell total counts, used as the CP10K size factor.

    Prefers the continuous column written at conversion time. Without one it
    reuses the sum the app cached under `de_cache/` when that exists, and
    otherwise sums the gene matrix here, in memory. It never writes (see the
    module docstring), so a dataset without a totals column pays one pass over
    its gene matrix per run that asks.
    """
    for name in ("cont_transcript_counts", "cont_total_counts", "cont_n_counts"):
        arr = fields.get(name)
        if arr is not None:
            totals = np.asarray(arr, dtype=np.float64)
            if np.isfinite(totals).all() and totals.max() > 0:
                return totals

    # The composed length, not meta["n_cells"]: for an atlas under a partial
    # or LOD composition the cell arrays are shorter than the full dataset,
    # and the totals must be indexable by the same composed indices.
    n = int(fields["x"].shape[0]) if "x" in fields else int(meta["n_cells"])
    # `full` must NOT be decided from the caller's meta dict: composition
    # binders overwrite meta["n_cells"] and even meta["sections"]. The
    # composition is checked against the pristine on-disk meta instead — see
    # _is_full_composition. Only a full composition can use the app's cache:
    # a partial composition's totals are request-specific.
    if _is_full_composition(dataset_dir, meta):
        cache = os.path.join(dataset_dir, "de_cache", "total_counts.npy")
        if os.path.exists(cache):
            totals = np.load(cache)
            if totals.shape[0] == n:
                return totals.astype(np.float64)

    gi, counts_path = _gene_index(dataset_dir, meta)
    totals = np.zeros(n, dtype=np.float64)
    with open(counts_path, "rb") as fh:
        for g in range(len(gi["lengths"])):
            idx, val = _read_gene(fh, gi, g)
            if idx.size:
                np.add.at(totals, idx, val)
    return totals


def _size_factors(totals: np.ndarray) -> np.ndarray:
    """CP10K scale factor per cell: count -> count * 1e4 / total."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = TARGET_SUM / totals
    sf[~np.isfinite(sf)] = 0.0
    return sf


# ---------------------------------------------------------------------------
# Gene matrix access. gene_counts.bin is gene-major: for every gene a block of
# uint32 cell indices followed by the matching values.
#
# Two dataset shapes reach this API. A flat dataset has ONE root
# gene_index.json / gene_counts.bin pair. A sharded atlas has one pair PER
# SECTION under sections/<sid>/ and no root pair at all. The atlas branch
# below returns a gene-index-shaped dict for the SAME composition the cell
# arrays were composed with, plus a reader `_read_gene` dispatches to, so every
# caller keeps the one pattern open(counts_path) + _read_gene(fh, gi, g).
# ---------------------------------------------------------------------------
def _gene_index(dataset_dir: str, meta: dict = None) -> tuple[dict, str]:
    """(gene index dict, counts path) for flat AND sharded-atlas datasets.

    For an atlas the returned dict mimics the flat schema (genes, nnz,
    lengths, max_counts, n_cells, ...) computed for the current composition,
    and carries a private `_atlas_reader` that `_read_gene` dispatches to.
    The returned path is os.devnull so `open(counts_path, "rb")` at every
    call site stays valid — and unused.

    `meta` matters for atlases: pass the SAME dict the cell arrays were
    composed from, so a caller-bound composition (`meta["_composition"]`)
    yields gene vectors in the same composed index space as
    `_load_cells_bin`'s output. With meta=None the meta is read from disk,
    which can only ever produce the default composition (every section, full
    detail).
    """
    root = os.path.join(dataset_dir, "gene_index.json")
    if os.path.exists(root):
        with open(root, "r", encoding="utf-8") as f:
            gi = json.load(f)
        return gi, os.path.join(dataset_dir, "gene_counts.bin")
    if meta is None:
        meta = _load_meta(dataset_dir)
    if meta.get("atlas"):
        comp = meta.get("_composition") or default_composition(meta)
        reader = _AtlasGeneReader(dataset_dir, meta, comp)
        return reader.as_gene_index(), os.devnull
    raise PluginError("no_genes",
                      "This dataset has no gene expression matrix "
                      "(its gene_index.json is missing).")


class _AtlasGeneReader:
    """Per-gene reads across every shard of a composed atlas.

    Built ONCE per (dataset, composition): each shard's gene_index.json is
    parsed a single time and the per-gene retained-entry cut for the
    composition's count is precomputed, so a full-matrix scan (the totals
    pass, for one) costs one JSON parse per shard instead of one per shard
    PER GENE.

    Returned indices live in the composed index space compose_cells builds
    for the same composition: shard-local row plus the cumulative count of
    the shards before it. Shard rows are a seeded permutation with
    ascending-sorted CSC indices, so an LOD prefix of `count` rows keeps
    exactly the first `cut` entries of each gene's column.
    """

    def __init__(self, dataset_dir: str, meta: dict, composition):
        g_meta = meta.get("genes") or {}
        self.genes = list(g_meta.get("genes") or [])
        self.value_dtype = g_meta.get("value_dtype", "u16")
        sec_cells = {s.get("section_id"): int(s.get("n_cells") or 0)
                     for s in meta.get("sections") or []}
        self._parts = []            # (bin path, shard gi, base index, cuts)
        pos = 0
        for entry in composition:
            sid = entry["section_id"]
            count = int(entry.get("count") or 0)
            sec_dir = os.path.join(dataset_dir, "sections", _safe_name(sid))
            gi = None
            try:
                with open(os.path.join(sec_dir, "gene_index.json"), "r",
                          encoding="utf-8") as fh:
                    gi = json.load(fh)
            except (OSError, ValueError):
                pass
            if gi is not None:
                # compose_cells' exact rule: an absent or ZERO count means
                # the full shard (`entry.get("count") or on_disk`); a
                # NEGATIVE count clamps to zero rows (`max(0, min(...))`).
                if count == 0:
                    count = int(gi["n_cells"])
                count = max(0, min(count, int(gi["n_cells"])))
                if not self.genes:
                    # Shard panels are identical by construction (the atlas
                    # builder hands every section the harmonized common list).
                    self.genes = list(gi.get("genes") or [])
                cuts = _gene_cuts_for_count(sec_dir, gi, count,
                                            len(gi["genes"]))
                self._parts.append(
                    (os.path.join(sec_dir, "gene_counts.bin"), gi, pos,
                     np.asarray(cuts, dtype=np.int64)))
            else:
                # No readable gene index: compose_cells would still compose
                # the shard, so mirror its row count via the section's
                # declared size or the base offsets of every later shard
                # shift and gene values land on other sections' cells.
                if count == 0:
                    count = sec_cells.get(sid, 0)
                count = max(0, count)
            pos += count
        self.n_cells = pos

    def read(self, gene_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """(composed cell indices uint32, values float64) for one gene."""
        idx_parts, val_parts = [], []
        for path, gi, base, cuts in self._parts:
            cut = int(cuts[gene_idx])
            if cut <= 0:
                continue
            nnz = int(gi["nnz"][gene_idx])
            off = int(gi["offsets"][gene_idx])
            np_val = {"u16": np.uint16, "f32": np.float32}[gi["value_dtype"]]
            with open(path, "rb") as fh:
                fh.seek(off)
                idx = np.frombuffer(fh.read(cut * 4), dtype=np.uint32)
                fh.seek(off + nnz * 4)
                val = np.frombuffer(
                    fh.read(cut * np.dtype(np_val).itemsize), dtype=np_val)
            idx_parts.append(idx + np.uint32(base))
            val_parts.append(val.astype(np.float64))
        if not idx_parts:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float64)
        return np.concatenate(idx_parts), np.concatenate(val_parts)

    def as_gene_index(self) -> dict:
        """A dict shaped like a root gene_index.json, for this composition.

        `nnz` counts the entries retained at the composition's LOD cuts (the
        entries `read` actually returns), and `lengths` is synthesized from
        it so loops shaped `range(len(gi["lengths"]))` keep working. There
        are deliberately NO `offsets`: byte offsets have no meaning across
        shards, and any code that seeks with them instead of calling
        `_read_gene` should fail loudly rather than read garbage.
        """
        n_genes = len(self.genes)
        nnz = np.zeros(n_genes, dtype=np.int64)
        mx = np.zeros(n_genes, dtype=np.float64)
        item = 2 if self.value_dtype == "u16" else 4
        for _path, gi, _base, cuts in self._parts:
            nnz += cuts[:n_genes]
            mx = np.maximum(mx, np.asarray(gi["max_counts"][:n_genes],
                                           dtype=np.float64))
        return {
            "atlas": True,
            "n_cells": int(self.n_cells),
            "n_genes": n_genes,
            "genes": self.genes,
            "value_dtype": self.value_dtype,
            "nnz": nnz.tolist(),
            "lengths": (nnz * (4 + item)).tolist(),
            "max_counts": mx.tolist(),
            "total_nnz": int(nnz.sum()),
            "_atlas_reader": self,
        }


def _gene_cuts_for_count(sec_dir: str, gi: dict, count: int, n_genes: int):
    """Per-gene retained-entry count when a shard is truncated to `count` rows.

    Uses the precomputed `lod_cuts` when `count` is one of the LOD levels (the
    only counts the viewer ever sends). Any other count still has an exact
    answer — CSC indices are sorted, so it is a binary search per gene — and
    computing it keeps every reader cutting exactly. Falling back to the full
    `nnz` here instead would hand back entries for cells the composition does
    not hold.
    """
    if count >= int(gi["n_cells"]):
        return gi["nnz"]
    cells = gi.get("lod_cells") or []
    cuts = gi.get("lod_cuts") or []
    for k, v in enumerate(cells):
        if v == count and k < len(cuts) and cuts[k]:
            return cuts[k]
    path = os.path.join(sec_dir, "gene_counts.bin")
    out = []
    with open(path, "rb") as f:
        for g in range(n_genes):
            nnz = int(gi["nnz"][g])
            if nnz == 0:
                out.append(0)
                continue
            f.seek(int(gi["offsets"][g]))
            idx = np.frombuffer(f.read(nnz * 4), dtype="<u4")
            out.append(int(np.searchsorted(idx, count, side="left")))
    return out


def _read_gene(fh, gi: dict, gene_idx: int) -> tuple[np.ndarray, np.ndarray]:
    reader = gi.get("_atlas_reader")
    if reader is not None:
        return reader.read(gene_idx)
    off = gi["offsets"][gene_idx]
    length = gi["lengths"][gene_idx]
    nnz = gi["nnz"][gene_idx]
    if length == 0 or nnz == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    np_val = {"u16": np.uint16, "f32": np.float32}[gi["value_dtype"]]
    fh.seek(off)
    raw = fh.read(length)
    idx = np.frombuffer(raw, dtype=np.uint32, count=nnz)
    val = np.frombuffer(raw, dtype=np_val, count=nnz, offset=nnz * 4)
    return idx, val.astype(np.float64)
