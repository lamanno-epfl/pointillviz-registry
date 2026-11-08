"""The converted-dataset format, read with nothing but numpy.

A converted dataset is a folder: `meta.json` says what the dataset holds,
`cells.bin` has one fixed-width record per cell in the layout `meta.json`
declares, and, when there is expression, `gene_counts.bin` holds gene-major
sparse counts indexed by `gene_index.json`: one block per gene in the packed
layout `_genecodec` reads (a copy of the app's backend/genecodec.py). A
sharded atlas has no root `cells.bin` or gene matrix. It keeps one of each per
section under `sections/<id>/`, and the cell order a reader sees is the order
a *composition* lays those shards out in.

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

What the gene matrix holds is the dataset's own record, `meta.genes.values`
(`values_of`): raw counts, values the import or the file already normalised
(CP10K, CPM, logged or not), or scaled values. The readers hand back the values
as stored; normalising takes the record into account the app's way
(`_linear_of`, `_total_counts`), so `cp10k` and `log1p` are the same numbers
whatever the dataset stores, and scaled values, which cannot be taken back to
expression, are refused (`PluginError("scaled_values")`).

Gene files that are not in the packed layout (a dataset converted before it)
refuse with `PluginError("gene_layout")`, whose message says to convert the
dataset again from its source; they are never read as a dataset without genes.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from . import _genecodec


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


GENE_LAYOUT_CODE = "gene_layout"


def _check_gene_index(gi: dict, what: str = "gene_index.json") -> None:
    """Refuse a gene index that is not in the packed layout (or has a value
    dtype the layout does not know) with PluginError("gene_layout"), in the
    codec's words: convert the dataset again from its source."""
    try:
        _genecodec.check_index(gi, what)
    except _genecodec.GeneLayoutError as exc:
        raise PluginError(GENE_LAYOUT_CODE, str(exc)) from None


TARGET_SUM = 1e4           # CP10K, the scanpy default
SCAN_ENTRIES = 8_000_000   # entries a whole-matrix pass decodes at once
                           # (_genecodec.decode_blocks): memory, never a number
READ_AHEAD_ENTRIES = 1 << 18  # entries of whole blocks an atlas shard decodes
                              # ahead in a gene-by-gene pass (memory only)

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


def bind_composition(meta: dict, composition: list) -> dict:
    """`meta` as it reads through a composition: the sections and cells on view.

    An atlas's meta.json describes the whole cohort, and the viewer shows some
    of its sections at some level of detail. The cell arrays follow the
    composition (compose_cells), so everything else a reader takes from the meta
    has to as well: `n_cells` is the composed length, and `sections` lists the
    composed sections, with `index` their position, which is the code cells
    carry in `section_idx`. `n_cells` counts as compose_cells does: an absent or
    zero count is the whole shard, and a count is clamped to it.
    """
    known = {s.get("section_id"): s for s in meta.get("sections") or []}
    sections, n_cells = [], 0
    for i, entry in enumerate(composition):
        sec = known.get(entry["section_id"])
        if sec is None:
            raise PluginError("bad_composition",
                              f"This atlas has no section {entry['section_id']!r}.")
        on_disk = int(sec.get("n_cells") or 0)
        n_cells += max(0, min(int(entry.get("count") or on_disk), on_disk))
        sections.append({**sec, "index": i, "offset_x": entry.get("offset_x") or 0.0,
                         "offset_y": entry.get("offset_y") or 0.0})
    bound = dict(meta)
    bound.update(_composition=composition, n_cells=n_cells, sections=sections,
                 n_sections=len(sections),
                 section_levels=[c["section_id"] for c in composition])
    return bound


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
# What the gene matrix holds (the app's backend/expression.py)
# ---------------------------------------------------------------------------
#: What `meta.genes.values["kind"]` can be.
KINDS = ("counts", "cp10k", "log1p_cp10k", "log1p_counts", "scaled")


def values_of(meta: dict) -> dict:
    """meta.genes.values, completed: {kind, origin, target_sum?, log_base?}.

      kind        counts | cp10k | log1p_cp10k | log1p_counts | scaled
      target_sum  for the cp10k kinds, the per-cell total the values were
                  normalised to (10,000 for CP10K, 1,000,000 for CPM)
      log_base    for the log kinds, "e" (log1p) or 2 (log2(x + 1))
      origin      "file" (the matrix arrived like this) or "import" (the
                  converter's own CP10K / log1p made it)

    A dataset converted before the record existed is read from its gene
    transform (the import's CP10K / log1p); without one its values are counts,
    the one kind the converter stored before the record."""
    g = (meta or {}).get("genes") or {}
    v = g.get("values")
    if isinstance(v, dict) and v.get("kind") in KINDS:
        out = {"kind": v["kind"], "origin": v.get("origin") or "file"}
        if "cp10k" in out["kind"]:
            out["target_sum"] = float(v.get("target_sum") or TARGET_SUM)
        if out["kind"].startswith("log1p"):
            out["log_base"] = 2 if v.get("log_base") in (2, "2") else "e"
        return out
    tr = g.get("transform") or {}
    if tr.get("cp10k") or tr.get("log1p"):
        kind = ("log1p_cp10k" if tr.get("cp10k") and tr.get("log1p")
                else "cp10k" if tr.get("cp10k") else "log1p_counts")
        out = {"kind": kind, "origin": "file" if tr.get("in_file") else "import"}
        if tr.get("cp10k"):
            out["target_sum"] = float(tr.get("target_sum") or TARGET_SUM)
        if tr.get("log1p"):
            out["log_base"] = 2 if tr.get("log_base") in (2, "2") else "e"
        return out
    return {"kind": "counts", "origin": "file"}


def _require_expression(meta: dict) -> None:
    """Refuse, in the user's words, normalising a matrix of scaled values."""
    if values_of(meta)["kind"] != "scaled":
        return
    raise PluginError(
        "scaled_values",
        "This tool needs expression values — counts, or counts normalised to "
        "CP10K / CPM, logged or not. This dataset stores scaled values "
        "(z-scores or residuals), which cannot be taken back to expression. "
        "Import a counts matrix (often .raw.X or a 'counts' layer) instead.")


def _stored_transform(gi: dict, meta: dict):
    """The record stored values are undone by, where the app's readers find
    it: a flat dataset's gene index; an atlas's meta.genes.transform."""
    if gi.get("_atlas_reader") is not None:
        return (meta.get("genes") or {}).get("transform") or None
    return gi.get("transform") or None


def _linear_of(val: np.ndarray, transform) -> np.ndarray:
    """Stored values back on the linear scale (counts, or CP10K / CPM): a
    stored log1p, or log2(x + 1), undone."""
    tr = transform or {}
    if not tr.get("log1p"):
        return val
    if tr.get("log_base") in (2, "2"):
        return np.exp2(val) - 1.0
    return np.expm1(val)


# ---------------------------------------------------------------------------
# Size factors for CP10K
# ---------------------------------------------------------------------------
def _normalised_totals(sums: np.ndarray, kind: dict) -> np.ndarray:
    """Values normalised per cell: the target sum for every cell that
    expresses anything, 0 for one that expresses nothing."""
    return np.where(sums > 0, float(kind.get("target_sum") or TARGET_SUM), 0.0)


def _total_counts(dataset_dir: str, meta: dict,
                  fields: dict[str, np.ndarray]) -> np.ndarray:
    """Per-cell total counts, used as the CP10K size factor.

    Prefers the continuous column written at conversion time. Without one it
    reuses the sum the app cached under `de_cache/` when that exists, and
    otherwise sums the gene matrix here, in memory. It never writes (see the
    module docstring), so a dataset without a totals column pays one pass over
    its gene matrix per run that asks.

    Values already normalised per cell (CP10K, CPM; the import's or the
    file's) total the target sum they were normalised to, for every cell that
    expresses anything, as in the app: a column of true totals would scale
    them a second time, and their own sums would normalise again a matrix
    whose genes were cut after normalising. A stored log is undone before
    summing. A matrix of scaled values is refused.
    """
    _require_expression(meta)
    kind = values_of(meta)
    cp10k = "cp10k" in kind["kind"]
    for name in () if cp10k else ("cont_transcript_counts", "cont_total_counts", "cont_n_counts"):
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
                totals = totals.astype(np.float64)
                return _normalised_totals(totals, kind) if cp10k else totals

    gi, counts_path = _gene_index(dataset_dir, meta)
    tr = _stored_transform(gi, meta)
    totals = np.zeros(n, dtype=np.float64)
    # Many genes per decode (_scan_matrix). np.add.at adds in the order given,
    # and each cell still takes its genes in gene order, so these are the
    # float64 sums of a gene-by-gene pass over _read_gene, bit for bit.
    for idx, val in _scan_matrix(gi, counts_path):
        if idx.size:
            np.add.at(totals, idx, _linear_of(val, tr))
    return _normalised_totals(totals, kind) if cp10k else totals


def _size_factors(totals: np.ndarray) -> np.ndarray:
    """CP10K scale factor per cell: count -> count * 1e4 / total."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = TARGET_SUM / totals
    sf[~np.isfinite(sf)] = 0.0
    return sf


# ---------------------------------------------------------------------------
# Gene matrix access. gene_counts.bin is gene-major: for every gene one block
# of its cell indices and values in the packed layout `_genecodec` decodes back
# to exactly the arrays that were written. An index not marked
# `"layout": "packed"` is refused with PluginError("gene_layout").
#
# Two dataset shapes reach this API. A flat dataset has ONE root
# gene_index.json / gene_counts.bin pair. A sharded atlas has one pair PER
# SECTION under sections/<sid>/ and no root pair at all. The atlas branch
# below returns a gene-index-shaped dict for the SAME composition the cell
# arrays were composed with, plus a reader `_read_gene` dispatches to, so every
# caller keeps the one pattern open(counts_path) + _read_gene(fh, gi, g).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Genes a section did not measure.
#
# A multi-section dataset whose sections were measured on different panels (a
# Xenium 5K section beside a ~500-gene MERFISH one) keeps the UNION of the
# genes. A section holds an EMPTY column for a gene it did not measure, so
# `_read_gene` returns nothing for those cells — exactly what it returns for a
# measured zero. The difference is recorded beside the matrix, as each
# section's `unmeasured_genes` in meta.json; GenePanels turns it into per-cell
# masks. The app's copy is backend/diffexp.py's GenePanels, held to this one
# by the app's tests.
# ---------------------------------------------------------------------------
class GenePanels:
    """Which cells' sections did not measure which gene, for one gene index.

    `groups` holds one entry per section that lacks some gene, in the index
    space the gene reads live in: {"section_id", "range": (lo, hi)} for an
    atlas composition, or {"section_id", "code"} for a flat dataset, whose
    cells carry their section as `section_idx`. `by_gene` maps a gene index to
    the positions in `groups` lacking it; a gene it does not list was measured
    everywhere. The cells a gene is missing from are always whole sections, so
    masks are built once per distinct set of sections and cached.
    """

    def __init__(self, n_cells: int, groups=(), by_gene=None,
                 section_codes=None):
        self.n_cells = int(n_cells)
        self.groups = list(groups)
        self._by_gene = dict(by_gene or {})
        self._section_codes = section_codes
        self._codes = None
        self._masks: dict = {}

    @property
    def partial(self) -> bool:
        """True when some gene is missing from some cells of this dataset."""
        return bool(self._by_gene)

    def lacking(self, gene_idx: int) -> tuple:
        """Positions in `groups` that did not measure the gene; () when every
        section did."""
        return self._by_gene.get(int(gene_idx), ())

    def section_ids(self, sig) -> list:
        return [self.groups[k]["section_id"] for k in sig]

    def mask(self, sig) -> np.ndarray:
        """Boolean over cells, True where the cell's section is in `sig`."""
        hit = self._masks.get(sig)
        if hit is not None:
            return hit
        out = np.zeros(self.n_cells, dtype=bool)
        for k in sig:
            grp = self.groups[k]
            if "range" in grp:
                lo, hi = grp["range"]
                out[lo:hi] = True
            else:
                if self._codes is None:
                    self._codes = np.asarray(self._section_codes())
                out |= self._codes == grp["code"]
        self._masks[sig] = out
        return out

    def unmeasured_mask(self, gene_idx: int):
        """Boolean over cells, True where the gene was not measured; None when
        every cell's section measured it."""
        sig = self.lacking(gene_idx)
        return self.mask(sig) if sig else None


_NO_PANELS = GenePanels(0)


def gene_panels(gi: dict) -> GenePanels:
    """The GenePanels a `_gene_index` dict carries (an empty one if none)."""
    return gi.get("_panels") or _NO_PANELS


def _flat_panels(dataset_dir: str, meta: dict, n_cells: int) -> GenePanels:
    """GenePanels for a flat dataset, from meta.json's `sections[]`."""
    if meta is None:
        try:
            meta = _load_meta(dataset_dir)
        except (OSError, ValueError):
            return GenePanels(n_cells)
    groups, by_gene = [], {}
    for s in meta.get("sections") or []:
        miss = s.get("unmeasured_genes")
        if not miss:
            continue
        k = len(groups)
        groups.append({"section_id": s.get("section_id"), "code": int(s.get("index") or 0)})
        for g in miss:
            by_gene.setdefault(int(g), []).append(k)
    if not groups:
        return GenePanels(n_cells)
    return GenePanels(
        n_cells, groups, {g: tuple(v) for g, v in by_gene.items()},
        section_codes=lambda: _load_cells_bin(dataset_dir, meta)["section_idx"])


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

    Both shapes carry `_panels`, a GenePanels: which cells' sections did not
    measure which gene. Read it with `gene_panels(gi)`.

    An index (root or shard) that is not in the packed layout refuses with
    PluginError("gene_layout").
    """
    root = os.path.join(dataset_dir, "gene_index.json")
    if os.path.exists(root):
        with open(root, "r", encoding="utf-8") as f:
            gi = json.load(f)
        _check_gene_index(gi, "gene_index.json")
        gi["_panels"] = _flat_panels(dataset_dir, meta, int(gi.get("n_cells") or 0))
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
        # By section_id, not position: a composition-bound meta may list only
        # the composed sections.
        sec_missing = {s.get("section_id"): s.get("unmeasured_genes")
                       for s in meta.get("sections") or []
                       if s.get("unmeasured_genes")}
        groups, by_gene = [], {}
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
                # Outside the try: a shard in the old layout is refused, never
                # composed as a shard without a gene index (empty genes).
                _check_gene_index(gi, f"sections/{_safe_name(sid)}/gene_index.json")
                # compose_cells' exact rule: an absent or ZERO count means
                # the full shard (`entry.get("count") or on_disk`); a
                # NEGATIVE count clamps to zero rows (`max(0, min(...))`).
                if count == 0:
                    count = int(gi["n_cells"])
                count = max(0, min(count, int(gi["n_cells"])))
                if not self.genes:
                    # Shard gene lists are identical by construction (the
                    # atlas builder writes every shard over the union list; a
                    # gene a section lacks is an empty column there).
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
            if sec_missing.get(sid) and count > 0:
                k = len(groups)
                groups.append({"section_id": sid, "range": (pos, pos + count)})
                for g in sec_missing[sid]:
                    by_gene.setdefault(int(g), []).append(k)
            pos += count
        self.n_cells = pos
        self.panels = GenePanels(pos, groups,
                                 {g: tuple(v) for g, v in by_gene.items()})
        # Read-ahead for a pass over the genes (read): per part, the run it
        # decoded ahead, (first gene, end gene, indptr, indices, values), the
        # shard's cumulative nnz, and whether it reads directly from now on.
        self._last, self._streak = -2, 0
        self._ahead = [None] * len(self._parts)
        self._cum = [None] * len(self._parts)
        self._direct = [False] * len(self._parts)

    def read(self, gene_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """(composed cell indices uint32, values float64) for one gene.

        In a pass over the genes (the third gene in a row asked for right
        after the one before it) each shard decodes a run of genes ahead
        (_run_ahead), and genes a held run covers come from it; any other
        request reads its blocks directly. The same arrays either way."""
        g = int(gene_idx)
        self._streak = self._streak + 1 if g == self._last + 1 else 0
        self._last = g
        sequential = self._streak >= 2
        idx_parts, val_parts = [], []
        for p, (path, gi, base, cuts) in enumerate(self._parts):
            nnz, length = gi["nnz"][g], gi["lengths"][g]
            if (nnz == 0) != (length == 0):
                # read_gene's rule, also where the cut below would skip it.
                raise _genecodec.GeneLayoutError(
                    f"gene {g}: {nnz} entries with a {length}-byte block")
            cut = int(cuts[g])
            if cut <= 0:
                continue
            run = self._ahead[p]
            if (run is None or not run[0] <= g < run[1]) and sequential and not self._direct[p]:
                run = self._run_ahead(p, g)
            if run is not None and run[0] <= g < run[1]:
                a, b = run[2][g - run[0]], run[2][g - run[0] + 1]
                idx, val = run[3][a:b], run[4][a:b]
            else:
                with open(path, "rb") as fh:
                    # The first `cut` entries, reading only their bytes.
                    idx, val = _genecodec.read_gene(fh, gi, g, cut=cut)
            idx_parts.append(idx + np.uint32(base))
            val_parts.append(val.astype(np.float64))
        if not idx_parts:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float64)
        return np.concatenate(idx_parts), np.concatenate(val_parts)

    def _run_ahead(self, p: int, g: int):
        """Part p's genes from g on, about READ_AHEAD_ENTRIES entries of whole
        blocks decoded at once (_genecodec.decode_blocks) and cut to each gene's
        first `cut` entries, held in place of the part's last run. On a block
        that does not decode, None and direct reads for this part from then on,
        so a broken gene raises when it is read, as before."""
        path, gi, base, cuts = self._parts[p]
        if self._cum[p] is None:
            self._cum[p] = np.cumsum(np.asarray(gi["nnz"], dtype=np.int64))
        cum = self._cum[p]
        end = int(np.searchsorted(cum, (cum[g - 1] if g else 0) + READ_AHEAD_ENTRIES, side="right"))
        end = min(len(cuts), max(end, g + 1))
        cut = cuts[g:end]
        nnz = np.asarray(gi["nnz"][g:end], dtype=np.int64)
        want = g + np.flatnonzero(cut > 0)
        try:
            indptr, idx, val = _genecodec.decode_blocks(_file_bytes(path), gi, want)
        except (_genecodec.GeneLayoutError, KeyError):
            self._ahead[p], self._direct[p] = None, True
            return None
        take = np.clip(np.minimum(cut, nnz), 0, None)       # read_gene's min(cut, nnz)
        counts = np.diff(indptr)
        if not np.array_equal(take[want - g], counts):
            within = np.arange(idx.size, dtype=np.int64) - np.repeat(indptr[:-1], counts)
            keep = within < np.repeat(take[want - g], counts)
            idx, val = idx[keep], val[keep]
        ptr = np.zeros(end - g + 1, dtype=np.int64)
        np.cumsum(take, out=ptr[1:])
        run = (g, end, ptr, idx, val)
        self._ahead[p] = run
        return run

    def scan(self):
        """What `read` returns for every gene, (composed cell indices uint32,
        values float64), shard by shard and many genes per decode: for a pass
        that only sums per cell (_scan_matrix). A cell lies in one shard, so
        its sums still take its genes in gene order.

        Whole blocks are decoded and cut to each gene's retained entries, the
        same first `cut` entries `read` takes. Every gene is checked first,
        those the cuts skip too, as `read` checks them."""
        n_genes = len(self.genes)
        for path, gi, base, cuts in self._parts:
            _check_entries(gi, n_genes)
            cut = cuts[np.arange(n_genes)]
            want = np.flatnonzero(cut > 0)
            if not want.size:
                continue
            raw = _file_bytes(path)
            for run in _gene_runs(want, gi["nnz"]):
                indptr, idx, val = _genecodec.decode_blocks(raw, gi, run)
                counts = np.diff(indptr)
                within = np.arange(idx.size, dtype=np.int64) - np.repeat(indptr[:-1], counts)
                keep = within < np.repeat(cut[run], counts)
                yield idx[keep] + np.uint32(base), val[keep].astype(np.float64)

    def check_entries(self) -> None:
        """read_gene's rule over every shard's index at once (_check_entries):
        for a pass that skips genes by their counts and never reads them."""
        n_genes = len(self.genes)
        for _path, gi, _base, _cuts in self._parts:
            _check_entries(gi, n_genes)

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
            "_panels": self.panels,
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
    # Many genes per decode (_gene_runs bounds the memory); the binary search
    # stays per gene.
    raw = _file_bytes(os.path.join(sec_dir, "gene_counts.bin"))
    nnz = np.asarray([int(gi["nnz"][g]) for g in range(n_genes)], dtype=np.int64)
    out = [0] * n_genes
    for run in _gene_runs(np.flatnonzero(nnz), nnz):
        indptr, idx, _vals = _genecodec.decode_blocks(raw, gi, run)
        for j, g in enumerate(run):
            out[int(g)] = int(np.searchsorted(idx[indptr[j]:indptr[j + 1]], count, side="left"))
    return out


def _read_gene(fh, gi: dict, gene_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """(cell indices uint32, values float64) of one gene, as stored. An empty
    gene of a flat dataset is (uint32[0], float32[0]); of an atlas,
    (uint32[0], float64[0])."""
    reader = gi.get("_atlas_reader")
    if reader is not None:
        return reader.read(gene_idx)
    if gi.get("layout") != _genecodec.LAYOUT:
        _check_gene_index(gi)          # an index loaded without the check
    length = gi["lengths"][gene_idx]
    nnz = gi["nnz"][gene_idx]
    if length == 0 and nnz == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    # Only one of the two 0 is a broken index entry: read_gene raises GeneLayoutError.
    idx, val = _genecodec.read_gene(fh, gi, gene_idx)
    return idx, val.astype(np.float64)


def _check_entries(gi: dict, n=None) -> None:
    """_genecodec.read_gene's rule over a whole index (its first `n` genes): an
    entry with exactly one of nnz and length 0 raises GeneLayoutError, never
    reads as an empty gene, also in a pass that skips genes by their counts."""
    nnz = np.asarray(gi["nnz"][:n], dtype=np.int64)
    lengths = np.asarray(gi["lengths"][:n], dtype=np.int64)
    bad = np.flatnonzero((nnz == 0) != (lengths == 0))
    if bad.size:
        g = int(bad[0])
        raise _genecodec.GeneLayoutError(
            f"gene {g}: {int(nnz[g])} entries with a {int(lengths[g])}-byte block")


def _gene_runs(genes, nnz, limit: int = SCAN_ENTRIES) -> list:
    """`genes` cut into consecutive pieces of about `limit` entries (by `nnz`);
    a gene with more is a piece of its own. Bounds the memory of a decode of
    many genes at once."""
    genes = np.asarray(genes, dtype=np.int64)
    cum = np.cumsum(np.asarray(nnz, dtype=np.int64)[genes])
    out, s = [], 0
    while s < genes.size:
        e = int(np.searchsorted(cum, (cum[s - 1] if s else 0) + limit, side="right"))
        e = min(genes.size, max(e, s + 1))
        out.append(genes[s:e])
        s = e
    return out


def _file_bytes(path: str) -> np.ndarray:
    """A whole gene file as a read-only uint8 memmap (np.memmap refuses an
    empty file, which holds no block: an empty array then)."""
    if os.path.getsize(path) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.memmap(path, dtype=np.uint8, mode="r")


def _scan_matrix(gi: dict, counts_path: str):
    """Every gene's entries as `_read_gene` gives them (cell indices uint32,
    values float64), concatenated a run of genes at a time and decoded
    together (_genecodec.decode_blocks): for a pass that only sums per cell,
    which adds each cell's genes in gene order either way."""
    reader = gi.get("_atlas_reader")
    if reader is not None:
        yield from reader.scan()
        return
    if gi.get("layout") != _genecodec.LAYOUT:
        _check_gene_index(gi)          # an index loaded without the check
    raw = _file_bytes(counts_path)
    for run in _gene_runs(np.arange(len(gi["lengths"])), gi["nnz"]):
        _, idx, val = _genecodec.decode_blocks(raw, gi, run)
        yield idx, val.astype(np.float64)
