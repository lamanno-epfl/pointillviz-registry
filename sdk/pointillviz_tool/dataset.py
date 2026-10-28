"""The `ds` handle a tool is given, and the refusal it can raise.

A PointillViz tool is a directory with a `plugin.json` manifest and an entry
file (`tool.py` or `tool.R`) defining one function:

    def run(ds, params): -> dict

`ds` is a `Dataset` (below): a read-only handle onto the ALREADY CONVERTED
dataset the user is looking at, never the source .h5ad. `params` is a plain
dict whose keys are the manifest's declared parameters, already coerced and
range-checked by the host. The return value is a **result** dict, validated by
`pointillviz_tool.results.check_result`.

Why the surface is this narrow
------------------------------
The store's promise is that a tool written against one dataset works on every
dataset, and that installing one cannot break the viewer. Both follow from the
same boundary: tools use a dataset API and run outside the server process.

  * **Same input shape everywhere.** `cells.bin` is a fixed-width record whose
    layout is per-dataset, and `gene_counts.bin` is gene-major sparse with a
    per-dataset value dtype. A tool that read either directly would be a tool
    that works on one dataset. Everything here hands back plain dense NumPy of
    a documented shape, so `ds.gene("Cd3e")` is `(n_cells,) float32` on a
    5,106-gene Xenium section and on a 10.4 M-cell atlas alike.
  * **Generated forms, extensible results.** Native outputs use the viewer's
    renderers. Tools can also return self-contained HTML views, file
    downloads and static images without changes to the host. Views run in an
    opaque-origin iframe, never in the viewer's document.
  * **Not in the server process.** Tools always run in a separate process, so
    a tool that hangs, leaks or segfaults costs one run, not the viewer.

What this is NOT
----------------
This is not a security sandbox. A tool is arbitrary code that runs on this
computer with the privileges of whoever started PointillViz, exactly like any
script downloaded from the internet and run. The gates report what a tool
imports and touches so that a human can decide; they do not and cannot prevent
a determined author from doing anything the language can do. Install tools you
are willing to run.

Coordinates, missing data, and the one convention that matters
--------------------------------------------------------------
Every per-cell array this module returns is length `n_cells`, in dataset order,
with **NaN meaning "no value here"** — a cell with no coordinates, a category
that was absent in the source `.obs`, a continuous column with a gap. Tools
must propagate that rather than filling it: `np.nanmean`, and `np.isfinite`
masks before anything that would otherwise poison a sum. The viewer applies the
same rule to a tool's own `cells` output, dimming NaN cells rather than
painting them the low end of the ramp — which is the difference between
"unmeasured" and "measured zero", and those are not the same statement.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ._format import (
    PluginError, _gene_index, _load_cells_bin, _load_meta, _read_gene,
    _size_factors, _total_counts, categorical_fields,
)

__all__ = ["Dataset", "PluginError"]

#: What `ds.device` can be. "cuda" only when the user chose GPU for a tool that
#: supports one; every gate run is "cpu".
DEVICES = ("cpu", "cuda")

#: How much on-disk neighbour cache one dataset may accumulate, in bytes.
#: Each entry is n_cells x k x 12 bytes, so it is k that drives the size, not
#: the cache count: on a 135k-cell section, k=15 is 24 MB and k=100 is
#: 162 MB. A conformance run sweeps a tool's whole declared parameter range,
#: so one afternoon of testing left 194 MB behind in a single dataset
#: directory with nothing ever removing it. Datasets live on shared volumes,
#: so "grows forever, silently, in someone else's storage" is a real defect
#: and not a tuning question. 256 MB comfortably holds the handful of k values
#: a session actually reuses.
KNN_CACHE_BUDGET_BYTES = 256 * 1024 * 1024


def _prune_cache(cache_dir: str) -> None:
    """Keep a dataset's plugin_cache under budget, evicting oldest-used first.

    By access time rather than size: the largest file is often the one a tool
    is about to ask for again (a big k is exactly the expensive case worth
    keeping), whereas the one nobody has touched since an experiment three
    hours ago is dead weight whatever it weighs. Failures here are ignored on
    purpose — pruning is housekeeping, and a cache that cannot be tidied must
    never break the run that was only trying to save a file.
    """
    try:
        entries = []
        for name in os.listdir(cache_dir):
            # Only finished cache files this module wrote. `.tmp.npz` is
            # excluded because another process may be mid-write into it, and
            # deleting that would turn a tidy-up into the very corruption this
            # whole path exists to avoid.
            if not name.endswith(".npz") or ".tmp." in name:
                continue
            full = os.path.join(cache_dir, name)
            st = os.stat(full)
            entries.append((st.st_atime, st.st_size, full))
    except OSError:
        return
    total = sum(e[1] for e in entries)
    if total <= KNN_CACHE_BUDGET_BYTES:
        return
    for _atime, size, full in sorted(entries):
        if total <= KNN_CACHE_BUDGET_BYTES:
            break
        try:
            os.remove(full)
            total -= size
        except OSError:
            pass


class Dataset:
    """Read-only access to one converted dataset.

    Constructed by the host, never by a tool. Every accessor is lazy and
    memoised for the life of one run, so asking for the same gene twice, or the
    same k-NN graph from two places in a tool, costs once.
    """

    def __init__(self, dataset_dir: str, *, subset=None, section_filter=None,
                 composition=None, device: str = "cpu", log=None, progress=None):
        if device not in DEVICES:
            raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}.")
        self.dir = dataset_dir
        self.meta = _load_meta(dataset_dir)
        if composition is not None:
            # THE extension point every atlas-aware reader honours —
            # `_load_cells_bin` and `_gene_index` both check
            # `meta.get("_composition")` before falling back to
            # `default_composition(meta)`. Setting it here, before any lazy
            # accessor below is ever touched, is what makes `ds.fields`,
            # `ds.xy`, `ds.gene(...)` and friends all read from the SAME
            # composed cell space a bound selection's indices were drawn in —
            # not a freshly-derived default one.
            self.meta["_composition"] = composition
        self.name = self.meta.get("name") or os.path.basename(dataset_dir)
        self.title = self.meta.get("title") or self.name
        self.n_cells = int(self.meta["n_cells"])
        #: "cpu" or "cuda": where this run may compute. A tool that supports a
        #: GPU puts its work here (e.g. `torch.device(ds.device)`). On "cpu"
        #: the host also hides every GPU from the process.
        self.device = device
        self._fields = None
        self._gi = None
        self._gene_pos = None
        self._gene_cache: dict = {}
        self._totals = None
        self._knn_cache: dict = {}
        self._tree_cache: dict = {}
        self._log_fn = log or (lambda _m: None)
        self._progress_fn = progress or (lambda _f, _m=None: None)

        #: Cell indices the user had selected (lasso / region), or None.
        #: Tools are free to ignore it; those that honour it should still
        #: return a FULL-length `cells` array with NaN outside the selection,
        #: so the viewer can dim rather than mislabel the rest of the tissue.
        self.subset = None if subset is None else np.asarray(subset, dtype=np.int64)
        #: Section index the user has isolated, or None for all sections.
        self.section_filter = None if section_filter is None else int(section_filter)

    # ---- logging ----------------------------------------------------------
    def log(self, message: str) -> None:
        """One line into the run log the user sees under the result."""
        self._log_fn(str(message))

    def progress(self, fraction: float, message: str = None) -> None:
        """Report 0..1 completion. Cheap; call it inside long loops."""
        self._progress_fn(float(fraction), message)

    # ---- raw record fields ------------------------------------------------
    @property
    def fields(self) -> dict:
        if self._fields is None:
            self._fields = _load_cells_bin(self.dir, self.meta)
        return self._fields

    # ---- geometry ---------------------------------------------------------
    @property
    def xy(self) -> np.ndarray:
        """(n_cells, 2) float32 world coordinates; NaN rows have no position."""
        f = self.fields
        out = np.empty((self.n_cells, 2), dtype=np.float32)
        out[:, 0] = f["x"]
        out[:, 1] = f["y"]
        return out

    @property
    def z(self) -> Optional[np.ndarray]:
        """(n_cells,) float32 depth for a 3D dataset, else None."""
        f = self.fields
        return np.asarray(f["z"], dtype=np.float32) if "z" in f else None

    @property
    def is_3d(self) -> bool:
        return (self.meta.get("spatial") or {}).get("mode") == "3d"

    @property
    def coords(self) -> np.ndarray:
        """(n_cells, 2) or (n_cells, 3) — the coordinates to do geometry in.

        Three columns only when the dataset is genuinely 3D. This, not `xy`,
        is what the k-NN graph and the trees are built on, so a tool that asks
        for neighbours on a 3D dataset gets neighbours in 3D.
        """
        f = self.fields
        parts = [np.asarray(f["x"], dtype=np.float32),
                 np.asarray(f["y"], dtype=np.float32)]
        if self.is_3d and "z" in f:
            parts.append(np.asarray(f["z"], dtype=np.float32))
        return np.stack(parts, axis=1)

    @property
    def section(self) -> np.ndarray:
        """(n_cells,) int32 section index. All zeros for a single-section set."""
        f = self.fields
        if "section_idx" in f:
            return np.asarray(f["section_idx"], dtype=np.int32)
        return np.zeros(self.n_cells, dtype=np.int32)

    @property
    def sections(self) -> list:
        return list(self.meta.get("sections") or [])

    # ---- genes ------------------------------------------------------------
    @property
    def gene_names(self) -> list:
        return list(self._gene_index()[0]["genes"])

    def has_gene(self, name: str) -> bool:
        return name in self._gene_positions()

    def gene(self, name: str, normalize: str = "cp10k") -> np.ndarray:
        """(n_cells,) float32 expression of one gene, densified.

        `normalize` is one of:
          "raw"   — the stored counts, untouched
          "cp10k" — counts per 10,000 transcripts (the viewer's default)
          "log1p" — log1p of cp10k, which is what most statistics want

        Densification is deliberate and is the point of this method: the file
        is gene-major sparse and every tool would otherwise reimplement the
        same scatter. A gene costs one seek and one read regardless of how many
        cells express it.
        """
        if normalize not in ("raw", "cp10k", "log1p"):
            raise PluginError("bad_normalize",
                              f"normalize must be raw, cp10k or log1p, not {normalize!r}")
        key = (name, normalize)
        hit = self._gene_cache.get(key)
        if hit is not None:
            return hit
        pos = self._gene_positions().get(name)
        if pos is None:
            raise PluginError(
                "unknown_gene",
                f"{name!r} is not in this dataset's panel ({len(self._gene_positions())} genes).")
        gi, path = self._gene_index()
        out = np.zeros(self.n_cells, dtype=np.float32)
        with open(path, "rb") as fh:
            idx, val = _read_gene(fh, gi, pos)
        if idx.size:
            out[idx] = val.astype(np.float32)
        if normalize != "raw":
            # `_size_factors` returns 10,000 / total per cell, so CP10K is
            # count * factor, exactly as the app's own statistics compute it.
            # Dividing here (until 2026-09-17) gave count * total / 10,000.
            out = out * self._size_factor()
            if normalize == "log1p":
                out = np.log1p(out, out=out)
        # Bounded: a tool sweeping a 5,000-gene panel would otherwise hold the
        # whole dense matrix. 64 genes at 135k cells is 35 MB.
        if len(self._gene_cache) >= 64:
            self._gene_cache.pop(next(iter(self._gene_cache)))
        self._gene_cache[key] = out
        return out

    def genes(self, names, normalize: str = "cp10k") -> np.ndarray:
        """(n_cells, len(names)) float32. Column j is `names[j]`."""
        names = list(names)
        out = np.empty((self.n_cells, len(names)), dtype=np.float32)
        for j, nm in enumerate(names):
            out[:, j] = self.gene(nm, normalize)
        return out

    def total_counts(self) -> np.ndarray:
        """(n_cells,) float64 transcripts per cell."""
        if self._totals is None:
            self._totals = _total_counts(self.dir, self.meta, self.fields)
        return self._totals

    # ---- annotations ------------------------------------------------------
    def categorical_fields(self) -> list:
        """[{name, label, levels, colors}] — every usable categorical column.

        The same set the viewer offers in its own grouping menus, so a tool's
        generated field picker can never name a column the app does not show.
        """
        return [{"name": e["field"], "label": e["label"],
                 "levels": list(e["levels"]), "colors": dict(e["colors"] or {})}
                for e in categorical_fields(self.meta)]

    def categorical(self, name: str = None):
        """(levels, codes) for one categorical column.

        `levels` is a list of label strings; `codes` is (n_cells,) int32 with
        **-1 for missing** rather than the record's own sentinel, so a tool can
        write `codes >= 0` and never has to know what 255 meant. `name=None`
        selects the cell-type column.
        """
        entries = categorical_fields(self.meta)
        if not entries:
            raise PluginError("no_categoricals",
                              "This dataset has no categorical annotations.")
        if name is None:
            entry = entries[0]
        else:
            entry = next((e for e in entries if e["field"] == name or e["label"] == name), None)
            if entry is None:
                have = ", ".join(e["field"] for e in entries)
                raise PluginError("unknown_field",
                                  f"{name!r} is not a categorical column. Have: {have}.")
        raw = self.fields.get(entry["field"])
        if raw is None:
            raise PluginError("unknown_field",
                              f"{entry['field']!r} is declared but absent from cells.bin.")
        codes = np.asarray(raw, dtype=np.int32).copy()
        n_levels = len(entry["levels"])
        codes[(codes < 0) | (codes >= n_levels)] = -1
        return list(entry["levels"]), codes

    def cell_types(self):
        """(labels, codes) for the cell-type column. -1 where unlabelled."""
        if not self.meta.get("cell_types"):
            raise PluginError("no_cell_types",
                              "This dataset has no cell-type annotation.")
        return self.categorical(None)

    @property
    def cell_type_colors(self) -> dict:
        return dict(self.meta.get("cell_type_colors") or {})

    def continuous_fields(self) -> list:
        return [{"name": c["name"], "label": c.get("source_obs") or c["name"],
                 "min": c.get("min"), "max": c.get("max")}
                for c in (self.meta.get("extra_continuous") or [])]

    def continuous(self, name: str) -> np.ndarray:
        """(n_cells,) float32 for one continuous .obs column. NaN where absent."""
        for c in (self.meta.get("extra_continuous") or []):
            if c["name"] == name or c.get("source_obs") == name:
                arr = self.fields.get(c["name"])
                if arr is None:
                    break
                return np.asarray(arr, dtype=np.float32)
        have = ", ".join(c["name"] for c in (self.meta.get("extra_continuous") or []))
        raise PluginError("unknown_field",
                          f"{name!r} is not a continuous column. Have: {have or 'none'}.")

    # ---- spatial structure ------------------------------------------------
    def kdtrees(self) -> list:
        """[{section, tree, rows}] — one scipy cKDTree PER SECTION.

        Per section, not one tree over everything, for the reason the rest of
        the app builds them that way: sections are laid out on a shared canvas
        and two of them can overlap in coordinate space, so a single tree would
        hand a cell a "neighbour" from a different specimen. `rows` maps a tree
        row back to the global cell index.
        """
        key = self.section_filter
        hit = self._tree_cache.get(key)
        if hit is not None:
            return hit
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise PluginError("no_scipy", "scipy is required for spatial queries.")
        xy = self.coords
        sect = self.section
        eligible = np.isfinite(xy).all(axis=1)
        if self.section_filter is not None:
            eligible &= (sect == self.section_filter)
        out = []
        for s in np.unique(sect[eligible]):
            rows = np.flatnonzero(eligible & (sect == s))
            if rows.size < 2:
                continue
            out.append({"section": int(s), "tree": cKDTree(xy[rows]), "rows": rows})
        if not out:
            raise PluginError("no_coordinates",
                              "No section has enough cells with usable coordinates.")
        self._tree_cache[key] = out
        return out

    def knn(self, k: int = 15):
        """(indices, distances), both (n_cells, k), built per section.

        A cell with no coordinates — or in a section smaller than k+1 — gets
        **-1** indices and **inf** distances for the slots it cannot fill, so a
        tool masks with `idx >= 0` and never silently averages cell 0 into
        every isolated neighbourhood. Cached on disk under the dataset's
        `plugin_cache/`, because the tree query is the expensive part of most
        tools and a user changing one unrelated parameter should not pay for it
        twice. That folder is the one place this package writes.
        """
        k = int(k)
        if k < 1:
            raise PluginError("bad_k", "k must be at least 1.")
        hit = self._knn_cache.get(k)
        if hit is not None:
            return hit
        cache = os.path.join(self.dir, "plugin_cache",
                             f"knn_k{k}_s{self.section_filter}.npz")
        if os.path.exists(cache):
            try:
                z = np.load(cache)
                if z["idx"].shape == (self.n_cells, k):
                    hit = (z["idx"], z["dist"])
                    self._knn_cache[k] = hit
                    return hit
            except Exception:
                pass                       # a corrupt cache is just a rebuild
        idx = np.full((self.n_cells, k), -1, dtype=np.int64)
        dist = np.full((self.n_cells, k), np.inf, dtype=np.float32)
        for part in self.kdtrees():
            rows, tree = part["rows"], part["tree"]
            kk = min(k, rows.size - 1)
            d, j = tree.query(tree.data, k=kk + 1)      # +1: self is always first
            d = np.atleast_2d(d)[:, 1:kk + 1]
            j = np.atleast_2d(j)[:, 1:kk + 1]
            idx[rows[:, None], np.arange(kk)[None, :]] = rows[j]
            dist[rows[:, None], np.arange(kk)[None, :]] = d
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            # Temp-then-rename, not a direct write. A write that runs out of
            # room truncates its target first and fails second, so writing
            # straight to `cache` turns a full disk into a 0-byte cache file
            # that the reader above then has to detect and rebuild. Worse, the
            # same failure mode once destroyed four source files on a shared
            # volume. rename(2) within a filesystem is atomic: the cache is
            # either absent or complete, never a truncated prefix.
            # The suffix must stay .npz: np.savez APPENDS .npz to any path that
            # lacks it, so a plain ".tmp" name silently becomes ".tmp.npz" and
            # the rename below then fails on a file that was never created —
            # caught by the atomic-write test, which found exactly that.
            tmp = f"{cache}.{os.getpid()}.tmp.npz"
            np.savez(tmp, idx=idx, dist=dist)
            os.replace(tmp, cache)
            _prune_cache(os.path.dirname(cache))
        except OSError:
            # A read-only dataset dir, or a full one, is not fatal — the graph
            # is already computed and in memory. Only the reuse is lost.
            try:
                os.remove(tmp)
            except (OSError, NameError, UnboundLocalError):
                pass
        self._knn_cache[k] = (idx, dist)
        return idx, dist

    # ---- internals --------------------------------------------------------
    def _gene_index(self):
        if self._gi is None:
            # self.meta, not a fresh read: on an atlas the gene reads must
            # share whatever composition the cell arrays were composed with.
            self._gi = _gene_index(self.dir, self.meta)
        return self._gi

    def _gene_positions(self) -> dict:
        if self._gene_pos is None:
            self._gene_pos = {g: i for i, g in enumerate(self._gene_index()[0]["genes"])}
        return self._gene_pos

    def _size_factor(self) -> np.ndarray:
        return _size_factors(self.total_counts()).astype(np.float32)
