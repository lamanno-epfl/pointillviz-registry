"""Cell-type contact network: which cell types touch which, as a network and a matrix.

Each section is triangulated on its own, in 2D or 3D, so no contact crosses sections, and every edge
of the triangulation is one contact. Each cell's neighbours are tallied by type into a count matrix
C. Row-normalising it gives W, the share of type i's contacts that are with type j, and comparing W
with the abundance of j gives the enrichment E. Sections pool by averaging W row by row, weighted by
how many cells of type i each section has.

The network's layout comes from the whole dataset's pooled contacts, with no distance limit, whatever
is isolated, selected or set in the form, so a node stands in the same place in every run on one
annotation. It is computed once and kept in plugin_cache/; computed again, it comes out the same.
"""

import csv
import gzip
import hashlib
import html
import io
import math
import os
import re

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import pdist

from pointillviz_tool import PluginError, download_file, image

PSEUDO = 1e-4           # the enrichment's pseudocount
MAX_LEVELS = 1000       # groups in one annotation: past this a column is not a set of cell types
LAYOUT_VERSION = 1      # raised when the layout's method changes: the kept layouts are then stale
LAYOUT_WORK = 1e8       # the most force evaluations a layout may spend, pairs of types times steps
CROSS_TYPES, CROSS_SHARE = 60, 0.04     # a layout is judged by these: the most abundant types' strong
#                                         contacts, whatever the form asks for
OUTPUT_LIMIT = 8 * 1024 * 1024      # images and downloads share this, in bytes on the wire
PLAIN_CSV = 1024 * 1024             # a download larger than this goes out gzipped
NO_LIMIT = "no distance limit"

# The figures, in px.
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
GROUND, INK, MUTED = "#0c0e14", "#f2f3f5", "#a3a8b4"    # the app's dark page
NO_CONTACT = "#3a3e4a"                                  # W = 0, and a type absent from this run
E_SPAN = 3.0                                            # E saturates here: 8 times, or 1/8
# Blue to red, each end bright enough to show on the page, the middle a light grey: an arrow that
# abundance alone explains is still an arrow, and it must not vanish into the ground.
RAMP = (("#2561e4", -1.0), ("#8db1f7", -0.5), ("#dcdde3", 0.0), ("#f7a18d", 0.5), ("#e8262b", 1.0))
HEAD_H, KEY_W = 56, 232                                 # the title above, the key's column beside
LABEL_FONT, LABEL_LINE, LABEL_CHARS = 13.5, 15.5, 22
MAX_NODE, MIN_NODE = 46.0, 3.5                          # the largest node's radius at most; any node's
RING_GAP, RING_SHARE = 4.0, 0.55                        # a ring's distance from its node: at W[i, i]
#                                                         = 0, and per unit of W[i, i], in radii
ARROW_ROOM = 22.0                                       # room left for an arrow between two rings
STROKE_SHARE, MAX_STROKE, CASING = 0.3, 10.0, 2.4       # an arrow's width at W = 1, in radii; its halo
HEAD_BASE, HEAD_PER_WIDTH = 7.0, 1.4                    # an arrow head's length ...
HEAD_HALF, HEAD_HALF_PER_WIDTH = 2.5, 1.1               # ... and half its width
HEAD_WEIGHT = 5                                         # a label on a head costs this many on its line
BENDS = (0.1, 0.15, 0.22, 0.3)                          # an arrow's bulge, in its length
# The viewer's own palette, for a dataset with no colours; past it, hues a golden angle apart.
PALETTE = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
           "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff", "#1abc9c", "#ff6348", "#7bed9f",
           "#ff4757", "#70a1ff", "#ffa502", "#2ed573", "#5352ed", "#ff6b81", "#dfe6e9", "#fd79a8",
           "#00cec9", "#e17055", "#0984e3", "#6c5ce7", "#fdcb6e", "#fab1a0", "#74b9ff", "#a29bfe")


def run(ds, params):
    cutoff = float(params["cutoff"]) / 100.0
    n_shown = int(params["max_types"])
    tissue = _Tissue(ds, params.get("field") or None)
    scope = tissue.scope(ds)

    ds.progress(0.05, "Counting contacts")
    scale = _Scale(ds)
    longest = float(params.get("max_distance") or 0)
    limit = scale.coordinates(longest) if longest > 0 else None
    limit_text = f"contacts up to {scale.text(limit)}" if limit else NO_LIMIT
    found = _count(tissue, scope, limit)
    _refuse_no_answer(tissue, found, limit_text, scale)

    ds.progress(0.3, "Placing the cell types")
    panel = _panel(tissue.n_types)
    places = _layout(ds, tissue, panel)

    shown = _most_abundant(found["cells"], n_shown)
    frame = {"tissue": tissue, "run": found, "shown": shown, "cutoff": cutoff, "panel": panel,
             "xy": places, "where": _where(ds, tissue, found, shown, limit_text)}
    ds.progress(0.8, "Drawing")
    network, drawn = _network_figure(frame)
    order = shown[_seriate(found["W"][np.ix_(shown, shown)])]
    matrix = _matrix_figure(frame, order)

    files = [download_file("contact-network.svg", network, media_type="image/svg+xml"),
             download_file("contact-matrix.svg", matrix, media_type="image/svg+xml")]
    spent = sum(len(s.encode("utf-8")) * 7 // 3 + 8 for s in (network, matrix))     # shown, and base64
    notes = list(found["notes"])
    if np.count_nonzero(found["cells"]) > shown.size:
        notes.insert(0, f"The figures and the table show the {shown.size} most abundant types; the "
                        "others still count in every share, and the downloads hold them.")
    files += _downloads(tissue, found, places, order, OUTPUT_LIMIT - spent, notes)
    ds.progress(1.0)
    return {
        "summary": _summary(ds, tissue, found, shown, drawn, cutoff, limit, limit_text, scale, notes),
        "images": [image(network, "Contact network: W as arrows, E as their colour"),
                   image(matrix, "Contact matrix: E as colour, W as numbers")],
        "table": _table(tissue, found, order),
        "files": files,
    }


# ---- the cells ------------------------------------------------------------------------------------
class _Tissue:
    """One table per section: each cell's position and type. Only the types that have a placed cell
    anywhere in the dataset are types at all here, numbered from 0."""

    def __init__(self, ds, field):
        levels, codes = ds.categorical(field)
        entries = ds.categorical_fields()
        entry = next((e for e in entries if e["name"] == field or e["label"] == field), entries[0])
        is_types = entry["name"] == "cell_type_idx"
        self.column = "cell types" if is_types else str(entry["label"])
        self.a_level = "a cell type" if is_types else f"a level of {entry['label']}"
        self.levels_text = "cell types" if is_types else f"groups of {entry['label']}"
        self.coords = np.asarray(ds.coords, dtype=np.float64)
        self.section = np.asarray(ds.section)
        self.has_place = np.isfinite(self.coords).all(axis=1)
        have = self.has_place & (codes >= 0)
        present = np.flatnonzero(np.bincount(codes[have], minlength=len(levels)))
        if not present.size:
            raise PluginError("no_cells", "No cell in this dataset has both coordinates and "
                                          f"{self.a_level}. Choose another column.")
        if present.size > MAX_LEVELS:
            raise PluginError("too_many_levels",
                              f"This column holds {present.size:,} {self.levels_text}, and a network "
                              f"of more than {MAX_LEVELS:,} cannot be read. Choose a coarser column.")
        number = np.full(len(levels), -1, dtype=np.int64)
        number[present] = np.arange(present.size)
        self.code = np.where(codes >= 0, number[np.maximum(codes, 0)], -1)
        self.names = [str(levels[i]) for i in present]
        self.n_types = present.size
        colours = entry.get("colors") or {}
        self.colours = [_colour(colours.get(self.names[k]), int(present[k]))
                        for k in range(present.size)]
        self.section_names = _section_names(ds)
        self.kept = {}          # the neighbours of the sections this run is about, once found
        self.failed = {}        # section -> its cells with coordinates, when Qhull refused it

    def scope(self, ds):
        """The cells this run is about: isolated, selected, placed and typed. Refuses when none is."""
        asked = np.ones(self.code.size, dtype=bool)
        if ds.section_filter is not None:
            asked &= self.section == ds.section_filter
        if ds.subset is not None and ds.subset.size:
            inside = ds.subset[(ds.subset >= 0) & (ds.subset < asked.size)]
            chosen = np.zeros(asked.size, dtype=bool)
            chosen[inside] = True
            asked &= chosen
        scope = asked & self.has_place & (self.code >= 0)
        self.n_asked = int(asked.sum())
        self.n_no_place = int((asked & ~self.has_place).sum())
        self.n_no_type = int((asked & self.has_place & (self.code < 0)).sum())
        whose = "selected cells" if ds.subset is not None and ds.subset.size else "cells"
        if not self.n_asked:
            raise PluginError("empty_scope", f"There are no {whose} in this section. Select cells "
                                             "in it, or isolate another section.")
        if not scope.any():
            raise PluginError("no_cells", f"None of these {self.n_asked:,} {whose} has both "
                                          f"coordinates and {self.a_level}. Widen the selection, or "
                                          "choose another column.")
        self.in_run = set(np.unique(self.section[scope]).tolist())
        return scope

    def neighbours(self, s):
        """(rows, from, to): the section's placed cells and their Delaunay contacts as positions in
        `rows`, each contact twice, once from each end; or None when Qhull refuses the section.

        Every placed cell enters, typed or not and selected or not: a contact is a fact of the
        tissue, and leaving a cell out would make its neighbours touch across it."""
        s = int(s)
        if s in self.kept:
            return self.kept[s]
        rows = np.flatnonzero(self.has_place & (self.section == s))
        try:
            tri = Delaunay(self.coords[rows])
        except (QhullError, ValueError):    # too few cells, or all on one line or plane
            self.failed[s] = rows.size
            return None
        start, to = tri.vertex_neighbor_vertices
        origin = np.repeat(np.arange(rows.size), np.diff(start))
        # Qhull sets aside a cell stacked on another: it touches that one and no other.
        aside = tri.coplanar
        if aside.size:
            origin = np.concatenate((origin, aside[:, 0], aside[:, 2]))
            to = np.concatenate((to, aside[:, 2], aside[:, 0]))
        part = (rows, origin, to, int(aside.shape[0]))
        if s in self.in_run:
            self.kept[s] = part
        return part

    def name_of(self, s):
        return self.section_names.get(int(s)) or f"section {int(s) + 1}"


def _section_names(ds):
    """Each code of ds.section -> its name, matched as the SDK matches them: by `index` on a flat
    dataset, by position in the composition on an atlas bound to one."""
    composition = ds.meta.get("_composition")
    if composition:
        return {i: str(c.get("section_id") or "") for i, c in enumerate(composition)}
    return {int(s.get("index", i)): str(s.get("section_id") or "")
            for i, s in enumerate(ds.sections)}


def _colour(known, i):
    if re.fullmatch(r"#[0-9a-fA-F]{6}", str(known or "")):
        return str(known).lower()
    if i < len(PALETTE):
        return PALETTE[i]
    h = (i * 0.618034) % 1.0
    return _hls(h, (0.58, 0.7, 0.5)[i % 3], 0.75)


def _hls(h, l, s):
    def part(n):
        k = (n + h * 12) % 12
        return l - s * min(l, 1 - l) * max(-1.0, min(k - 3, 9 - k, 1.0))
    return "#%02x%02x%02x" % tuple(round(part(n) * 255) for n in (0, 8, 4))


class _Scale:
    """Lengths as the dataset's coordinates hold them, and as a person reads them."""

    def __init__(self, ds):
        self.per_micron = _units_per_micron(ds)
        self.units = str((ds.meta.get("spatial") or {}).get("units") or "coordinate units")

    def coordinates(self, value):
        return value if self.per_micron is None else value * self.per_micron

    def text(self, length):
        if self.per_micron is None:
            return f"{length:.3g} {self.units}"
        return f"{length / self.per_micron:.3g} µm"


def _units_per_micron(ds):
    """Coordinate units per µm, or None when the dataset does not say."""
    spatial = ds.meta.get("spatial") or {}
    units = spatial.get("units")
    if units in ("µm", "μm", "um", "micron", "microns"):
        return 1.0
    if units == "mm":
        return 1e-3
    # The rule the app has for the same question: an image layer that covers the cells.
    x, y = (spatial.get("ranges") or {}).get("x"), (spatial.get("ranges") or {}).get("y")
    for layer in (ds.meta.get("images") or {}).get("layers") or []:
        size, box = layer.get("microns_per_pixel"), layer.get("bounds")
        if (units == "px" and size and box and x and y
                and box[0] <= x[0] and x[1] <= box[2] and box[1] <= y[0] and y[1] <= box[3]):
            return 1.0 / float(size)
    return None


# ---- the contacts ---------------------------------------------------------------------------------
def _tally(tissue, part, among, limit=None):
    """C for one section, over the contacts between two cells of `among`, and those contacts' lengths
    once each."""
    rows, origin, to, _ = part
    K = tissue.n_types
    code = tissue.code[rows]
    ok = among[rows]
    keep = ok[origin] & ok[to]
    a, b = origin[keep], to[keep]
    points = tissue.coords[rows]
    once = a < b
    lengths = np.sqrt(((points[a[once]] - points[b[once]]) ** 2).sum(axis=1))
    if limit is not None:
        near = ((points[a] - points[b]) ** 2).sum(axis=1) <= limit * limit
        a, b = a[near], b[near]
    C = np.bincount(code[a] * K + code[b], minlength=K * K).reshape(K, K)
    return C, lengths


def _shares(C):
    """W: each row of C as shares of its total. A type with no contact has a row of zeros."""
    total = C.sum(axis=1, keepdims=True).astype(np.float64)
    return np.divide(C, total, out=np.zeros(C.shape, dtype=np.float64), where=total > 0)


def _enrichment(W, abundance):
    return np.log2((W + PSEUDO) / (abundance[None, :] + PSEUDO))


class _Pool:
    """W over sections: each row averaged, weighted by the cells of that type in each section. A
    section where the type is absent, or has no contact, says nothing about its row."""

    def __init__(self, K):
        self.total = np.zeros((K, K))
        self.weight = np.zeros(K)
        self.cells = np.zeros(K, dtype=np.int64)
        self.counts = np.zeros((K, K), dtype=np.int64)

    def add(self, C, cells):
        rows = (cells > 0) & (C.sum(axis=1) > 0)
        self.total[rows] += cells[rows, None] * _shares(C)[rows]
        self.weight[rows] += cells[rows]
        self.cells += cells
        self.counts += C

    def shares(self):
        return np.divide(self.total, self.weight[:, None], out=np.zeros_like(self.total),
                         where=self.weight[:, None] > 0)


def _count(tissue, scope, limit):
    """Everything about this run's cells: per section and pooled."""
    K = tissue.n_types
    pool = _Pool(K)
    sections, lengths, stacked = [], [], 0
    n_contacts_all = 0
    for s in np.unique(tissue.section[scope]):
        part = tissue.neighbours(s)
        if part is None:
            continue
        C, seen = _tally(tissue, part, scope, limit)
        cells = np.bincount(tissue.code[part[0][scope[part[0]]]], minlength=K)
        pool.add(C, cells)
        lengths.append(seen)
        stacked += part[3]
        n_contacts_all += seen.size
        sections.append({"section": int(s), "name": tissue.name_of(s), "C": C, "cells": cells})
    failed = [s for s in np.unique(tissue.section[scope]).tolist() if s in tissue.failed]
    if not sections:
        counts = sum(tissue.failed[s] for s in failed)
        names = ", ".join(tissue.name_of(s) for s in failed[:5])
        other = " Isolate another section." if len(tissue.section_names) > 1 else ""
        raise PluginError("no_triangulation",
                          f"No triangulation exists here: section {names} holds "
                          f"{_many(counts, 'cell')} with coordinates, too few, stacked on one "
                          f"another, or on one line or plane.{other}")
    W = pool.shares()
    abundance = pool.cells / pool.cells.sum()
    lengths = np.concatenate(lengths)
    notes = []
    if failed:
        left = sum(int((scope & (tissue.section == s)).sum()) for s in failed)
        names = ", ".join(tissue.name_of(s) for s in failed[:5])
        more = f" and {len(failed) - 5} more" if len(failed) > 5 else ""
        notes.append(f"Left out {_many(left, 'cell')} of {names}{more}: too few, stacked on one "
                     "another, or on one line or plane, so no triangulation exists there.")
    if tissue.n_no_place:
        notes.append(f"Left out {_many(tissue.n_no_place, 'cell')} with no coordinates.")
    if tissue.n_no_type:
        notes.append("In the triangulation, but with no contact counted: "
                     f"{_many(tissue.n_no_type, 'cell')} with no {tissue.a_level[2:]}.")
    if stacked:
        notes.append(f"Touching only the cell they sit on: {_many(stacked, 'cell')} stacked on "
                     "another.")
    silent = [tissue.names[i] for i in np.flatnonzero((pool.cells > 0) & (pool.weight == 0))]
    if silent:
        notes.append(f"{len(silent)} present type{'s have' if len(silent) > 1 else ' has'} no "
                     f"contact in this run{_named(silent)}.")
    return {"sections": sections, "W": W, "E": _enrichment(W, abundance), "abundance": abundance,
            "cells": pool.cells, "counts": pool.counts, "lengths": lengths, "notes": notes,
            "n_cells": int(pool.cells.sum()), "n_contacts": int(pool.counts.sum()) // 2,
            "n_contacts_all": n_contacts_all, "n_sections": len(sections)}


def _many(n, thing):
    return f"{n:,} {thing}" + ("" if n == 1 else "s")


def _named(names):
    return ": " + ", ".join(names) if len(names) <= 5 else ""


def _refuse_no_answer(tissue, found, limit_text, scale):
    """A run with no contact between two different groups has nothing to draw."""
    present = np.flatnonzero(found["cells"])
    many = _many(found["n_cells"], "cell")
    if present.size < 2:
        raise PluginError("one_type",
                          f"All {many} here are of one group, {tissue.names[int(present[0])]}, and a "
                          "contact network needs two. Widen the selection, isolate another section, "
                          f"or choose another column than {tissue.column}.")
    if not found["n_contacts_all"]:
        raise PluginError("no_contacts", f"None of these {many} touches another of them. Select "
                                         "more cells, close together.")
    if not found["n_contacts"]:
        raise PluginError("limit_removes_all",
                          f"This limit keeps none of the {_many(found['n_contacts_all'], 'contact')} "
                          f"here: the shortest is {scale.text(float(found['lengths'].min()))}. Raise "
                          "the longest contact, or set it to 0.")
    if not found["counts"][~np.eye(tissue.n_types, dtype=bool)].any():
        if limit_text != NO_LIMIT:
            raise PluginError("no_cross_contacts",
                              f"With {limit_text}, no two different {tissue.levels_text} touch. Raise "
                              "the longest contact, or set it to 0.")
        raise PluginError("no_cross_contacts",
                          f"No two different {tissue.levels_text} touch here: each section holds "
                          "one, or they lie apart. Choose a column with several groups in a section.")


def _most_abundant(cells, n):
    """The types to show, most cells first, ties by their order in the annotation."""
    order = np.lexsort((np.arange(cells.size), -cells))
    return order[cells[order] > 0][:n]


def _seriate(W):
    """An order for W's rows and columns that seats types of similar contacts together: hierarchical
    clustering of the symmetric (W + Wᵀ) / 2 with its leaves ordered, one order for both axes.

    Rows are compared by whom they touch, whatever their scale: each is scaled to unit length, and its
    own self-contact, which says how much a type keeps to itself and not whom it sits beside, counts
    no more than its largest other share. Plain distances between rows set a type that keeps to itself
    apart from its neighbours: on a colon section they left the smooth muscle more than 20 places from
    the types that give it 40% of their contacts."""
    if W.shape[0] < 3:
        return np.arange(W.shape[0])
    rows = (W + W.T) / 2
    np.fill_diagonal(rows, 0.0)
    np.fill_diagonal(rows, rows.max(axis=1))
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return leaves_list(linkage(pdist(rows), method="average", optimal_ordering=True))


# ---- the layout -----------------------------------------------------------------------------------
def _layout(ds, tissue, panel):
    """Each type's place in the panel, in px, (K, 2). It depends on the dataset and the annotation
    alone, so it is computed once: a hash of what it depends on names the file that keeps it."""
    key = hashlib.sha256(f"{LAYOUT_VERSION}|{tissue.names}".encode("utf-8"))
    placed = tissue.has_place
    key.update(tissue.coords[placed].tobytes())
    key.update(tissue.section[placed].tobytes())
    key.update(tissue.code[placed].tobytes())
    folder = os.path.join(getattr(ds, "dir", "") or "", "plugin_cache")
    path = os.path.join(folder, f"contact_network_layout_{key.hexdigest()[:32]}.npz")
    try:
        with np.load(path, allow_pickle=False) as kept:
            if kept["xy"].shape == (tissue.n_types, 2):
                return kept["xy"]
    except Exception:       # noqa: BLE001 — absent or damaged, it is computed again
        pass

    K = tissue.n_types
    pool = _Pool(K)
    typed = placed & (tissue.code >= 0)
    sections = np.unique(tissue.section[placed])
    for i, s in enumerate(sections):
        ds.progress(0.3 + 0.4 * i / sections.size, "Placing the cell types")
        part = tissue.neighbours(s)
        if part is not None:
            pool.add(_tally(tissue, part, typed)[0],
                     np.bincount(tissue.code[part[0][typed[part[0]]]], minlength=K))
    ds.progress(0.7, "Placing the cell types")
    xy = _node_places(_force_layout(pool.shares(), pool.cells), panel)
    if getattr(ds, "dir", None):
        temporary = f"{path}.{os.getpid()}.tmp.npz"
        try:
            os.makedirs(folder, exist_ok=True)
            np.savez(temporary, xy=xy)
            os.replace(temporary, path)
        except OSError:     # a read-only dataset: the next run computes the same layout
            try:
                os.remove(temporary)
            except OSError:
                pass
    return xy


def _force_layout(W, cells):
    """Places in [0, 1], from Fruchterman and Reingold's force-directed layout on the symmetric
    contacts: pairs that touch a lot attract, every pair repels.

    A force-directed layout settles in one of many arrangements, so several starts are settled and
    the one whose strong contacts cross the least is kept: on a 3D breast tumour of 17 types, the best
    of 8 has 47 crossings against 85 for the first. No random numbers: the starts are sunflowers of
    the types in orders given by arithmetic, and the same contacts give the same places."""
    K = W.shape[0]
    if K == 1:
        return np.array([[0.5, 0.5]])
    symmetric = (W + W.T) / 2
    np.fill_diagonal(symmetric, 0.0)
    rank = np.empty(K, dtype=np.int64)
    rank[np.lexsort((np.arange(K), -cells))] = np.arange(K)
    judged = np.argsort(rank)[:CROSS_TYPES]
    low, high = np.nonzero(np.triu(symmetric[np.ix_(judged, judged)] >= CROSS_SHARE, 1))
    # as many force evaluations as a second or two allow, the first thing to give at many types
    steps = int(max(60, min(300, LAYOUT_WORK / (K * K))))
    starts = int(max(1, min(8, LAYOUT_WORK / (steps * K * K))))
    multipliers = [a for a in (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29) if math.gcd(a, K) == 1]
    top = symmetric.max()
    best, fewest = None, None
    for i, a in enumerate(multipliers[:starts]):
        xy = _spread(_settle(symmetric / top if top > 0 else symmetric, (rank * a + i) % K, steps),
                     cells)
        crossing = _crossings(xy[judged], low, high)
        if fewest is None or crossing < fewest:
            best, fewest = xy, crossing
    return best


def _settle(strength, order, steps):
    """One run of the forces from a sunflower of the types in this order, fitted to [0, 1]."""
    K = order.size
    radius = np.sqrt((order + 0.5) / K)
    turn = order * 2.399963229728653
    xy = np.column_stack((radius * np.cos(turn), radius * np.sin(turn)))
    room = 1.0 / math.sqrt(K)
    for step in range(steps):
        away = xy[:, None, :] - xy[None, :, :]
        d2 = (away * away).sum(axis=2) + 1e-9
        force = room * room / d2 - strength * np.sqrt(d2) / room
        np.fill_diagonal(force, 0.0)
        move = (force[:, :, None] * away).sum(axis=1)
        move -= 0.5 * xy * np.sqrt((xy * xy).sum(axis=1, keepdims=True))    # holds it together
        size = np.sqrt((move * move).sum(axis=1, keepdims=True)) + 1e-12
        xy = xy + move / size * np.minimum(size, 0.25 * (1.0 - step / steps) + 1e-3)
    low, high = xy.min(axis=0), xy.max(axis=0)
    return (xy - low) / np.maximum(high - low, 1e-9)


def _spread(xy, cells):
    """Gives the abundant types more room, which is where a run has its nodes: along each axis each
    type's place moves halfway to its rank, each rank as wide as the type's √cells. The forces alone
    pack the abundant types into the middle and leave the rim to the rare."""
    wide = np.sqrt(cells / cells.max())
    out = np.empty_like(xy)
    for axis in (0, 1):
        order = np.argsort(xy[:, axis], kind="stable")
        rank = np.cumsum(wide[order]) - wide[order] / 2
        rank = (rank - rank[0]) / max(rank[-1] - rank[0], 1e-9)
        out[order, axis] = 0.5 * xy[order, axis] + 0.5 * rank
    return out


def _crossings(xy, low, high):
    """How many pairs of the straight lines between these places cross, sharing no end."""
    if low.size < 2:
        return 0
    i, j = np.triu_indices(low.size, 1)
    apart = (low[i] != low[j]) & (low[i] != high[j]) & (high[i] != low[j]) & (high[i] != high[j])
    i, j = i[apart], j[apart]

    def side(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    p, q, r, t = xy[low[i]], xy[high[i]], xy[low[j]], xy[high[j]]
    return int(((side(p, q, r) * side(p, q, t) < 0) & (side(r, t, p) * side(r, t, q) < 0)).sum())


def _node_places(placed, panel):
    """px positions: the layout fitted to the panel, its nodes pushed apart to the closest allowed."""
    margin_x, margin_y = 92.0, 72.0
    span = np.array([panel["width"] - 2 * margin_x, panel["height"] - 2 * margin_y])
    xy = placed * span
    closest = panel["closest"]
    for _ in range(400):
        away = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt((away * away).sum(axis=2))
        np.fill_diagonal(d, closest)
        near = d < closest
        if not near.any():
            break
        push = near * (closest - d) / (2 * np.maximum(d, 1e-6))
        xy = np.clip(xy + (push[:, :, None] * away).sum(axis=1), 0.0, span)
    return xy + (margin_x, margin_y)


# ---- colour and text ------------------------------------------------------------------------------
def _rgb(colour):
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def _e_colour(e):
    t = max(-1.0, min(1.0, float(e) / E_SPAN))
    stops = [s[1] for s in RAMP]
    parts = [np.interp(t, stops, [_rgb(s[0])[c] for s in RAMP]) for c in range(3)]
    return "#%02x%02x%02x" % tuple(int(round(p)) for p in parts)


def _luminance(colour):
    lin = [(c / 255) / 12.92 if c / 255 <= 0.03928 else ((c / 255 + 0.055) / 1.055) ** 2.4
           for c in _rgb(colour)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _safe(text):
    """Text as an SVG may hold it: escaped, and without what XML cannot carry at all."""
    return html.escape(re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", " ", str(text)))


def _text_px(text, size):
    """About how wide a label is set, erring wide: an SVG cannot measure text as it is written."""
    return size * (0.6 * len(text) + 0.2 * sum(c.isupper() or c in "mw@%" for c in text))


def _wrap(name, chars, lines=2):
    """A name on at most `lines` lines of about `chars` characters, broken after a space, an
    underscore, a slash or a comma, and cut with an ellipsis if it runs over."""
    out, line = [], ""
    for piece in re.findall(r"[^ _/,]+[ _/,]*|[ _/,]+", name):
        if line and len(line.rstrip()) + len(piece.rstrip()) > chars:
            out.append(line.rstrip())
            line = piece
        else:
            line += piece
    out.append(line.rstrip())
    cut = len(out) > lines
    out = out[:lines]
    if cut or len(out[-1]) > chars + 4:
        out[-1] = out[-1][:chars].rstrip(" ,·_/") + "…"
    return [s if len(s) <= chars + 4 else s[:chars] + "…" for s in out]


def _round_down(v):
    """The largest 1, 2 or 5 times a power of ten that is at most v."""
    if v <= 0:
        return 1
    power = 10.0 ** math.floor(math.log10(v))
    return next((m * power for m in (5, 2) if m * power <= v), power)


def _svg_head(width, height, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT}"><title>{_safe(title)}</title>'
            f'<rect width="{width}" height="{height}" fill="{GROUND}"/>')


def _figure_title(title, where):
    return (f'<text x="24" y="28" font-size="17" fill="{INK}">{_safe(title)}</text>'
            f'<text x="24" y="46" font-size="12.5" fill="{MUTED}">{_safe(where)}</text>')


def _ramp_key(x, y, text):
    """The ramp of E, with its name, its end values and what they mean. Returns the y below it."""
    stops = "".join(f'<stop offset="{(t + 1) / 2:g}" stop-color="{c}"/>' for c, t in RAMP)
    out = [f'<defs><linearGradient id="ramp">{stops}</linearGradient></defs>'
           f'<text x="{x}" y="{y}" font-size="14" fill="{INK}">E: log₂ enrichment</text>'
           f'<rect x="{x}" y="{y + 9}" width="{KEY_W}" height="14" fill="url(#ramp)"/>'
           f'<text x="{x}" y="{y + 39}" font-size="12.5" fill="{INK}">−{E_SPAN:g} or less</text>'
           f'<text x="{x + KEY_W / 2:.1f}" y="{y + 39}" font-size="12.5" fill="{INK}" '
           'text-anchor="middle">0</text>'
           f'<text x="{x + KEY_W}" y="{y + 39}" font-size="12.5" fill="{INK}" '
           f'text-anchor="end">+{E_SPAN:g} or more</text>'
           f'<text x="{x}" y="{y + 55}" font-size="12" fill="{MUTED}">depleted</text>'
           f'<text x="{x + KEY_W}" y="{y + 55}" font-size="12" fill="{MUTED}" '
           'text-anchor="end">enriched</text>']
    lines, y = _key_text(x, y + 75, text)
    out.append(lines)
    return "".join(out), y


def _where(ds, tissue, found, shown, limit_text):
    """The run in one line, above each figure."""
    whose = "selected cells" if ds.subset is not None and ds.subset.size else "cells"
    where = f"{found['n_cells']:,} {whose}"
    if found["n_sections"] > 1:
        where += f", {found['n_sections']} sections pooled"
    elif len(tissue.section_names) > 1:
        where += f", section {found['sections'][0]['name']}"
    present = int(np.count_nonzero(found["cells"]))
    kinds = (f"{present} types" if present <= shown.size
             else f"the {shown.size} most abundant of {present} types")
    return f"{where} · {kinds} · {limit_text}"


# ---- the network ----------------------------------------------------------------------------------
def _panel(K):
    """The network's panel for K types in the dataset: its size in px, and the closest two nodes may
    stand. It depends on K alone, so the panel is the same in every run on an annotation."""
    spacing = 165.0 * min(1.0, (30.0 / K) ** 0.3)
    area = K * spacing * spacing
    width = max(700.0, math.sqrt(area * 1.12))
    height = max(620.0, area / width)
    return {"width": width, "height": height,
            "closest": 0.8 * min(170.0, math.sqrt(width * height / K))}


def _node_sizes(xy, cells, self_share, panel):
    """How large this run's marks can be: the largest node, as a radius, such that no two rings meet,
    an arrow fits between any two nodes and every ring stays inside the panel. Node sizes follow this
    run's cells; places do not move."""
    k = xy.shape[0]
    away = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((away * away).sum(axis=2)) + np.eye(k) * 1e6
    scale = np.sqrt(cells / cells.max())
    room = np.minimum.reduce((xy[:, 0], panel["width"] - xy[:, 0],
                              xy[:, 1], panel["height"] - xy[:, 1]))

    def outer(radius):
        ring = RING_SHARE * radius
        return (np.maximum(MIN_NODE, radius * scale)
                + np.where(self_share > 0, RING_GAP + ring * self_share + 1.3, 0.0) + 2.0)

    low, high = 6.0, MAX_NODE
    for _ in range(20):
        mid = (low + high) / 2
        reach = outer(mid)
        if (reach[:, None] + reach[None, :] + ARROW_ROOM <= d).all() and (reach <= room).all():
            low = mid
        else:
            high = mid
    return low


def _curve_at(p, c, q, t):
    t = np.asarray(t)[..., None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * c + t * t * q


def _control(p, q, bend):
    """The control point of the curve from p to q that bulges to the right of travel: seen from
    above, the two directions of a pair pass each other like traffic, one on each side."""
    d = q - p
    return (p + q) / 2 + 2 * bend * np.array([-d[1], d[0]])


def _exit(p, c, q, centre, radius, leaving):
    """The t at which the curve crosses the circle around one of its ends."""
    lo, hi = (0.0, 0.5) if leaving else (0.5, 1.0)
    for _ in range(24):
        mid = (lo + hi) / 2
        inside = np.hypot(*(_curve_at(p, c, q, mid) - centre)) < radius
        if inside == leaving:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _arrow(p, q, bend, reach_p, reach_q, width, colour):
    """One arrow from the edge of p's outer circle to the edge of q's, the tip of its head on q's.
    Returns its SVG and the points it passes, to keep labels off it."""
    c = _control(p, q, bend)
    start_t = _exit(p, c, q, p, reach_p, True)
    tip_t = _exit(p, c, q, q, reach_q, False)
    if tip_t <= start_t:
        return "", np.empty((0, 2))
    tip = _curve_at(p, c, q, tip_t)
    length = HEAD_BASE + HEAD_PER_WIDTH * width
    lo, hi = start_t, tip_t             # where the line ends and the head begins
    for _ in range(24):
        mid = (lo + hi) / 2
        if np.hypot(*(_curve_at(p, c, q, mid) - tip)) > length:
            lo = mid
        else:
            hi = mid
    base_t = lo
    start, base = _curve_at(p, c, q, start_t), _curve_at(p, c, q, base_t)
    along = (tip - base) / max(np.hypot(*(tip - base)), 1e-9)
    side = np.array([-along[1], along[0]]) * (HEAD_HALF + HEAD_HALF_PER_WIDTH * width)
    # the piece of the curve between start_t and base_t, as a curve of its own
    mid_c = ((1 - start_t) * (1 - base_t) * p
             + ((1 - start_t) * base_t + start_t * (1 - base_t)) * c + start_t * base_t * q)
    path = f"M{start[0]:.1f} {start[1]:.1f}Q{mid_c[0]:.1f} {mid_c[1]:.1f} {base[0]:.1f} {base[1]:.1f}"
    head = " ".join(f"{v[0]:.1f},{v[1]:.1f}" for v in (tip, base + side, base - side))
    svg = (f'<path d="{path}" stroke="{GROUND}" stroke-width="{width + CASING:.1f}"/>'
           f'<path d="{path}" stroke="{colour}" stroke-width="{width:.1f}"/>'
           f'<polygon points="{head}" fill="{colour}" stroke="{GROUND}"/>')
    along_curve = _curve_at(p, c, q, np.linspace(start_t, tip_t, 2 + int(np.hypot(*(q - p)) / 8)))
    return svg, np.vstack((along_curve, np.repeat(tip[None, :], HEAD_WEIGHT, axis=0)))


def _bend(p, q, ends, others, reach, width):
    """How far an arrow bulges: the least that keeps it off other nodes, or the one that passes
    through the fewest, and always enough to clear the line between its two nodes, where the arrow of
    the other direction bulges to its own side: at the middle, and where it leaves the smaller node."""
    clear = max((width / 2 + 2.0) / max(np.hypot(*(q - p)), 1.0), (width / 2 + 1.0) / (4.0 * min(ends)))
    clear = min(clear, 0.45)
    best, fewest = None, None
    for bend in BENDS:
        bend = max(bend, clear)
        line = _curve_at(p, _control(p, q, bend), q, np.linspace(0.05, 0.95, 24))
        hits = np.hypot(line[:, None, 0] - others[:, 0], line[:, None, 1] - others[:, 1]) < reach
        count = int(hits.any(axis=1).sum())
        if fewest is None or count < fewest:
            best, fewest = bend, count
        if not count:
            break
    return best


def _network_figure(frame):
    """The network as one SVG document, and how many arrows it draws."""
    tissue, found, shown, cutoff = frame["tissue"], frame["run"], frame["shown"], frame["cutoff"]
    W, E, cells = found["W"], found["E"], found["cells"]
    panel, xy = frame["panel"], frame["xy"]
    self_share = np.diag(W)[shown]
    largest = _node_sizes(xy[shown], cells[shown], self_share, panel)
    ring = RING_SHARE * largest
    stroke = float(np.clip(STROKE_SHARE * largest, 4.0, MAX_STROKE))
    biggest = float(cells[shown].max())

    radius = np.zeros(tissue.n_types)
    radius[shown] = np.maximum(MIN_NODE, largest * np.sqrt(cells[shown] / biggest))
    circle = np.zeros(tissue.n_types)
    circle[shown] = np.where(self_share > 0, radius[shown] + RING_GAP + ring * self_share, 0.0)
    reach = np.maximum(radius, circle) + np.where(circle > 0, 1.3, 0.0) + 2.0

    # thick arrows first, so that the thin ones stay in sight
    pairs = [(int(i), int(j)) for i in shown for j in shown if i != j and W[i, j] >= cutoff]
    pairs.sort(key=lambda ij: (-W[ij], ij))
    arrows, passes = [], []
    for i, j in pairs:
        others = (shown != i) & (shown != j)
        width = max(1.0, stroke * math.sqrt(W[i, j]))
        bend = _bend(xy[i], xy[j], (reach[i], reach[j]), xy[shown[others]],
                     reach[shown[others]] + 3.0, width)
        svg, points = _arrow(xy[i], xy[j], bend, reach[i], reach[j], width, _e_colour(E[i, j]))
        arrows.append(svg)
        passes.append(points)
    passes = np.concatenate(passes) if passes else np.empty((0, 2))

    nodes = []
    for k in shown.tolist():
        x, y = xy[k]
        if circle[k] > 0:
            nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{circle[k]:.1f}" fill="none" '
                         f'stroke="{GROUND}" stroke-width="5"/><circle cx="{x:.1f}" cy="{y:.1f}" '
                         f'r="{circle[k]:.1f}" fill="none" stroke="{_e_colour(E[k, k])}" '
                         'stroke-width="2.6"/>')
        nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius[k]:.1f}" '
                     f'fill="{tissue.colours[k]}" stroke="{INK}"/>')
    labels = _node_labels(tissue, shown, xy, reach, passes, panel)

    where = f'{frame["where"]} · arrows from {100 * cutoff:g}%'
    if not pairs:
        between = W[np.ix_(shown, shown)] * (1 - np.eye(shown.size))
        where += (" · none reaches it: the largest share between two types is "
                  f"{100 * between.max():.3g}%")
    key_x = int(panel["width"]) + 22
    key, key_bottom = _network_key(key_x, HEAD_H + 8, biggest, largest, ring, stroke)
    width = key_x + KEY_W + 20
    height = int(math.ceil(max(panel["height"] + HEAD_H, key_bottom + 16)))
    title = "Cell-type contact network"
    return (_svg_head(width, height, title) + _figure_title(title, where)
            + f'<g transform="translate(0 {HEAD_H})"><g fill="none">{"".join(arrows)}</g>'
            + f'{"".join(nodes)}{labels}</g>{key}</svg>'), len(pairs)


def _node_labels(tissue, shown, xy, reach, passes, panel):
    """Each shown type's name beside its node, where it covers the least, with a halo of the ground
    so that it stays legible over the arrows that do pass under it."""
    centres, reaches = xy[shown], reach[shown]
    taken, out = [], []
    for k in shown[np.argsort(-reach[shown], kind="stable")].tolist():
        lines = _wrap(tissue.names[k], LABEL_CHARS)
        w = max(_text_px(s, LABEL_FONT) for s in lines)
        h = LABEL_LINE * len(lines)
        x, y, r = xy[k, 0], xy[k, 1], reach[k] + 2.0
        d = r * 0.72
        spots = ((x - w / 2, y + r, 0.0), (x - w / 2, y - r - h, 0.2), (x + r, y - h / 2, 0.5),
                 (x - r - w, y - h / 2, 0.5), (x + d, y + d, 0.7), (x - d - w, y + d, 0.7),
                 (x + d, y - d - h, 0.7), (x - d - w, y - d - h, 0.7))
        best, least = None, None
        for left, top, cost in spots:
            right, bottom = left + w, top + h
            if left < 4 or top < 2 or right > panel["width"] - 2 or bottom > panel["height"] - 2:
                cost += 40.0
            nearest = np.column_stack((np.clip(centres[:, 0], left, right),
                                       np.clip(centres[:, 1], top, bottom)))
            cost += 6.0 * int((np.hypot(*(centres - nearest).T) < reaches).sum())
            cost += 12.0 * sum(left < a[2] and a[0] < right and top < a[3] and a[1] < bottom
                               for a in taken)
            if passes.size:
                cost += 0.2 * int(((passes[:, 0] > left) & (passes[:, 0] < right)
                                   & (passes[:, 1] > top) & (passes[:, 1] < bottom)).sum())
            if least is None or cost < least:
                best, least = (left, top), cost
        left, top = best
        taken.append((left, top, left + w, top + h))
        spans = "".join(f'<tspan x="{left + w / 2:.1f}" y="{top + LABEL_LINE * (i + 0.8):.1f}">'
                        f'{_safe(s)}</tspan>' for i, s in enumerate(lines))
        out.append(f'<text font-size="{LABEL_FONT}" fill="{INK}" text-anchor="middle" '
                   f'stroke="{GROUND}" stroke-width="3" paint-order="stroke" '
                   f'stroke-linejoin="round">{spans}</text>')
    return "".join(out)


def _key_text(x, y, text, size=12.0, fill=MUTED, chars=38):
    """Muted lines of text in the key's column. Returns the SVG and the y below them."""
    out = []
    for line in _wrap(text, chars, 6):
        out.append(f'<text x="{x}" y="{y}" font-size="{size:g}" fill="{fill}">{_safe(line)}'
                   '</text>')
        y += 16
    return "".join(out), y


def _network_key(x, y, biggest, largest, ring, stroke):
    """What the marks mean: colour, width, ring and size, each with its name and scale."""
    gap = 26
    out, y = _ramp_key(x, y + 8, "Colour of an arrow or a ring. Above 0, type i touches j more than "
                                 "the abundance of j predicts.")
    out = [out]
    y += gap - 6
    out.append(f'<text x="{x}" y="{y}" font-size="14" fill="{INK}">Arrow i → j: W</text>')
    text, y = _key_text(x, y + 18, "The share of i's contacts that are with j, its width growing as "
                                   "√W. Each direction has its own arrow.")
    out.append(text)
    y += 4
    for share in (0.05, 0.2, 0.5):
        w = max(1.0, stroke * math.sqrt(share))
        head = HEAD_BASE + HEAD_PER_WIDTH * w
        half = HEAD_HALF + HEAD_HALF_PER_WIDTH * w
        y += max(10.0, half)
        out.append(f'<path d="M{x} {y:.1f}h{70 - head:.1f}" stroke="{INK}" stroke-width="{w:.1f}"/>'
                   f'<polygon points="{x + 70},{y:.1f} {x + 70 - head:.1f},{y - half:.1f} '
                   f'{x + 70 - head:.1f},{y + half:.1f}" fill="{INK}"/>'
                   f'<text x="{x + 82}" y="{y + 4.5:.1f}" font-size="12.5" fill="{INK}">'
                   f'{100 * share:.0f}%</text>')
        y += max(10.0, half)
    y += gap
    out.append(f'<text x="{x}" y="{y}" font-size="14" fill="{INK}">Ring: W[i, i]</text>')
    text, y = _key_text(x, y + 18, "The share of i's contacts that are within i: the wider the ring, "
                                   "the larger the share.")
    out.append(text)
    disc = min(13.0, 0.5 * largest)
    circles = [(share, disc + RING_GAP + ring * share) for share in (0.1, 0.5, 0.9)]
    widest = circles[-1][1]
    cy = y + widest - 6
    cx = x
    for share, r in circles:
        out.append(f'<circle cx="{cx + r + 1.3:.1f}" cy="{cy:.1f}" r="{disc:g}" fill="{MUTED}"/>'
                   f'<circle cx="{cx + r + 1.3:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" '
                   f'stroke="{INK}" stroke-width="2.6"/><text x="{cx + r + 1.3:.1f}" '
                   f'y="{cy + widest + 18:.1f}" font-size="12.5" fill="{INK}" text-anchor="middle">'
                   f'{100 * share:.0f}%</text>')
        cx += 2 * r + 20
    y = cy + widest + 18 + gap + 4
    out.append(f'<text x="{x}" y="{y}" font-size="14" fill="{INK}">Node: number of cells</text>')
    text, y = _key_text(x, y + 18, "Its area grows with the cells of the type in this run. Its colour "
                                   "is the type's on the map.")
    out.append(text)
    counts = [_round_down(biggest)]
    if counts[0] >= 8:
        counts.append(_round_down(counts[0] / 8))
    sizes = [max(MIN_NODE, largest * math.sqrt(c / biggest)) for c in counts]
    cy = y + sizes[0] - 6
    cx = x
    for count, r in zip(counts, sizes):
        width = max(2 * r, _text_px(f"{count:,.0f}", 12.5))
        out.append(f'<circle cx="{cx + width / 2:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{MUTED}"/>'
                   f'<text x="{cx + width / 2:.1f}" y="{cy + sizes[0] + 18:.1f}" font-size="12.5" '
                   f'fill="{INK}" text-anchor="middle">{count:,.0f}</text>')
        cx += width + 26
    return "".join(out), cy + sizes[0] + 18


# ---- the matrix -----------------------------------------------------------------------------------
def _matrix_figure(frame, order):
    """W and E with no cutoff, in one order on both axes, as one SVG document."""
    tissue, found = frame["tissue"], frame["run"]
    W, E = found["W"], found["E"]
    n = order.size
    cell = max(15, min(30, 760 // n))
    font = max(10.5, min(13.0, cell * 0.62))
    names = [_wrap(tissue.names[k], 34, 1)[0] for k in order]
    longest = max(_text_px(s, font) for s in names)
    left = max(int(longest) + 22, 40)
    top = HEAD_H + int(longest * 0.8) + 10
    size = n * cell
    key_x = left + size + 28
    width = key_x + KEY_W + 20
    numbered = cell >= 18
    number_font = max(9.0, min(11.5, cell * 0.44))
    where = frame["where"] + " · no cutoff"
    out = [_svg_head(width, 0, "Cell-type contact matrix"),
           _figure_title("Cell-type contact matrix", where)]
    for r, i in enumerate(order.tolist()):
        for c, j in enumerate(order.tolist()):
            x, y = left + c * cell, top + r * cell
            fill = _e_colour(E[i, j]) if W[i, j] > 0 else NO_CONTACT
            out.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{fill}"/>')
            if numbered and W[i, j] >= 0.01:
                ink = GROUND if _luminance(fill) > 0.28 else "#ffffff"
                out.append(f'<text x="{x + (cell - 1) / 2:.1f}" '
                           f'y="{y + (cell - 1) / 2 + number_font * 0.36:.1f}" '
                           f'font-size="{number_font:.1f}" fill="{ink}" text-anchor="middle">'
                           f'{100 * W[i, j]:.0f}</text>')
    for d in range(n):
        x, y = left + d * cell, top + d * cell
        out.append(f'<rect x="{x - 0.5}" y="{y - 0.5}" width="{cell}" height="{cell}" fill="none" '
                   f'stroke="{INK}" stroke-width="1.6"/>')
    for r, name in enumerate(names):
        safe = _safe(name)
        out.append(f'<text x="{left - 7}" y="{top + r * cell + cell / 2 + font * 0.32:.1f}" '
                   f'font-size="{font:.1f}" fill="{INK}" text-anchor="end">{safe}</text>')
        out.append(f'<text transform="translate({left + r * cell + cell * 0.62:.1f} {top - 6}) '
                   f'rotate(-52)" font-size="{font:.1f}" fill="{INK}">{safe}</text>')

    key, y = _ramp_key(key_x, top + 8, "Above 0, the row's type touches the column's more than the "
                                       "column's abundance predicts.")
    out.append(key)
    y += 12
    out.append(f'<rect x="{key_x}" y="{y - 12}" width="16" height="16" fill="{NO_CONTACT}"/>'
               f'<text x="{key_x + 24}" y="{y + 1}" font-size="12.5" fill="{INK}">Grey: no contact, '
               'W = 0</text>')
    y += 24
    out.append(f'<rect x="{key_x + 0.5}" y="{y - 12}" width="16" height="16" fill="none" '
               f'stroke="{INK}" stroke-width="1.6"/><text x="{key_x + 24}" y="{y + 1}" '
               f'font-size="12.5" fill="{INK}">Outlined: self-contact</text>')
    y += 28
    numbers = ("Numbers: W, the row's share of contacts with the column, in %, from 1%."
               if numbered else "W is in the table: these cells are too small for numbers.")
    for line in (numbers, "A row shows what that type touches. A column shows what touches it.",
                 "Each row of W sums to 100% over all types, shown or not."):
        text, y = _key_text(key_x, y, line)
        out.append(text)
        y += 6
    height = int(math.ceil(max(top + size + 20, y + 10)))
    out[0] = _svg_head(width, height, "Cell-type contact matrix")
    out.append("</svg>")
    return "".join(out)


# ---- the numbers ----------------------------------------------------------------------------------
def _table(tissue, found, order):
    W, E, C = found["W"], found["E"], found["counts"]
    rows = [[tissue.names[i], tissue.names[j], round(100 * float(W[i, j]), 2),
             round(float(E[i, j]), 2), int(C[i, j])]
            for i in order.tolist() for j in order.tolist() if W[i, j] > 0]
    rows.sort(key=lambda r: (-r[3], r[0], r[1]))
    return {"title": "Contacts between the types shown", "sort_by": "E",
            "columns": ["From", "To", "W (%)", "E", "C"], "rows": rows}


def _downloads(tissue, found, xy, order, budget, notes):
    """The numbers of this run, for every type: one row per ordered pair in contact, and one per type.
    Several sections add their own rows beside the pooled ones, as long as they fit."""
    def pair_rows(scope, W, E, C):
        out = []
        for i, j in zip(*np.nonzero(W)):
            out.append((scope, tissue.names[i], tissue.names[j], int(C[i, j]),
                        f"{W[i, j]:.8g}", f"{E[i, j]:.5f}"))
        return out

    def type_rows(scope, cells, abundance, C):
        return [(scope, tissue.names[k], int(cells[k]), f"{abundance[k]:.8g}", int(C[k].sum()),
                 f"{xy[k, 0]:.2f}", f"{xy[k, 1]:.2f}") for k in np.flatnonzero(cells)]

    pooled = "pooled" if found["n_sections"] > 1 else found["sections"][0]["name"]
    pairs = pair_rows(pooled, found["W"], found["E"], found["counts"])
    types = type_rows(pooled, found["cells"], found["abundance"], found["counts"])
    pair_head = ("section", "from", "to", "C", "W", "E")
    type_head = ("section", "type", "cells", "fraction", "contacts", "network_x", "network_y")
    if found["n_sections"] > 1:
        extra_pairs, extra_types = [], []
        for part in found["sections"]:
            share = part["cells"] / max(1, part["cells"].sum())
            W = _shares(part["C"])
            extra_pairs += pair_rows(part["name"], W, _enrichment(W, share), part["C"])
            extra_types += type_rows(part["name"], part["cells"], share, part["C"])
        whole = _csv_file("contact-matrices", pair_head, pairs + extra_pairs)
        kinds = _csv_file("contact-types", type_head, types + extra_types)
        if _wire(whole) + _wire(kinds) <= budget:
            return [whole, kinds]
        notes.append("The downloads hold the pooled numbers: those of each section would not fit "
                     "the output limit. Isolate a section for its own.")
    files = [_csv_file("contact-matrices", pair_head, pairs),
             _csv_file("contact-types", type_head, types)]
    if sum(_wire(f) for f in files) > budget:
        notes.append("The numbers would not fit the output limit, so the downloads hold the types "
                     "shown.")
        shown = set(order.tolist())
        names = {tissue.names[k] for k in shown}
        files = [_csv_file("contact-matrices", pair_head,
                           [r for r in pairs if r[1] in names and r[2] in names]),
                 _csv_file("contact-types", type_head, [r for r in types if r[1] in names])]
    return files


def _wire(file):
    return len(file["base64"])


def _csv_file(stem, head, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(head)
    writer.writerows(rows)
    text = buffer.getvalue().encode("utf-8", "replace")
    if len(text) <= PLAIN_CSV:
        return download_file(f"{stem}.csv", text, media_type="text/csv")
    return download_file(f"{stem}.csv.gz", gzip.compress(text, compresslevel=6, mtime=0),
                         media_type="application/gzip")


def _summary(ds, tissue, found, shown, drawn, cutoff, limit, limit_text, scale, notes):
    W, E = found["W"], found["E"]
    present = int(np.count_nonzero(found["cells"]))
    whose = "selected cell" if ds.subset is not None and ds.subset.size else "cell"
    where = _many(found["n_sections"], "section")
    arrows_text = "1 arrow reaches" if drawn == 1 else f"{drawn:,} arrows reach"
    lines = [f"**{_many(present, 'type')}, {_many(found['n_cells'], whose)}, "
             f"{_many(found['n_contacts'], 'contact')}** in {where}, {limit_text}. {arrows_text} "
             f"{100 * cutoff:g}%."]
    lengths = found["lengths"]
    middle = float(np.median(lengths))
    if limit is None:
        long = float((lengths > 3 * middle).mean())
        lines.append(f"The median contact is {scale.text(middle)}; {100 * long:.1f}% are longer than "
                     f"{scale.text(3 * middle)}, three medians, and may cross gaps.")
    else:
        dropped = 1 - found["n_contacts"] / found["n_contacts_all"]
        lines.append(f"The median contact is {scale.text(middle)}; the limit drops "
                     f"{100 * dropped:.1f}% of the {_many(found['n_contacts_all'], 'contact')}.")
    between = np.zeros(W.shape, dtype=bool)
    between[np.ix_(shown, shown)] = True
    between &= (W >= cutoff) & ~np.eye(tissue.n_types, dtype=bool)
    extremes = []
    if between.any():
        high = np.unravel_index(np.argmax(np.where(between, E, -np.inf)), E.shape)
        low = np.unravel_index(np.argmin(np.where(between, E, np.inf)), E.shape)
        for pair, word, wanted in ((high, "enriched", E[high] > 0), (low, "depleted", E[low] < 0)):
            if wanted:
                extremes.append(f"Most {word} arrow: **{tissue.names[pair[0]]} → "
                                f"{tissue.names[pair[1]]}** (E = {E[pair]:+.2f}, W = "
                                f"{100 * W[pair]:.1f}%).")
    if extremes:
        lines.append(" ".join(extremes))
    if notes:
        lines.append(" ".join(notes))
    return "\n\n".join(lines)
