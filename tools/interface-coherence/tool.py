"""Interface coherence: do two populations meet along one front, or in specks?

What this computes
------------------
Not how OFTEN two populations touch — `celltype-cooccurrence` answers that —
but the SHAPE of the contact. The interface is constructed, not counted, as the
dual of the Delaunay triangulation of the cells:

  * every Delaunay edge whose two endpoints carry different labels is an A|B
    contact, and the Voronoi facet dual to that edge is a line segment with a
    real length. Divided by the LOCAL cell diameter (the mean surviving incident
    edge length at its two endpoints), it becomes a dimensionless count of cell
    widths, and L_AB = the sum of those is a boundary LENGTH in cell diameters,
    not a pair count.
  * the facets join up. Two are contiguous exactly when their Delaunay edges
    share a triangle AND carry the same unordered label pair; a triangle with
    three distinct labels is a triple junction where the A|B curve ends. A
    triangle holds at most two facets of one pair, so every connected component
    of that relation is a simple path or a cycle — a SEGMENT of boundary.
  * S_AB is how many separate segments there are, and

        Lw_AB = sum(s^2) / sum(s)

    is the length-weighted mean segment length: the stretch of boundary that a
    random point ON the interface belongs to, in cell diameters. The weighting
    is the point. A plain mean is dominated by the specks — a front of 200
    diameters flanked by fifty 1-diameter contacts averages to 5 — while the
    length-weighted mean answers the question a reader is actually asking.

Why the ruler is local
----------------------
A facet is only meaningful as a multiple of the cell size where it sits. One
median over the whole run is not that: on datasets/co_m3_r2 the mean incident
edge length spans 17.9-55.4 units between the 1st and 99th percentile of
surviving edges — 3.1x inside ONE section, and 10.1x before pruning. With a
single global ruler, two triangular-lattice fronts of identical shape at a 3x
density ratio placed in one section scored 2.16x apart (1.112 log2); with the
local ruler they score 1.15x apart (0.201 log2), and measured in isolation the
two return Lw = 32.9089653 to 13 significant figures. On the colon the pair
ranking's rank correlation with the mean local density of its two types fell
from -0.608 to -0.383 for a ranking that otherwise moved by Spearman +0.934.
The residual is the edge PRUNING threshold, which stays global on purpose: a
local prune would license exactly the boundaries across a lumen that pruning
exists to prevent.

Why the null is a label shuffle, and why the ratio is what is reported
----------------------------------------------------------------------
Lw is confounded by abundance: shuffle the labels and two abundant populations
still produce longer runs of coincidental agreement than two rare ones. Under a
complete shuffle every contact is isolated and Lw collapses towards one cell
diameter, at a rate that depends on both abundances — which is exactly the term
that has to be divided out. So the reported index is

        coherence C_AB = log2( Lw_observed / Lw_shuffled )

C > 0: the pair meets along extended fronts. C ~ 0: scattered isolated
contacts, however many of them there are.

That null destroys ALL spatial structure, so on real tissue most pairs beat it
(372 of 460 on the colon) and the absolute ratio is not a test — the RANKING is
what the tool is for. `log2 contact amount` in the table is the plain length
ratio (the ICI of the source note); it is kept as a labelled reference column
beside the shape, and it largely restates the adjacency enrichment
`celltype-cooccurrence` already ships (Spearman +0.856 on the colon, against
+0.102 for coherence itself). README.md gives the full correlation table.

What was tried and dropped
--------------------------
Interface FRACTALITY, by two estimators, both measured on a 135k-cell colon
section and both rejected. Richardson coarse-graining (bin A and B at pitch l,
majority-label each bin, measure the boundary between opposite-majority bins):
in a 73-type tissue any one pair occupies a small part of space and third types
sit between them, so L(l) RISES with l and the fitted dimension lands at
0.48-1.84, median 0.92 — below 1, impossible for a curve. It measures bin
occupancy. Box-counting on the facets themselves is well posed, but the facet
sets are 300-8,000 points, so N(l) saturates against the sample size and the
fitted D tracks the facet count (0.36 for a 336-facet pair, 1.15 for an
8,037-facet one) rather than the shape. Segment tortuosity (arc length over
bounding-box diagonal) IS well behaved but is a monotone restatement of Lw, so
it adds a column and no information.

Cost
----
One Delaunay per section plus one connected-components pass per label shuffle.
Measured on datasets/co_m3_r2 (135,232 cells, 73 types, one section) through the
production subprocess path, median of 3, shipped defaults: 1.8-2.1 s of tool
time across two batches (2.2-2.4 s wall), of which 0.79 s is the triangulation
and 0.48 s the 19 shuffles. The worst corner a user can reach through the form
is 3.8-3.9 s. Peak RSS 205-208 MB. Nothing here is
BLAS-bound: medians of 1.80 s at one thread and 1.77 s at eight.
"""

from __future__ import annotations

import numpy as np

from pointillviz_tool import PluginError

#: 100 x 100 is exactly the result contract's 10,000-cell matrix ceiling, so
#: this is where an annotation too fine to draw is refused rather than silently
#: truncated into a matrix that omits the pairs a reader came for.
MAX_LEVELS = 100
#: Points per survival curve; five curves, well inside the 5,000-point ceiling.
CURVE_POINTS = 60
#: Pairs drawn in the segment-length panel, focus pair first.
CURVE_LINES = 4
#: The table ceiling is 5,000 rows. At min_interface = 0 the 73-type colon
#: section reports 2,059 pairs and a finer annotation reports more, so the cap
#: bites only where the tail is noise anyway.
MAX_TABLE_PAIRS = 4000
#: A shuffle count below this is not a null, it is one draw of a random object.
MIN_SHUFFLES = 2
#: Below either of these the leading pair is a ratio computed from a few
#: segments, and the summary says so rather than letting it head the ranking
#: unqualified.
THIN_SEGMENTS = 10
THIN_DIAMETERS = 20.0
#: Ratio of the 99th to the 1st percentile local cell diameter above which the
#: summary says the section's packing is uneven. Facets are normalised by that
#: local diameter, but the edge PRUNING distance is one global number, so a
#: residual density term survives and the reader should know it is there. The
#: 135k-cell colon section sits at 3.1x.
DIAM_SPREAD = 3.0

_CURVE_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
_NULL_COLOR = "#94a3b8"


# ===========================================================================
# Geometry
# ===========================================================================
def _section_geometry(points):
    """(ei, ej, facet_length, edge_length, tri_edges) for one section.

    `tri_edges` is (n_triangles, 3) of edge ids, which is what turns facet
    contiguity — "these two facets share a triangle" — into an array operation
    instead of a walk over a half-edge structure.
    """
    from scipy.spatial import Delaunay, QhullError
    try:
        tri = Delaunay(points)
    except (QhullError, ValueError):
        return None                      # collinear or degenerate: no triangles
    simplices = tri.simplices.astype(np.int64)
    n_tri = len(simplices)
    if n_tri == 0:
        return None
    n = len(points)

    # Circumcentres by the determinant form. The Voronoi vertex of a triangle
    # IS its circumcentre, so the facet dual to an interior edge is the segment
    # between the circumcentres of the two triangles sharing it — no scipy
    # Voronoi object, and no second triangulation of the same points.
    a, b, c = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]
    den = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / den
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / den
    centres = np.stack([ux, uy], axis=1)

    ends_i = np.empty((n_tri, 3), dtype=np.int64)
    ends_j = np.empty((n_tri, 3), dtype=np.int64)
    for k in range(3):                       # side k is the one opposite vertex k
        ends_i[:, k] = simplices[:, (k + 1) % 3]
        ends_j[:, k] = simplices[:, (k + 2) % 3]
    key = (np.minimum(ends_i, ends_j) * np.int64(n)
           + np.maximum(ends_i, ends_j)).ravel()
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(n_tri, 3)
    n_edges = uniq.size
    ei, ej = uniq // n, uniq % n

    tri_of = np.repeat(np.arange(n_tri, dtype=np.int64), 3)
    flat = inv.ravel()
    order = np.argsort(flat, kind="stable")
    tri_sorted = tri_of[order]
    count = np.bincount(flat, minlength=n_edges)
    starts = np.concatenate([[0], np.cumsum(count)[:-1]])
    first = tri_sorted[starts]
    second = tri_sorted[np.minimum(starts + 1, tri_sorted.size - 1)]
    # A hull edge borders one triangle, so its facet runs to infinity and has no
    # length this tool can honestly quote. It contributes zero and is not an
    # interface: the outside of the section is not a cell type.
    inner = count == 2
    facet = np.zeros(n_edges, dtype=np.float64)
    if inner.any():
        facet[inner] = np.linalg.norm(centres[first[inner]] - centres[second[inner]], axis=1)
    facet[~np.isfinite(facet)] = 0.0
    return ei, ej, facet, np.linalg.norm(points[ei] - points[ej], axis=1), inv


def _build_graph(coords, section, eligible, edge_prune):
    """One triangulation PER SECTION, concatenated into a single edge list.

    Per section for the reason the rest of the codebase builds neighbour graphs
    that way: sections share one canvas and two of them can overlap in
    coordinate space, so a single triangulation would draw a boundary between
    two different specimens.

    Every facet is returned already divided by a LOCAL cell diameter — the mean
    surviving Delaunay edge length at its two endpoints — so `w` is a count of
    cell diameters, not a length in dataset units. See `_local_diameter`.
    """
    ei_all, ej_all, facet_all, elen_all, tri_all = [], [], [], [], []
    offset = 0
    for s in np.unique(section[eligible]):
        rows = np.flatnonzero(eligible & (section == s))
        if rows.size < 4:
            continue
        geom = _section_geometry(coords[rows].astype(np.float64))
        if geom is None:
            continue
        ei, ej, facet, elen, inv = geom
        ei_all.append(rows[ei])
        ej_all.append(rows[ej])
        facet_all.append(facet)
        elen_all.append(elen)
        tri_all.append(inv + offset)
        offset += facet.size
    if not ei_all:
        return None
    facet = np.concatenate(facet_all)
    elen = np.concatenate(elen_all)
    ei = np.concatenate(ei_all)
    ej = np.concatenate(ej_all)
    spacing = float(np.median(elen))
    if not np.isfinite(spacing) or spacing <= 0:
        return None
    longest = edge_prune * spacing
    keep = elen <= longest
    # Dropping the long EDGES stops a boundary being invented across a crypt
    # lumen or a tear in the section, where Delaunay will happily connect the
    # two sides. This one threshold is global on purpose: "longer than three
    # median spacings" is a statement about the section, and deriving it from
    # the local spacing it is used to compute would be circular.
    cell_diam, edge_diam = _local_diameter(len(coords), ei, ej, elen, keep, spacing)
    # Clamping the facet at `edge_prune` diameters stops a sliver triangle at
    # the tissue border — whose circumcentre flies far outside the tissue —
    # contributing a kilometre of "boundary".
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.minimum(facet / edge_diam, edge_prune)
    weight = np.where(keep & np.isfinite(scaled), scaled, 0.0)
    return {"ei": ei, "ej": ej, "w": weight, "keep": keep,
            "tri_edges": np.concatenate(tri_all, axis=0),
            "spacing": spacing, "cell_diam": cell_diam,
            "n_edges": facet.size, "n_kept": int(keep.sum())}


def _local_diameter(n_cells, ei, ej, elen, keep, spacing):
    """The ruler, per cell and per edge: mean surviving incident edge length.

    A facet is only meaningful as a multiple of the cell size where it sits, and
    one median over the whole run is not that. On the 135k-cell colon section
    this mean spans 17.9 to 55.4 units between the 1st and 99th percentile of
    surviving edges — 3.1x inside ONE section, crypt lumen against lamina
    propria, and 10.1x before pruning — so a global median makes two
    geometrically identical fronts at a 3x density ratio score 2.16x apart.
    Dividing each facet by the local diameter first drops that to 1.15x, and on
    the colon it takes the pair ranking's rank correlation with the mean local
    density of its two types from -0.608 to -0.383 while moving the ranking
    itself only Spearman +0.934.

    Pruned edges are excluded from the mean — they are the lumen crossings the
    pruning exists to disown, and averaging them back in would inflate the ruler
    exactly at the tissue borders. A cell left with no surviving edge, and any
    coincident pair whose mean comes out zero, falls back to the global median.
    """
    sides = np.concatenate([ei[keep], ej[keep]])
    lens = np.concatenate([elen[keep], elen[keep]])
    num = np.bincount(sides, weights=lens, minlength=n_cells)
    den = np.bincount(sides, minlength=n_cells).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_diam = np.where(den > 0, num / np.where(den > 0, den, 1.0), spacing)
    cell_diam = np.where(np.isfinite(cell_diam) & (cell_diam > 0), cell_diam, spacing)
    edge_diam = 0.5 * (cell_diam[ei] + cell_diam[ej])
    return cell_diam, np.where(edge_diam > 0, edge_diam, spacing)


def _interface_stats(graph, codes, n_levels):
    """Per-pair interface length, segment count and coherence length Lw.

    Pairs are addressed by a flat id `min*n_levels + max`, so every quantity
    here is one `np.bincount` over the heterotypic facets and the whole
    level x level table falls out of a single pass.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ci = codes[graph["ei"]]
    cj = codes[graph["ej"]]
    iface = graph["keep"] & (ci >= 0) & (cj >= 0) & (ci != cj)
    n_iface = int(iface.sum())
    size = n_levels * n_levels
    if n_iface == 0:
        zeros = np.zeros(size)
        return {"L": zeros, "S": zeros.copy(), "Lw": np.full(size, np.nan),
                "iface": iface, "comp": None, "seglen": None, "segpair": None,
                "pid": None}
    pid = np.where(iface,
                   np.minimum(ci, cj).astype(np.int64) * n_levels
                   + np.maximum(ci, cj).astype(np.int64), -1)
    w = graph["w"]
    lengths = np.bincount(pid[iface], weights=w[iface], minlength=size)
    local = np.full(pid.size, -1, dtype=np.int64)
    local[iface] = np.arange(n_iface)
    tri = graph["tri_edges"]
    src, dst = [], []
    for u, v in ((tri[:, 0], tri[:, 1]), (tri[:, 1], tri[:, 2]), (tri[:, 0], tri[:, 2])):
        same = iface[u] & iface[v] & (pid[u] == pid[v])
        src.append(local[u[same]])
        dst.append(local[v[same]])
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    adj = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)),
                     shape=(n_iface, n_iface))
    n_comp, comp = connected_components(adj, directed=False)

    seglen = np.bincount(comp, weights=w[iface], minlength=n_comp)
    segpair = np.zeros(n_comp, dtype=np.int64)
    segpair[comp] = pid[iface]               # every facet of a component agrees
    counts = np.bincount(segpair, minlength=size).astype(np.float64)
    sum_sq = np.bincount(segpair, weights=seglen ** 2, minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence_len = np.where(lengths > 0,
                                 sum_sq / np.where(lengths > 0, lengths, 1.0), np.nan)
    return {"L": lengths, "S": counts, "Lw": coherence_len, "iface": iface,
            "comp": comp, "seglen": seglen, "segpair": segpair, "pid": pid}


def _survival(seglen, grid):
    """Fraction of a pair's interface LENGTH sitting in segments >= x.

    Length-weighted rather than count-weighted for the same reason Lw is: the
    count-weighted curve of any real pair is dominated by its specks and drops
    to nothing before the front it was drawn to show. `seglen` is already in
    cell diameters, so the grid needs no rescaling.
    """
    if seglen.size == 0:
        return np.full(grid.size, np.nan)
    ordered = np.sort(seglen)
    tail = np.concatenate([np.cumsum(ordered[::-1])[::-1], [0.0]])
    total = tail[0]
    if total <= 0:
        return np.full(grid.size, np.nan)
    return tail[np.searchsorted(ordered, grid, side="left")] / total


# ===========================================================================
# The tool
# ===========================================================================
def run(ds, params):
    field = params.get("field") or None
    min_interface = float(params.get("min_interface", 50.0))
    edge_prune = float(params.get("edge_prune", 3.0))
    n_shuffles = max(MIN_SHUFFLES, int(params.get("n_shuffles", 19)))
    want_a = params.get("type_a") or None
    want_b = params.get("type_b") or None

    labels, codes = ds.categorical(field)
    n_levels = len(labels)
    if n_levels < 2:
        raise PluginError("one_level", "This annotation has only one level, and an "
                                       "interface needs two populations.")
    if n_levels > MAX_LEVELS:
        raise PluginError(
            "too_many_levels",
            f"This annotation has {n_levels} levels; the matrix ceiling is "
            f"{MAX_LEVELS}. Pick a coarser grouping.")
    if want_a and want_b and want_a == want_b:
        raise PluginError("same_type", f"{want_a} cannot border itself. Pick two "
                                       "different groups, or leave both blank.")

    coords = ds.coords
    if coords.shape[1] != 2:
        raise PluginError("not_2d", "An interface length is a 2D quantity; the "
                                    "Voronoi dual of a 3D dataset gives areas.")
    section = ds.section
    eligible = np.isfinite(coords).all(axis=1)
    if ds.section_filter is not None:
        eligible &= section == ds.section_filter
    if ds.subset is not None and ds.subset.size:
        # The triangulation is rebuilt inside the selection, so a boundary with
        # the tissue AROUND it is not measured. Masking a graph built on
        # everything would be the other choice and it is the wrong one: it
        # would report contacts with cells the user excluded.
        chosen = np.zeros(ds.n_cells, dtype=bool)
        chosen[ds.subset] = True
        eligible &= chosen
    if int(eligible.sum()) < 4:
        raise PluginError("no_coordinates",
                          "Fewer than four cells here have usable coordinates.")

    ds.progress(0.05, "triangulating")
    graph = _build_graph(coords, section, eligible, edge_prune)
    if graph is None:
        raise PluginError("no_geometry",
                          "No section could be triangulated — the cells here are "
                          "collinear or too few.")
    spacing = graph["spacing"]

    ds.progress(0.2, "tracing interface segments")
    obs = _interface_stats(graph, codes, n_levels)
    if obs["comp"] is None:
        raise PluginError(
            "no_interface",
            f"No two cells of different groups share a Delaunay edge shorter than "
            f"{edge_prune:g} median spacings ({edge_prune * spacing:,.0f} units). "
            "Raise the edge pruning distance.")
    n_sections = int(np.unique(section[eligible]).size)
    lo_d, hi_d = np.percentile(graph["cell_diam"][eligible], [1, 99])
    diam_spread = float(hi_d / max(lo_d, 1e-12))
    ds.log(f"{graph['n_edges']:,} Delaunay edges over {n_sections} section(s), "
           f"{graph['n_kept']:,} kept after pruning; {int(obs['iface'].sum()):,} "
           f"heterotypic; median cell spacing {spacing:.4g}, local cell diameter "
           f"{lo_d:.4g}-{hi_d:.4g} (1st-99th pct, {diam_spread:.1f}x)")

    L, S, Lw = obs["L"], obs["S"], obs["Lw"]
    floor = min_interface
    focus = _focus_pair(labels, L, n_levels, floor, want_a, want_b, ds)

    # The survival grid is fixed from the OBSERVED segments and reused for the
    # null, so the two curves are sampled at the same x and the dashed line is
    # readable against the solid one rather than merely nearby.
    grid = np.geomspace(0.3, max(3.0, float(np.max(obs["seglen"]))), CURVE_POINTS)

    ds.progress(0.25, f"shuffling labels ({n_shuffles}x)")
    null = _null(graph, codes, section, eligible, n_levels, n_shuffles, focus,
                 grid, ds)

    with np.errstate(divide="ignore", invalid="ignore"):
        good = np.isfinite(Lw) & np.isfinite(null["Lw"]) & (null["Lw"] > 0)
        coherence = np.where(good, np.log2(np.where(good, Lw, 1.0)
                                           / np.where(good, null["Lw"], 1.0)), np.nan)
        amount = np.where((L > 0) & (null["L"] > 0),
                          np.log2(np.where(L > 0, L, 1.0)
                                  / np.where(null["L"] > 0, null["L"], 1.0)), np.nan)

    ds.progress(0.85, "painting cells")
    values = _paint(ds, graph, obs, null["Lw"])

    lo, hi = np.divmod(np.arange(n_levels * n_levels), n_levels)
    reportable = (lo < hi) & (L >= floor) & np.isfinite(coherence)
    if not reportable.any():
        raise PluginError(
            "no_interface",
            f"No pair of groups shares at least {min_interface:g} cell diameters "
            "of boundary. Lower the minimum reportable interface.")
    pairs = [(int(a), int(b)) for a, b in zip(lo[reportable], hi[reportable])]
    pairs.sort(key=lambda p: (-L[p[0] * n_levels + p[1]], p[0], p[1]))
    kept = np.array(sorted({i for pair in pairs for i in pair}), dtype=np.int64)

    ds.progress(0.95, "assembling")
    counts = np.bincount(codes[codes >= 0], minlength=n_levels)
    rows = _rows(labels, pairs, counts, L, S, Lw, coherence, amount, n_levels)
    notes = _notes(ds, graph, obs, rows, pairs, coherence, n_levels, n_sections,
                   focus, floor, min_interface, edge_prune, diam_spread)
    return {
        "summary": _summary(rows, pairs, values, obs, L, n_shuffles, field,
                            spacing, ds, notes),
        "cells": {"values": values, "label": "Interface coherence",
                  "colormap": "icefire", "diverging": True,
                  "units": "log2 vs shuffled"},
        "matrix": _matrix(labels, kept, coherence, reportable, n_levels),
        "series": _series(labels, n_levels, obs, pairs, focus, grid, null,
                          n_shuffles),
        "table": {"title": "Interface geometry by pair",
                  "columns": ["Group A", "Group B", "Cells A", "Cells B",
                              "Interface length (diam)", "Segments",
                              "Coherence length (diam)", "log2 coherence",
                              "log2 contact amount (ref)"],
                  "rows": rows, "sort_by": "log2 coherence"},
    }


def _focus_pair(labels, L, n_levels, floor, want_a, want_b, ds):
    """The pair whose segment-length curve is drawn, chosen BEFORE the null.

    Before, so the null curve can be averaged over every shuffle instead of
    being one arbitrary draw kept from the end of the loop.
    """
    if want_a and want_b:
        if want_a in labels and want_b in labels:
            a, b = labels.index(want_a), labels.index(want_b)
            a, b = min(a, b), max(a, b)
            # A chosen pair below the reporting floor is still DRAWN: refusing
            # would turn "these two barely meet" — an answer — into an error.
            # Only a pair that never meets at all has no curve.
            if L[a * n_levels + b] <= 0:
                raise PluginError(
                    "pair_never_touches",
                    f"{labels[a]} and {labels[b]} never share a Delaunay edge here, "
                    "so there is no boundary between them to measure.")
            return a, b
        ds.log(f"{want_a} | {want_b} is not a pair of this grouping's levels; "
               "drawing the longest interface instead.")
    order = np.argsort(-L, kind="stable")
    above = order[L[order] >= max(floor, np.nextafter(0.0, 1.0))]
    best = int(above[0]) if above.size else int(order[0])
    return int(best // n_levels), int(best % n_levels)


def _null(graph, codes, section, eligible, n_levels, n_shuffles, focus, grid,
          ds):
    """Label shuffles WITHIN each section, holding every cell position fixed.

    Within a section because segments never cross one, so a null that moved
    labels between specimens would compare a boundary against a mixture of
    tissues that never touched. Positions are held because the question is
    "given where cells are at all, do these two meet along a front" — scattering
    uniform points instead would be rejected by every pair in any real tissue.
    """
    size = n_levels * n_levels
    rng = np.random.default_rng(0)           # fixed: two runs must be identical
    acc_len = np.zeros(size)
    acc_lw = np.zeros(size)
    seen_lw = np.zeros(size)
    acc_curve = np.zeros(grid.size)
    seen_curve = 0
    focus_pid = focus[0] * n_levels + focus[1]
    blocks = [np.flatnonzero(eligible & (section == s) & (codes >= 0))
              for s in np.unique(section[eligible])]
    shuffled = codes.copy()
    for r in range(n_shuffles):
        for block in blocks:
            shuffled[block] = codes[rng.permutation(block)]
        stats = _interface_stats(graph, shuffled, n_levels)
        acc_len += stats["L"]
        finite = np.isfinite(stats["Lw"])
        acc_lw[finite] += stats["Lw"][finite]
        seen_lw[finite] += 1
        if stats["segpair"] is not None:
            curve = _survival(stats["seglen"][stats["segpair"] == focus_pid],
                              grid)
            if np.isfinite(curve).any():
                acc_curve += np.nan_to_num(curve)
                seen_curve += 1
        ds.progress(0.25 + 0.55 * (r + 1) / n_shuffles, f"shuffle {r + 1}/{n_shuffles}")
    return {"L": acc_len / n_shuffles,
            "Lw": np.where(seen_lw > 0, acc_lw / np.maximum(seen_lw, 1), np.nan),
            "curve": acc_curve / seen_curve if seen_curve else np.full(grid.size, np.nan)}


def _paint(ds, graph, obs, null_lw):
    """Per cell: the facet-length-weighted mean over its own heterotypic facets
    of log2(the segment that facet lies in / the shuffled expectation for THAT
    pair).

    Dividing by the pair's own null inside the sum, rather than once at the end,
    is what stops the map being a map of which abundant type a cell happens to
    border.
    """
    iface = obs["iface"]
    seglen = obs["seglen"][obs["comp"]]
    base = null_lw[obs["pid"][iface]]
    usable = np.isfinite(base) & (base > 0)
    ratio = np.zeros(seglen.size)
    ratio[usable] = np.log2(np.maximum(seglen[usable], 1e-9) / base[usable])
    weight = graph["w"][iface]
    sides = np.concatenate([graph["ei"][iface][usable], graph["ej"][iface][usable]])
    w2 = np.concatenate([weight[usable], weight[usable]])
    r2 = np.concatenate([ratio[usable], ratio[usable]])
    num = np.bincount(sides, weights=w2 * r2, minlength=ds.n_cells)
    den = np.bincount(sides, weights=w2, minlength=ds.n_cells)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _rows(labels, pairs, counts, L, S, Lw, coherence, amount, n_levels):
    rows = []
    for a, b in pairs[:MAX_TABLE_PAIRS]:
        p = a * n_levels + b
        rows.append([labels[a], labels[b], int(counts[a]), int(counts[b]),
                     round(float(L[p]), 1), int(S[p]),
                     round(float(Lw[p]), 2),
                     round(float(coherence[p]), 3),
                     None if not np.isfinite(amount[p]) else round(float(amount[p]), 3)])
    rows.sort(key=lambda r: (-r[7], r[0], r[1]))
    return rows


def _matrix(labels, kept, coherence, reportable, n_levels):
    flat = (np.minimum(kept[:, None], kept[None, :]) * n_levels
            + np.maximum(kept[:, None], kept[None, :]))
    values = np.where(reportable[flat], coherence[flat], np.nan)
    np.fill_diagonal(values, np.nan)         # a group has no boundary with itself
    return {"title": "Interface coherence, log2 observed / label-shuffled",
            "rows": [labels[i] for i in kept], "cols": [labels[i] for i in kept],
            "values": values.tolist(), "diverging": True, "colormap": "icefire"}


def _series(labels, n_levels, obs, pairs, focus, grid, null, n_shuffles):
    shown = [focus] + [p for p in pairs[:CURVE_LINES] if p != focus]
    lines = []
    for i, pair in enumerate(shown[:CURVE_LINES]):
        pid = pair[0] * n_levels + pair[1]
        seglen = obs["seglen"][obs["segpair"] == pid]
        lines.append({"label": f"{labels[pair[0]]} | {labels[pair[1]]}",
                      "x": grid.tolist(),
                      "y": _survival(seglen, grid).tolist(),
                      "color": _CURVE_COLORS[i % len(_CURVE_COLORS)]})
    lines.append({"label": f"{labels[focus[0]]} | {labels[focus[1]]}, "
                           f"mean of {n_shuffles} label shuffles",
                  "x": grid.tolist(), "y": null["curve"].tolist(),
                  "color": _NULL_COLOR, "dashed": True})
    return {"title": "Interface segment-length survival",
            "x_label": "segment length (cell diameters)",
            "y_label": "fraction of interface length in longer segments",
            "lines": lines, "log_x": True}


def _notes(ds, graph, obs, rows, pairs, coherence, n_levels, n_sections, focus,
           floor, min_interface, edge_prune, diam_spread):
    """Caveats that apply to THIS run, in the summary rather than only the log."""
    out = []
    if ds.subset is not None and ds.subset.size:
        out.append(f"Restricted to your selection of {int(ds.subset.size):,} cells; "
                   "the triangulation was rebuilt inside it, so boundaries with the "
                   "tissue around the selection are not measured.")
    pruned = graph["n_edges"] - graph["n_kept"]
    if pruned > 0.25 * graph["n_edges"]:
        out.append(f"{pruned / graph['n_edges']:.0%} of Delaunay edges were longer "
                   f"than {edge_prune:g} median spacings and were dropped, so this "
                   "tissue is patchy at the pruning distance and the boundary lengths "
                   "are lower bounds.")
    scored = np.array([coherence[a * n_levels + b] for a, b in pairs])
    if scored.size and (scored > 0).all():
        out.append(f"All {scored.size:,} reported pairs score above their shuffled "
                   "null, which a complete label shuffle makes close to inevitable: "
                   "rank the pairs, do not read the ratio as a test.")
    focus_pid = focus[0] * n_levels + focus[1]
    if obs["L"][focus_pid] < floor:
        out.append(f"The pair drawn in the curve carries less than the "
                   f"{min_interface:g}-diameter reporting floor, so its curve rests "
                   "on a handful of segments.")
    if n_sections > 1:
        out.append(f"{n_sections} sections are pooled; segments never cross one, and "
                   "the shuffle is within-section, so each section's own packing "
                   "divides out of its own pairs.")
    if diam_spread >= DIAM_SPREAD:
        out.append(f"Cell packing varies {diam_spread:.1f}-fold across this "
                   "tissue (1st to 99th percentile of local cell diameter). Every "
                   "facet is measured in local diameters so that is mostly "
                   f"divided out, but the {edge_prune:g}-spacing pruning distance "
                   "is one global number, so a pair living in sparse tissue still "
                   "ranks a little higher for that reason alone.")
    # Lw is a length-weighted mean over segments, so a leader built from a
    # handful of them is an estimate from a handful of numbers however large the
    # ratio looks. At a low reporting floor that is what tops the ranking.
    if rows[0][5] < THIN_SEGMENTS or rows[0][4] < THIN_DIAMETERS:
        out.append(f"The top-ranked pair carries only {rows[0][5]:,} "
                   f"{'segment' if rows[0][5] == 1 else 'segments'} over "
                   f"{rows[0][4]:.1f} cell diameters of boundary, so its "
                   "ratio rests on very little. Raise the minimum reportable "
                   "interface to rank pairs that carry a front.")
    return out


def _summary(rows, pairs, values, obs, L, n_shuffles, field, spacing, ds, notes):
    """What this run found: the ranking, the numbers behind it, its caveats."""
    top = rows[0]
    worst = rows[-1]
    name = field or (ds.meta.get("celltype_obs") or "cell type")
    parts = [
        (f"**{top[0]} | {top[1]}** is the only reportable interface over "
         f"`{name}`" if len(pairs) == 1 else
         f"**{top[0]} | {top[1]}** is the most coherent of the {len(pairs):,} "
         f"interfaces measured over `{name}`")
        + (f": a random point on it lies in an unbroken stretch "
           f"**{top[6]:.1f} cell diameters** long, {2 ** top[7]:.1f}x the "
           f"shuffled expectation (log2 {top[7]:+.2f}), over {top[5]:,} "
           f"{'segment' if top[5] == 1 else 'segments'} and {top[4]:,.0f} cell "
           f"diameters of boundary."),
    ]
    if len(rows) > 1:
        parts.append(
            f"Most fragmented: **{worst[0]} | {worst[1]}** at {worst[6]:.2f} "
            f"diameters (log2 {worst[7]:+.2f}) over {worst[5]:,} "
            f"{'speck' if worst[5] == 1 else 'specks'}.")
    rho = _spearman(np.array([r[7] for r in rows], dtype=np.float64),
                    np.array([np.nan if r[8] is None else r[8] for r in rows],
                             dtype=np.float64))
    painted = int(np.isfinite(values).sum())
    parts.append(
        f"{int(obs['iface'].sum()):,} heterotypic Voronoi facets, "
        f"{L.sum():,.0f} cell diameters of boundary in total, median cell "
        f"spacing {spacing:.4g} units; {n_shuffles} label shuffles. **{painted:,} of "
        f"{ds.n_cells:,} cells** sit on some interface and are painted, the rest "
        "are interior and blank."
        + ("" if rho is None else
           f" Coherence and the reference contact-amount column rank-correlate at "
           f"rho = {rho:+.2f} across these pairs."))
    parts.extend(notes)
    return "\n\n".join(parts)


def _spearman(a, b):
    """Rank correlation of two columns of this run's own table, or None.

    Hand-rolled, midrank-tied, because the one number wanted here does not
    justify importing scipy.stats into a module that otherwise needs only
    spatial and sparse.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 8:
        return None
    ra, rb = _ranks(a[ok]), _ranks(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return None if denom <= 0 else float((ra * rb).sum() / denom)


def _ranks(x):
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    sx = x[order]
    start = 0
    for i in range(1, x.size + 1):           # midranks, so ties do not bias rho
        if i == x.size or sx[i] != sx[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i - 1) / 2.0
            start = i
    return ranks
