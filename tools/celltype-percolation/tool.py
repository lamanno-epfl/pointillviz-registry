"""Percolation: is a cell type one connected mass, or many separate islands?

What this computes
------------------
Take one category's cells on their own, join every pair closer than a linking
distance r, and record how much of the type ends up in the single largest
connected component:

    P_c(r) = |largest component| / N_c

Percolation is measured WITHIN a section — two specimens laid out on a shared
canvas never share a distance — so with several sections the reported curve is
the cell-weighted mean of the per-section curves,

    P_c(r) = sum_s n_s P_c^(s)(r) / N_c ,

which is the sum of the per-section largest components over the whole count,
and is exactly the line above when there is one section. It is never below the
union-wide largest-component fraction and equals it only when one section holds
the type; so on a multi-section dataset r* is a within-section threshold and is
not comparable with a single-section run. r*/null is, because the null pools
identically.

The percolation threshold r*_c is the smallest r at which P_c reaches a chosen
fraction (default 1/2). A type that already forms a continuous sheet percolates
at r* barely above its own cell spacing; a type broken into islands does not
percolate until r is large enough to bridge between islands, which is a
property of the tissue rather than of the cells.

This is the question none of the enrichment statistics can answer. Adjacency
enrichment, neighbourhood composition and a univariate Ripley L all say
"clustered" for a single sheet and for eight hundred crypt bases alike; the two
sit at opposite ends of P_c(r).

Why one pass gives the whole curve
----------------------------------
The connected components of the r-disc graph, **at every r at once**, are the
components of the Euclidean minimum spanning tree cut at r — that is single
linkage, and it is what makes this cheap. And the EMST is a subgraph of the
Delaunay triangulation in any dimension, so ~3N candidate edges replace the
N(N-1)/2 of the complete graph. One triangulation, one `minimum_spanning_tree`,
one union-find sweep over the sorted tree edges is the entire P_c(r) curve,
exact, in O(N log N).

The same triangulation gives each cell's exact nearest-neighbour distance for
free, because the nearest-neighbour graph is also a Delaunay subgraph — so no
KD-tree is built anywhere in this tool.

Rejected: re-running `connected_components` on a grid of radii (R times the
work for a discretised version of the same curve); `cKDTree.query_pairs(r)` per
radius, where the pair count grows like r^2 — measured on the largest type of
the colon section used throughout (Smooth muscle, 15,060 cells), 45,116
Delaunay candidate edges against 662,714 pairs already at that type's own r*
and 27.6 million at the top of the plotted grid; and
`scipy.cluster.hierarchy.linkage(method='single')`, which reaches the same
merge heights exactly (checked on those points, max difference 0.0) but calls
`pdist` first, so it allocates the condensed distance matrix — 0.91 GB for that
one type, 1.07 GB peak RSS and 1.12 s measured, against 0.09 s here.

In 3D the algorithm is unchanged (`ds.coords` carries the third column and
Qhull triangulates it), which is why this tool declares `spatial: "any"`. The
COST is not unchanged: a 3D Delaunay carries 7.73 candidate edges per point
against 3.00 in 2D, and one 200,000-point triangulation takes 8.0 s. The
timings quoted in the README are 2D timings. On a large 3D section lower
`n_sims` before `max_per_type`: the null loop is the term that multiplies, at a
measured 8.1 s per draw on a 250,000-cell synthetic 3D cloud, so the slider's
49 draws there project to well over six minutes.

Making r* comparable across types
---------------------------------
Raw r* mostly measures abundance — a type with more cells has smaller gaps to
bridge. Two normalisations answer different questions, and both are reported:

  * **r* / d_nn**, against the type's own median within-type nearest-neighbour
    distance. Removes density exactly, and is dimensionless, so every type can
    be drawn on one axis.
  * **r* / r*_null**, against a null that RE-LABELS rather than re-scatters:
    the same number of cells, drawn without replacement from the positions
    that are actually occupied in that section. This holds the tissue's own
    outline and density gradient fixed. A uniform-CSR null does not, and on
    real tissue it is rejected by everything everywhere, which is not a test.

The null is over labels only. It says nothing about whether the cell-type calls
are right: an annotation error correlated with position — a tile seam, a
segmentation-quality gradient — reads as spurious fragmentation, and on a
tile-trained pipeline that is a specific failure mode rather than a generic
caveat.

Determinism
-----------
Every draw comes from one `default_rng` seeded with a SHA-1 of the parameters,
never from the clock and never from `hash()`, which PYTHONHASHSEED randomises
per process. Types and sections are iterated in a fixed sorted order, so the
stream of draws — and therefore the result — is byte-identical between runs.
"""

from __future__ import annotations

import hashlib

import numpy as np

from pointillviz_tool import PluginError

#: Points on the shared dimensionless r/d_nn axis, and its ends. The axis is
#: log-spaced: r* spans two decades across the types of one section, and a
#: linear axis spends nine tenths of its points past every crossing.
N_GRID = 240
GRID_LO, GRID_HI = 0.5, 200.0

#: Observed curves drawn. More than this and the legend is longer than the plot.
MAX_PLOT_TYPES = 10

#: Below this many points a dense (n x n) distance matrix beats a triangulation
#: and cannot hit a Qhull degeneracy. 48 x 48 float64 is 18 kB.
DENSE_MST_CUTOFF = 48

#: The table ceiling is 5,000 rows; a fine annotation on a big atlas can pass
#: `min_cells` with more categories than that.
MAX_TABLE_TYPES = 5000

_NULL_COLOR = "#9aa5b1"

#: Compared against the manifest by the `manifest` gate; the host passes the
#: manifest's values, so this only feeds direct calls to run().
DEFAULTS = {"field": None, "min_cells": 50, "giant_frac": 0.5, "n_sims": 19,
            "link_scale": 3.0, "max_per_type": 40000}


# ===========================================================================
# Entry point
# ===========================================================================
def run(ds, params):
    field = (params.get("field") or "").strip() or None
    min_cells = int(params.get("min_cells", DEFAULTS["min_cells"]))
    frac = float(params.get("giant_frac", DEFAULTS["giant_frac"]))
    n_sims = int(params.get("n_sims", DEFAULTS["n_sims"]))
    link = float(params.get("link_scale", DEFAULTS["link_scale"]))
    cap = int(params.get("max_per_type", DEFAULTS["max_per_type"]))

    levels, codes = ds.categorical(field)
    coords = np.asarray(ds.coords, dtype=np.float64)
    sect = np.asarray(ds.section)

    in_play = np.isfinite(coords).all(axis=1)
    if ds.section_filter is not None:
        in_play &= (sect == ds.section_filter)
    if ds.subset is not None:
        # A lasso restricts the cells AND the positions the null may draw from,
        # so honouring it by restriction alone keeps the null inside whatever
        # the user is looking at.
        keep = np.zeros(ds.n_cells, dtype=bool)
        keep[ds.subset] = True
        in_play &= keep

    counts = np.bincount(codes[in_play & (codes >= 0)], minlength=len(levels))
    included = [c for c in range(len(levels)) if counts[c] >= min_cells]
    if not included:
        biggest = int(counts.max()) if counts.size else 0
        raise PluginError(
            "no_types",
            f"No category has {min_cells:,} cells with coordinates — the largest "
            f"has {biggest:,}. Lower the minimum-cells parameter, or pick a "
            "coarser annotation column.")
    dropped_types = 0
    if len(included) > MAX_TABLE_TYPES:
        included.sort(key=lambda c: (-int(counts[c]), levels[c]))
        dropped_types = len(included) - MAX_TABLE_TYPES
        included = sorted(included[:MAX_TABLE_TYPES])

    seed = int.from_bytes(hashlib.sha1(
        "|".join(str(v) for v in
                 (ds.name, field, min_cells, frac, n_sims, link, cap)).encode()
    ).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    sections = [int(s) for s in np.unique(sect[in_play])]
    pool = {s: np.flatnonzero(in_play & (sect == s)) for s in sections}
    plan, capped = _plan(ds, rng, codes, sect, in_play, included, sections, cap)

    n_graphs = sum(len(v) for v in plan.values())
    ds.log(f"{len(plan)} categories over {len(sections)} section(s): "
           f"{n_graphs:,} {coords.shape[1]}D graphs per pass x {1 + n_sims} passes "
           f"= {n_graphs * (1 + n_sims):,} spanning trees. The pass count, and so "
           f"the runtime, is linear in the null-draw parameter")

    grid = np.geomspace(GRID_LO, GRID_HI, N_GRID)
    observed, patch_all, patch_stats = _observed(ds, coords, plan, included,
                                                 sections, grid, frac, link)
    if not observed:
        raise PluginError(
            "no_types",
            "No category had two cells with coordinates in the same section, so "
            "there is nothing to connect. Percolation is measured within a "
            "section — two specimens never share a distance.")
    null = _null(ds, coords, rng, pool, plan, observed, sections, grid, frac, n_sims)

    return _report(ds, levels, observed, null, patch_all, patch_stats, grid,
                   frac, n_sims, link, cap, capped, dropped_types, min_cells,
                   len(levels), field, sections)


# ===========================================================================
# Which cells each type contributes, after the work cap
# ===========================================================================
def _plan(ds, rng, codes, sect, in_play, included, sections, cap):
    """{type: {section: rows}} — the cells each type is scored on.

    A type above `cap` is subsampled ONCE here, and the null then draws that
    same reduced count, so the observed statistic and its baseline are always
    measured on the same number of points. Subsampling with the run's own
    seeded generator rather than a stride: dataset order is not guaranteed to
    be unrelated to position (a converter that wrote cells tile by tile makes
    every k-th cell a spatial comb), and a comb through a fragmented type is
    exactly the structure this tool measures.
    """
    plan, capped = {}, {}
    for c in included:
        rows = np.flatnonzero(in_play & (codes == c))
        if rows.size > cap:
            capped[c] = int(rows.size)
            rows = np.sort(rng.choice(rows, size=cap, replace=False))
        by_section = {}
        s_of = sect[rows]
        for s in sections:
            part = rows[s_of == s]
            if part.size >= 2:
                by_section[s] = part
        if by_section:
            plan[c] = by_section
    if capped:
        ds.log(f"{len(capped)} categor{'y' if len(capped) == 1 else 'ies'} "
               f"subsampled to {cap:,} cells before the graph")
    return plan, capped


# ===========================================================================
# Geometry: EMST edges and exact nearest-neighbour distances
# ===========================================================================
#: Zero-length edges are not stored by a sparse matrix, so coincident cells
#: would drop out of the graph entirely; the smallest positive float keeps them
#: joined at a distance that rounds to nothing, and is small enough to be told
#: apart from a real one when the nearest-neighbour spacing is taken.
_TINY = np.finfo(np.float64).tiny


def _mst_edges(P):
    """(u, v, w) EMST edges sorted by length, plus each point's NN distance.

    The nearest-neighbour distance is read back off the tree rather than
    measured: the shortest edge at a vertex is in every minimum spanning tree
    (the cut property), so the minimum incident tree edge IS that cell's
    nearest neighbour, and no KD-tree has to be built to learn it.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = P.shape[0]
    if float(np.ptp(P, axis=0).max()) == 0.0:
        # Every cell at one position — a segmentation that emitted N centroids
        # on top of each other. There is no geometry to sweep; the zero NN
        # spacing this returns is what makes the caller skip the category.
        chain = np.arange(n - 1, dtype=np.int64)
        return ((chain, chain + 1, np.full(n - 1, _TINY)), np.zeros(n))

    if n <= DENSE_MST_CUTOFF:
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(P))
        np.maximum(D, _TINY, out=D)
        np.fill_diagonal(D, 0.0)
        return _finish(minimum_spanning_tree(D).tocoo(), n)

    E = _candidate_edges(P)
    d = np.maximum(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1), _TINY)
    G = coo_matrix((d, (E[:, 0], E[:, 1])), shape=(n, n))
    return _finish(minimum_spanning_tree(G).tocoo(), n)


def _finish(M, n):
    o = np.argsort(M.data, kind="stable")
    u = M.row[o].astype(np.int64)
    v = M.col[o].astype(np.int64)
    w = M.data[o].astype(np.float64)
    nn = np.full(n, np.inf)
    np.minimum.at(nn, u, w)
    np.minimum.at(nn, v, w)
    return (u, v, w), nn


def _candidate_edges(P):
    """Unique Delaunay edges as an (m, 2) int array, u < v."""
    from scipy.spatial import Delaunay

    n = P.shape[0]
    try:
        tri = Delaunay(P)
    except Exception:
        try:
            tri = Delaunay(_flatten(P))
        except Exception:
            raise PluginError(
                "degenerate_geometry",
                "One category's cells are in a configuration Qhull cannot "
                "triangulate even after projection — duplicated coordinates, "
                "most likely. Raise the minimum-cells parameter to drop it.")
    s = tri.simplices
    m = s.shape[1]
    E = np.vstack([s[:, [a, b]] for a in range(m) for b in range(a + 1, m)])
    # Points Qhull set aside as coplanar (duplicates, and points on a facet)
    # appear in no simplex. Left out they would be isolated vertices that never
    # join anything, so each is tied to the vertex Qhull reports it sits on.
    cop = np.asarray(getattr(tri, "coplanar", np.empty((0, 3), np.int64)))
    if cop.size:
        E = np.vstack([E, cop[:, [0, 2]]])
    E = np.sort(E, axis=1)
    # De-duplicated by hand: coo_matrix SUMS duplicate entries, which would
    # multiply the weight of every edge two simplices share.
    key = E[:, 0].astype(np.int64) * n + E[:, 1]
    _, keep = np.unique(key, return_index=True)
    return E[keep]


def _flatten(P):
    """The same points in the smallest dimension that holds them.

    Qhull refuses a degenerate cloud — every cell on one line, or on one plane
    in a 3D dataset. Dropping the null directions of the SVD is an isometry, so
    the triangulation of the projection has the same distances and the same
    EMST, rather than the joggled approximation `QJ` would give.
    """
    C = P - P.mean(axis=0)
    _, sv, vt = np.linalg.svd(C, full_matrices=False)
    rank = max(1, int(np.count_nonzero(sv > 1e-9 * max(sv[0], 1e-300))))
    Q = C @ vt[:rank].T
    if rank == 1:
        # A line has no triangulation. Its EMST is the chain through the sorted
        # points, which Delaunay of (x, x^2) reproduces exactly and cheaply.
        Q = np.column_stack([Q[:, 0], Q[:, 0] ** 2])
    return np.ascontiguousarray(Q)


# ===========================================================================
# The union-find sweep
# ===========================================================================
def _sweep(edges, n, r_cut=None):
    """Single-linkage sweep over sorted EMST edges.

    Returns (r_break, best, roots): the largest component grows to best[i] at
    linking distance r_break[i], and — when `r_cut` is given — roots[j] is the
    component point j belongs to once every edge shorter than r_cut has been
    added. Lists rather than arrays on purpose: this is a scalar Python loop,
    and indexing a list is about three times as fast as indexing an ndarray
    (measured 19.0 ms against 55.7 ms for the same scalar loop).
    """
    u, v, w = edges
    parent = list(range(n))
    size = [1] * n
    rb = [0.0]
    bs = [1]
    best = 1
    snap = None
    for e in range(w.shape[0]):
        if r_cut is not None and snap is None and w[e] > r_cut:
            snap = list(parent)
        x = u[e]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = v[e]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x == y:
            continue
        if size[x] < size[y]:
            x, y = y, x
        parent[y] = x
        s = size[x] + size[y]
        size[x] = s
        if s > best:
            best = s
            rb.append(float(w[e]))
            bs.append(s)
    roots = None
    if r_cut is not None:
        if snap is None:                       # every edge was below the cut
            snap = list(parent)
        roots = np.empty(n, np.int64)
        for i in range(n):
            r = i
            while snap[r] != r:
                snap[r] = snap[snap[r]]
                r = snap[r]
            roots[i] = r
    return np.asarray(rb, np.float64), np.asarray(bs, np.int64), roots


def _r_at(rb, bs, target):
    """Smallest linking distance with a component of at least `target` cells."""
    j = int(np.searchsorted(bs, target, side="left"))
    return float(rb[j]) if j < bs.shape[0] else float(rb[-1])


def _step_eval(rb, bs, grid):
    """Largest-component size as the step function it is, read off `grid`."""
    j = np.searchsorted(rb, grid, side="right") - 1
    np.clip(j, 0, bs.shape[0] - 1, out=j)
    return bs[j].astype(np.float64)


def _median_nn(nn):
    """Median nearest-neighbour distance, over the cells that have one.

    Coincident cells are excluded rather than counted as spacing zero: a
    segmentation that duplicates centroids would otherwise pull the whole
    dimensionless axis towards zero for the type it duplicated.
    """
    d = nn[np.isfinite(nn) & (nn > _TINY)]
    return float(np.median(d)) if d.size else np.nan


def _threshold(k, frac):
    """Cells needed for a component to count as giant. At least 2: a lone cell
    is a component of size 1 at r = 0, and 10% of nine cells is not a giant."""
    return max(2, int(np.ceil(frac * k)))


# ===========================================================================
# The observed pass
# ===========================================================================
def _observed(ds, coords, plan, included, sections, grid, frac, link):
    """Per type: the pooled P_c curve, r*, d_nn — and the per-cell patch map."""
    out = {}
    patch_all = np.full(coords.shape[0], np.nan, np.float64)
    stats = {}
    order = [c for c in included if c in plan]
    for i, c in enumerate(order):
        pooled = np.zeros(N_GRID)
        dnn_parts, w_parts, single = [], [], None
        patch_of_cell, n_comp = [], 0
        for s in sorted(plan[c]):
            rows = plan[c][s]
            edges, nn = _mst_edges(coords[rows])
            dnn = _median_nn(nn)
            if not np.isfinite(dnn) or dnn <= 0:
                continue                      # every cell of this type coincident
            rb, bs, roots = _sweep(edges, rows.size, r_cut=link * dnn)
            _, inv, comp = np.unique(roots, return_inverse=True, return_counts=True)
            n_comp += int(comp.size)
            patch_all[rows] = comp[inv]
            patch_of_cell.append(comp[inv])
            pooled += _step_eval(rb, bs, grid * dnn)
            dnn_parts.append(dnn)
            w_parts.append(rows.size)
            single = (rb, bs)
        if not w_parts:
            continue
        wts = np.asarray(w_parts, float)
        n_c = int(wts.sum())
        dnn_c = float(np.dot(np.asarray(dnn_parts), wts) / wts.sum())
        pooled /= float(n_c)
        if len(w_parts) == 1:
            # One section: read r* off the sweep itself rather than off the
            # plotted grid, which would round it to the nearest of 240 steps.
            r_raw = _r_at(single[0], single[1], _threshold(n_c, frac))
            r_norm = r_raw / dnn_c
        else:
            r_norm = _cross(pooled, grid, frac)
            r_raw = r_norm * dnn_c
        out[c] = {"n": n_c, "dnn": dnn_c, "r_raw": r_raw, "r_norm": r_norm,
                  "curve": pooled}
        # The median is taken over CELLS, not over patches: a type that is one
        # sheet plus a scatter of strays has a patch-weighted median of 1 and a
        # cell-weighted median of the sheet, and the sheet is what the map shows.
        stats[c] = (n_comp, float(np.median(np.concatenate(patch_of_cell))))
        ds.progress(0.05 + 0.25 * (i + 1) / len(order), "percolation curves")
    return out, patch_all, stats


def _cross(curve, grid, frac):
    hit = np.flatnonzero(curve >= frac)
    return float(grid[hit[0]]) if hit.size else np.nan


# ===========================================================================
# The null
# ===========================================================================
def _null(ds, coords, rng, pool, plan, observed, sections, grid, frac, n_sims):
    """{type: (r* draws, null curves)} — same counts, same support, no labels."""
    out = {c: ([], []) for c in observed}
    for sim in range(n_sims):
        for c in sorted(observed):
            pooled = np.zeros(N_GRID)
            dnn_parts, w_parts, single = [], [], None
            for s in sorted(plan[c]):
                k = int(plan[c][s].size)
                rows = rng.choice(pool[s], size=k, replace=False)
                edges, nn = _mst_edges(coords[rows])
                dnn = _median_nn(nn)
                if not np.isfinite(dnn) or dnn <= 0:
                    continue
                rb, bs, _ = _sweep(edges, k)
                pooled += _step_eval(rb, bs, grid * dnn)
                dnn_parts.append(dnn)
                w_parts.append(k)
                single = (rb, bs)
            if not w_parts:
                continue
            wts = np.asarray(w_parts, float)
            n_c = int(wts.sum())
            dnn_c = float(np.dot(np.asarray(dnn_parts), wts) / wts.sum())
            pooled /= float(n_c)
            if len(w_parts) == 1:
                r_raw = _r_at(single[0], single[1], _threshold(n_c, frac))
            else:
                r_raw = _cross(pooled, grid, frac) * dnn_c
            out[c][0].append(r_raw)
            out[c][1].append(pooled)
        ds.progress(0.3 + 0.65 * (sim + 1) / n_sims, f"null draw {sim + 1}/{n_sims}")
    return out


# ===========================================================================
# Reporting
# ===========================================================================
def _report(ds, levels, observed, null, patch_all, patch_stats, grid, frac,
            n_sims, link, cap, capped, dropped_types, min_cells, n_levels,
            field, sections):
    order = sorted(observed, key=lambda c: (-observed[c]["n"], levels[c]))
    rows = []
    for c in order:
        d = observed[c]
        draws = np.asarray(null[c][0], float)
        draws = draws[np.isfinite(draws)]
        if draws.size >= 2:
            mu = float(draws.mean())
            sd = float(draws.std(ddof=1))
            ratio = d["r_raw"] / mu if mu > 0 else np.nan
            z = (d["r_raw"] - mu) / sd if sd > 0 else np.nan
            # Two-sided rank p, so fragmentation and connectedness are equally
            # reportable. It cannot go below 2/(draws+1) — with 19 draws the
            # floor is 0.1 and every type that clears the null sits on it.
            lo = 1.0 + float(np.count_nonzero(draws <= d["r_raw"]))
            hi = 1.0 + float(np.count_nonzero(draws >= d["r_raw"]))
            p = min(1.0, 2.0 * min(lo, hi) / (draws.size + 1.0))
        else:
            ratio = z = p = np.nan
        n_patch, med_patch = patch_stats.get(c, (None, np.nan))
        # A capped category is marked in its NAME rather than in its cell
        # count, so every numeric column stays numeric and sortable; the
        # summary says what the mark means.
        rows.append([_marked(levels, capped, c), d["n"],
                     _round(d["r_raw"], 1), _round(d["r_norm"], 2),
                     _round(ratio, 3), _round(z, 2), _round(p, 4),
                     n_patch, _round(med_patch, 0)])
    # Sorted by the headline statistic so the connected end of the tissue is at
    # the top of the table whatever the viewer's own sort does; types with no
    # null (0 draws, or a degenerate one) go last rather than to the top.
    rows.sort(key=lambda r: (r[4] is None, r[4] if r[4] is not None else 0.0, r[0]))

    values = np.full(ds.n_cells, np.nan, np.float64)
    ok = np.isfinite(patch_all) & (patch_all >= 1)
    values[ok] = np.log10(patch_all[ok])

    colors = _colors(ds, field)
    lines = []
    n_banded = 0
    if n_sims >= 2:
        banded = [c for c in order[:MAX_PLOT_TYPES] if null[c][1]]
        if banded:
            n_banded = len(banded)
            # One fill can only sit between two lines, so a per-curve envelope
            # is not drawable for ten curves. The band is therefore the 5th and
            # 95th percentile of every plotted type's draws POOLED, and is
            # labelled as the mixture it is; the per-type test is z and p.
            stack = np.vstack([np.vstack(null[c][1]) for c in banded])
            lines.append({"label": f"random subsets, 5th pct "
                                   f"({n_banded} plotted types pooled, "
                                   f"{n_sims} draws each)",
                          "x": grid, "y": np.percentile(stack, 5, axis=0),
                          "color": _NULL_COLOR, "band": True})
            lines.append({"label": "random subsets, 95th pct (same pool)",
                          "x": grid, "y": np.percentile(stack, 95, axis=0),
                          "color": _NULL_COLOR, "band": True})
    for c in order[:MAX_PLOT_TYPES]:
        # The colour is looked up under the real level name; only what the
        # reader sees carries the cap mark.
        lines.append({"label": _marked(levels, capped, c), "x": grid,
                      "y": observed[c]["curve"], "color": colors.get(levels[c])})

    return {
        "summary": _summary(levels, observed, order, rows, frac, n_sims, link,
                            cap, capped, dropped_types, min_cells, n_levels,
                            field, sections, values, ds.n_cells, n_banded),
        "cells": {"values": values,
                  "label": f"log10 own-type patch size at r = {link:g} x NN",
                  "colormap": "magma", "units": "log10 cells"},
        "series": {"title": ("Largest connected component vs linking distance"
                             if len(sections) == 1 else
                             "Within-section largest component vs linking "
                             "distance, cell-weighted over sections"),
                   "x_label": "linking distance r / type NN spacing",
                   "y_label": ("largest component / cells of that type"
                               if len(sections) == 1 else
                               "mean within-section largest component / cells "
                               "of that type"),
                   "log_x": True, "lines": lines, "rules": [frac]},
        "table": {"title": "Percolation by category",
                  "columns": ["Type", "Cells", "r*", "r*/NN", "r*/null", "z", "p",
                              "Patches", "Median patch"],
                  "rows": rows, "sort_by": "r*/null"},
    }


def _marked(levels, capped, c):
    """A category's display name, marked when it was subsampled."""
    return levels[c] + (" *" if c in capped else "")


def _colors(ds, field):
    for entry in ds.categorical_fields():
        if field is None or entry["name"] == field or entry["label"] == field:
            return entry["colors"] or {}
    return {}


def _summary(levels, observed, order, rows, frac, n_sims, link, cap, capped,
             dropped_types, min_cells, n_levels, field, sections, values,
             n_cells, n_banded=0):
    name = field or "cell type"
    n_painted = int(np.count_nonzero(np.isfinite(values)))
    parts = [
        f"**{len(observed)} of {n_levels} `{name}` categories** scored "
        f"({min_cells:,}+ placed cells, {len(sections)} section(s)); "
        f"{n_painted:,} of {n_cells:,} cells painted with the size of their own "
        f"contiguous patch at r = {link:g} x NN spacing."
    ]
    ranked = [r for r in rows if r[4] is not None]
    if ranked:
        top, bottom = ranked[0], ranked[-1]
        parts.append(
            f"Most connected for its abundance: **{top[0]}** — r* = {top[2]:g} "
            f"units, **{top[4]}x** the null, {top[7]:,} patches with the median "
            f"cell in one of {top[8]:,.0f}. Most fragmented: **{bottom[0]}** — "
            f"{bottom[4]}x the null, {bottom[7]:,} patches, median cell in one "
            f"of {bottom[8]:,.0f}.")
        ratios = np.asarray([r[4] for r in ranked], float)
        below = int(np.count_nonzero(ratios < 1.0))
        parts.append(
            f"{below} of {ratios.size} categories percolate sooner than a random "
            f"subset of the same size drawn from the same tissue, {ratios.size - below} "
            f"later; median r*/null **{np.median(ratios):.2f}**. "
            f"{n_sims} draws, so the smallest reportable p is "
            f"{min(1.0, 2.0 / (n_sims + 1)):.3f}.")
    elif n_sims == 0:
        parts.append(
            "No null was drawn, so r* is reported raw and in units of each "
            "category's own nearest-neighbour spacing only — neither is "
            "comparable between categories of different abundance.")
    else:
        parts.append("No category produced a usable null draw, so r*/null is blank.")

    unfinished = [levels[c] for c in order if not np.isfinite(observed[c]["r_norm"])]
    if unfinished:
        parts.append(
            f"{len(unfinished)} categor{'y' if len(unfinished) == 1 else 'ies'} "
            f"never reach {frac:.0%} in one component within {GRID_HI:g}x their own "
            f"spacing ({', '.join(unfinished[:4])}"
            f"{', ...' if len(unfinished) > 4 else ''}); their curves stop short "
            "and their r* is blank.")
    if capped:
        worst = max(capped.values())
        parts.append(
            f"{len(capped)} categor{'y was' if len(capped) == 1 else 'ies were'} "
            f"subsampled to {cap:,} cells before the graph (largest: {worst:,}), "
            f"marked **\\*** in the table. The null draws the same reduced count, "
            "so r*/null, z and p stand. Everything else in those rows is the "
            "thinned type and not the whole one: r*, r*/NN, Patches, Median "
            "patch, **and the patch size painted on the map**. The shift is not "
            "a fixed factor you can divide out: in this tool's reference "
            "measurement a 3.4x thinning moved the painted value on the cells "
            "the two runs have in common by 17.2x. Where some categories are "
            "capped and others are not, the map is two scales on one ramp.")
    if dropped_types:
        parts.append(f"{dropped_types:,} further categories passed the cell "
                     f"minimum but are beyond the {MAX_TABLE_TYPES:,}-row table limit.")
    if len(sections) > 1:
        parts.append(
            f"Cells are never joined across sections, so with {len(sections)} "
            "sections the curve is the cell-weighted mean of the per-section "
            "largest-component fractions, not the largest component of the "
            "union. r* and r*/NN are therefore within-section thresholds and "
            "are not comparable with a single-section run of the same tissue; "
            "r*/null is, because the null is pooled the same way.")
    if n_banded > 1:
        parts.append(
            f"The grey band pools the null draws of all {n_banded} plotted "
            "curves — one fill cannot be ten envelopes — so it is a mixture and "
            "a curve can sit inside it while lying outside its own null. The "
            "per-type test is the z and p columns, which are never pooled.")
    parts.append(
        "r* is a threshold one bridging cell can set: a type in two equal "
        "sheets reads as fragmented however solid each sheet is. The plateau in "
        "the curve is the more robust reading.")
    return "\n\n".join(parts)


def _round(v, nd):
    if v is None:
        return None
    f = float(v)
    if not np.isfinite(f):
        return None
    return round(f, nd) if nd else int(round(f))
