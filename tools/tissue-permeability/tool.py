"""Tissue Permeability — what one cell type walls off from the rest of a section.

Every cell is routed along a pruned Delaunay graph of cell centroids to its
nearest source (the tissue's own exposed surface, or a second cell type), once
with the barrier type removed from the graph and once with it in place. Two
things come out of the pair: the extra path the barrier forces, and the set of
cells for which no path exists at all.

Design notes — the reasoning, not the manual
--------------------------------------------
WHAT CARRIES THE PATH. A pruned Delaunay triangulation, not a raster. A grid
needs a resolution parameter with no good answer: pixels much larger than the
cell spacing turn a one-cell-thick epithelium into a solid wall, pixels much
smaller turn the tissue into Swiss cheese, and an 8-connected lattice inflates
every straight line by up to 8% — a systematic bias straight into a ratio. The
Delaunay graph is parameter-free in resolution, contains every
nearest-neighbour edge, and its own stretch is measured here rather than
assumed (the "open tissue" line in the series IS that measurement).

WHY BOTH NODES AND EDGES ARE BLOCKED, AND WHAT THAT IS WORTH. Deleting only the
barrier NODES leaves a sheet leaky: Delaunay puts an edge straight across a
single-cell-thick wall wherever two cells on opposite sides are each other's
neighbours. Blocking the SEGMENT closes those. Measured on the 135k-cell colon
section at the shipped defaults, Smooth muscle as barrier — sealed cells /
compartments:

    pure node deletion (radius 0)          1,592 / 764
    node dilation to 0.75 spacings         1,638 / 760
    dilation + segment probe (shipped)     1,942 / 910

So the probe is worth 22% more enclosure — real, and less than an earlier draft
of this note claimed. It does not rescue a genuine monolayer: on a jittered
lattice ringed by a ONE-cell-thick wall around 310 interior cells, nothing seals
at any half-width below 1.25 spacings (0/310 at the default 0.75, 269/310 at
1.25, 310/310 once the ring is 1.4 cells thick). A one-cell-thick barrier needs
barrier_width >= ~1.25, and the parameter help says so.

WHY ONE PARAMETER SETS BOTH "CROSSABLE" AND "IS A SURFACE". `max_gap` prunes
the long edges AND flags their endpoints as surface, so a void is either routed
across or supplies a source — which means every open region keeps at least one
source and no amount of pruning can manufacture a false enclosure. The flag is
specific but not sensitive: on a synthetic annulus it catches 14% of a known
lumen lining and 21% of the outer rim with ZERO interior false positives, and
the outermost flagged inner-rim cell sits at radius 12.34 against a lumen edge
of 12.0. Every flagged cell is in the right place; most correct ones are not
flagged. The cost is interpretive, not numerical — `extra` and `tau` measure the
route to the nearest FLAGGED surface cell, and since numerator and denominator
use the same source set, tau is still a tortuosity.

WHY ENCLOSURE IS THE HEADLINE AND THE MEDIAN IS NOT. A coherent wall leaves
almost every cell at the open-tissue floor and seals a compartment instead;
scattered obstacles of the same abundance are a fine-grained maze that raises
the typical detour and seal nothing. So the median tortuosity of a real barrier
runs BELOW its own null while its enclosure count runs orders of magnitude
above it, and a tool that led with the median would report the strong finding
as a null result. The map paints the compartments and leaves the open corridors
blank; the series shows the crossover that explains why.
"""

from __future__ import annotations

import numpy as np

# Mirrors the manifest; the validator compares the two. `barrier` has no
# manifest default on purpose — see README — and is None here only so that a
# direct call can fill it.
DEFAULTS = {
    "field": None,
    "barrier": None,
    "source": "surface",
    "source_type": None,
    "barrier_width": 0.75,
    "max_gap": 5.0,
    "paint": "compartment",
    "n_null": 5,
}

# Percentiles the series is drawn at. Weighted towards the tail because that is
# where a coherent barrier acts at all.
_QS = np.array([50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 92.0, 94.0, 95.0,
                96.0, 97.0, 98.0, 98.5, 99.0, 99.5, 99.9])

_MIN_TABLE_CELLS = 25          # below this a per-type percentage is theatre


def _plugin_error(code, message):
    from pointillviz_tool import PluginError
    return PluginError(code, message)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _simplex_edges(simplices):
    """Every undirected edge of a Delaunay complex, deduplicated.

    Pairs are enumerated rather than hard-coded to three columns so that a
    tetrahedral (3D) complex works unchanged.
    """
    m = simplices.shape[1]
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    e = np.concatenate([simplices[:, [a, b]] for a, b in pairs], axis=0)
    e.sort(axis=1)
    return np.unique(e, axis=0)


def _section_geometry(P, max_gap_mult):
    """Pruned Delaunay edges, surface flags and the median spacing for one section."""
    from scipy.spatial import Delaunay
    n = P.shape[0]
    try:
        tri = Delaunay(P)
    except Exception:
        # Collinear or otherwise degenerate: no triangulation, so no geometry.
        return np.zeros((0, 2), np.int64), np.zeros(0), np.ones(n, bool), np.nan
    E = _simplex_edges(tri.simplices.astype(np.int64))
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    # Median nearest-neighbour distance, free: the Delaunay graph contains the
    # nearest-neighbour graph, so the shortest edge at each node IS its NN.
    shortest = np.full(n, np.inf)
    np.minimum.at(shortest, E[:, 0], L)
    np.minimum.at(shortest, E[:, 1], L)
    finite = shortest[np.isfinite(shortest)]
    spacing = float(np.median(finite)) if finite.size else np.nan
    if not np.isfinite(spacing) or spacing <= 0:
        return np.zeros((0, 2), np.int64), np.zeros(0), np.ones(n, bool), np.nan
    keep = L <= max_gap_mult * spacing
    surface = np.zeros(n, bool)
    surface[E[~keep].ravel()] = True                       # faces a void wider than max_gap
    surface[np.asarray(tri.convex_hull).ravel()] = True    # or faces the outside
    return E[keep], L[keep], surface, spacing


def _graph(n, edges, lengths, mask):
    from scipy.sparse import coo_matrix
    e, w = edges[mask], lengths[mask]
    if e.shape[0] == 0:
        return coo_matrix((n, n)).tocsr()
    ii = np.concatenate([e[:, 0], e[:, 1]])
    jj = np.concatenate([e[:, 1], e[:, 0]])
    ww = np.concatenate([w, w])
    return coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()


def _blocked(P, edges, lengths, barrier_rows, radius, exempt, tree_cls):
    """(node blocked, edge passable) for one barrier set.

    A node is blocked when a barrier centre lies within `radius`; an edge is
    blocked when the SEGMENT does, probed along its length at a spacing of
    `radius`, up to 16 probes. Sampling is adaptive — a short edge needs one
    probe, a long one needs len/radius of them — which is what keeps this off
    the critical path.

    The 16-probe cap is a real approximation, and it binds only at tiny widths.
    On the colon section no edge reaches it at barrier_width >= 0.5; 1.4% of
    edges do at 0.25, and 96.7% at the slider's floor of 0.05, where the probes
    spread to 6.2x `radius` and can miss a crossing. Re-running the identical
    blocking with the cap at 64 and at 256 moves the answer by at most one
    sealed cell and three passable edges anywhere on the slider, while costing
    3.0 s instead of 1.5 s — so the cap stays and this note exists.
    """
    n = P.shape[0]
    tree = tree_cls(P[barrier_rows])
    if radius > 0:
        d, _ = tree.query(P, distance_upper_bound=radius)
        node_blocked = np.isfinite(d)
    else:
        node_blocked = np.zeros(n, bool)
        node_blocked[barrier_rows] = True
    node_blocked[exempt] = False                           # sources are never obstacles
    ok = (~node_blocked[edges[:, 0]]) & (~node_blocked[edges[:, 1]])
    if radius > 0 and edges.shape[0]:
        A, B = P[edges[:, 0]], P[edges[:, 1]]
        need = np.clip(np.ceil(lengths / radius).astype(np.int64), 1, 16)
        hit = np.zeros(edges.shape[0], bool)
        for s in range(int(need.max())):
            sel = np.flatnonzero((need > s) & ok & ~hit)
            if sel.size == 0:
                continue
            t = (s + 0.5) / need[sel]
            pts = A[sel] + t[:, None] * (B[sel] - A[sel])
            d, _ = tree.query(pts, distance_upper_bound=radius)
            hit[sel] = np.isfinite(d)
        ok &= ~hit
    return node_blocked, ok


def _route(n, edges, lengths, P, barrier_rows, radius, src, d_open, tree_cls):
    """One barrier realisation: sealed mask, compartment sizes, blocked geodesic.

    Compartments come from the connected components of the blocked graph rather
    than a second search: a component holding no source is exactly a sealed
    pocket, and its size is what the map paints. Barrier cells themselves fall
    out as isolated singletons — every edge touching one is cut — so the count
    is restricted to the routable domain or every wall cell would read as a
    one-cell compartment.

    The component LABELS come back rather than a compartment count, because the
    caller may be reporting a lassoed subset and a count taken here would be the
    whole section's — which is how the summary once managed to put 157 enclosed
    cells in 910 compartments.
    """
    from scipy.sparse.csgraph import connected_components, dijkstra
    nb, eok = _blocked(P, edges, lengths, barrier_rows, radius, src, tree_cls)
    G = _graph(n, edges, lengths, eok)
    d_bar = dijkstra(G, directed=False, indices=src, min_only=True)
    domain = (~nb) & np.isfinite(d_open)
    sealed = domain & ~np.isfinite(d_bar)
    _, lab = connected_components(G, directed=False)
    size = np.bincount(lab, weights=domain.astype(np.float64),
                       minlength=int(lab.max()) + 1)
    comp = np.full(n, np.nan)
    comp[sealed] = size[lab[sealed]]
    return domain, sealed, comp, d_bar, lab, size


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------
def run(ds, params):
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    field = params.get("field") or None
    source_mode = str(params.get("source") or "surface")
    paint = str(params.get("paint") or "compartment")
    width_mult = float(params.get("barrier_width", 0.75))
    gap_mult = float(params.get("max_gap", 5.0))
    # The manifest floors this at 2 and the host cannot send less, but a direct
    # caller can: at 0 the null arrays are empty, `np.mean([])` puts "Mean of
    # empty slice" on stderr and `null_sealed_n.min()` then dies with "zero-size
    # array to reduction operation minimum". Refuse in words instead — the same
    # class of defect as the NaNs in _summary, one layer up.
    n_null = int(params.get("n_null", 5))
    if n_null < 1:
        raise _plugin_error("no_null",
                            "The null needs at least one random barrier to draw. Every "
                            "number this tool reports is read against it, so there is "
                            "nothing to say with zero draws.")

    levels, codes = ds.categorical(field)
    barrier_name = params.get("barrier")
    if not barrier_name:
        raise _plugin_error("no_barrier",
                            "Choose the cell type that acts as the barrier. There is no "
                            "sensible default — which population walls the tissue off is "
                            "the question, not a setting.")
    if barrier_name not in levels:
        raise _plugin_error("unknown_level", f"{barrier_name!r} is not a level of that column.")
    b_code = levels.index(barrier_name)

    s_code = None
    source_name = params.get("source_type")
    if source_mode == "type":
        if not source_name:
            raise _plugin_error("no_source",
                                "Routing from a cell type needs one chosen. Pick a source "
                                "type, or route from the tissue surface instead.")
        if source_name not in levels:
            raise _plugin_error("unknown_level", f"{source_name!r} is not a level of that column.")
        if source_name == barrier_name:
            raise _plugin_error("same_type",
                                "The source and the barrier are the same cell type, so every "
                                "route starts inside the wall. Choose two different types.")
        s_code = levels.index(source_name)

    n_cells = ds.n_cells
    coords = ds.coords
    sect = ds.section
    placed = np.isfinite(coords).all(axis=1)
    if ds.section_filter is not None:
        placed &= (sect == int(ds.section_filter))
    if int(placed.sum()) < 50:
        raise _plugin_error("too_few", "Fewer than 50 cells have usable coordinates here.")

    # The geometry stays whole even when the user has lassoed a subset: deleting
    # the unselected cells would change the tissue's topology and answer a
    # different question. Only the REPORTING is restricted — and it has to be
    # known here, before the loop, because the compartment count is taken over
    # the reported set and not over the section.
    reported = np.ones(n_cells, bool)
    if ds.subset is not None and ds.subset.size:
        reported = np.zeros(n_cells, bool)
        reported[ds.subset] = True
    subset_on = bool(ds.subset is not None and ds.subset.size)

    extra = np.full(n_cells, np.nan)        # extra path length, in cell diameters
    tau = np.full(n_cells, np.nan)          # geodesic / straight line, with the barrier
    tau_free = np.full(n_cells, np.nan)     # the same ratio with the barrier removed
    comp = np.full(n_cells, np.nan)         # sealed-compartment size, in cells
    ratio_ok = np.zeros(n_cells, bool)      # tau has a usable denominator here
    in_domain = np.zeros(n_cells, bool)
    sealed = np.zeros(n_cells, bool)
    null_tau = np.full((n_null, n_cells), np.nan)
    null_extra = np.full((n_null, n_cells), np.nan)
    null_sealed = np.zeros((n_null, n_cells), bool)
    null_comps = np.zeros(n_null, np.int64)
    rng = np.random.default_rng(0)

    spacings, n_barrier_tot, n_source_tot = [], 0, 0
    comp_sizes = []                         # one array of compartment sizes per section
    sec_ids = np.unique(sect[placed])
    n_sec_used = n_sec_tried = n_degenerate = 0
    for si, s in enumerate(sec_ids):
        rows = np.flatnonzero(placed & (sect == s))
        if rows.size < 50:
            continue
        ds.progress(0.05 + 0.8 * si / max(1, len(sec_ids)), f"section {int(s)}: geometry")
        P = coords[rows].astype(np.float64)
        edges, lengths, surface, spacing = _section_geometry(P, gap_mult)
        n_sec_tried += 1
        if edges.shape[0] == 0:
            n_degenerate += 1        # collinear or coincident: no triangulation
            continue
        n = rows.size
        cc = codes[rows]
        rep_loc = reported[rows]
        barrier_rows = np.flatnonzero(cc == b_code)
        src_mask = surface.copy() if source_mode == "surface" else (cc == s_code)
        src_mask &= (cc != b_code)
        src = np.flatnonzero(src_mask)
        n_barrier_tot += int(barrier_rows.size)
        n_source_tot += int(src.size)
        if barrier_rows.size == 0 or src.size == 0:
            continue
        spacings.append(spacing)
        n_sec_used += 1

        radius = width_mult * spacing
        d_open = dijkstra(_graph(n, edges, lengths, np.ones(edges.shape[0], bool)),
                          directed=False, indices=src, min_only=True)
        eu, _ = cKDTree(P[src]).query(P)

        ds.progress(0.05 + 0.8 * (si + 0.3) / max(1, len(sec_ids)), f"section {int(s)}: barrier")
        domain, sld, cmp_, d_bar, lab, size = _route(n, edges, lengths, P, barrier_rows,
                                                     radius, src, d_open, cKDTree)
        sel = sld & rep_loc
        if sel.any():
            comp_sizes.append(size[np.unique(lab[sel])])
        reach = domain & np.isfinite(d_bar)
        # The ratio needs a denominator: inside one cell spacing of a source it
        # is two small numbers, and the quotient is graph noise. Enclosure has
        # no such problem, so it is NOT gated on this.
        rok = domain & (eu > spacing)
        in_domain[rows[domain]] = True
        sealed[rows[sld]] = True
        comp[rows[sld]] = cmp_[sld]
        extra[rows[reach]] = (d_bar[reach] - d_open[reach]) / spacing
        ratio_ok[rows[rok]] = True
        tau[rows[rok & reach]] = d_bar[rok & reach] / eu[rok & reach]
        tau_free[rows[rok]] = d_open[rok] / eu[rok]

        pool = np.flatnonzero(~src_mask)
        take = int(min(barrier_rows.size, pool.size))
        for r in range(n_null):
            ds.progress(0.05 + 0.8 * (si + 0.4 + 0.5 * r / n_null) / max(1, len(sec_ids)),
                        f"section {int(s)}: random barrier {r + 1}/{n_null}")
            nrows = rng.choice(pool, size=take, replace=False)
            dom2, sld2, cmp2, d2, lab2, size2 = _route(n, edges, lengths, P, nrows,
                                                       radius, src, d_open, cKDTree)
            reach2 = dom2 & np.isfinite(d2)
            rok2 = dom2 & (eu > spacing)
            null_sealed[r, rows[sld2]] = True
            sel2 = sld2 & rep_loc
            null_comps[r] += int(np.unique(lab2[sel2]).size) if sel2.any() else 0
            null_extra[r, rows[reach2]] = (d2[reach2] - d_open[reach2]) / spacing
            null_tau[r, rows[rok2 & reach2]] = d2[rok2 & reach2] / eu[rok2 & reach2]

    if not in_domain.any():
        if n_sec_tried and n_degenerate == n_sec_tried:
            raise _plugin_error(
                "no_geometry",
                "No section could be triangulated — its cells are collinear or sit on top "
                "of one another, so there is no graph to route on. The coordinates are "
                "finite; they are just not two-dimensional.")
        if n_barrier_tot == 0:
            raise _plugin_error(
                "empty_barrier",
                f"{barrier_name!r} has no cells in any section with usable coordinates, "
                "so there is nothing for a path to go around.")
        if n_source_tot == 0:
            raise _plugin_error(
                "empty_source",
                "Nothing to route towards: no source cell survived. "
                + ("Raise the gap setting so fewer cells count as surface, or route from a "
                   "cell type instead." if source_mode == "surface" else
                   "The source type has no cells outside the barrier."))
        raise _plugin_error(
            "no_interior",
            f"No cell is both outside {barrier_name!r} and connected to a source, so there "
            "is no route to measure. The barrier half-width "
            f"({width_mult:.2f} spacings) may be wide enough to swallow the section — "
            "lower it, or raise the gap setting.")

    # Every cell being a source is not a small answer, it is no answer: d_open
    # is zero everywhere, the ratio has no denominator, and nothing is enclosed
    # because nothing has anywhere to get to. Which of the two settings did it
    # depends on where the sources came from, so the sentence does too.
    if not ((ratio_ok & reported).any() or (sealed & reported).any()):
        raise _plugin_error(
            "no_interior",
            "Nothing here is more than one cell spacing from a source, so there is no route "
            "to measure and nothing is enclosed. "
            + (f"At a gap of {gap_mult:.1f} spacings almost every cell counts as exposed "
               "surface — raise it, or route towards a cell type instead of the surface."
               if source_mode == "surface" else
               f"Either {source_name!r} is everywhere, or the barrier half-width "
               f"({width_mult:.2f} spacings) has swallowed everything between the sources — "
               "lower it, or choose a rarer source type."))

    spacing_med = float(np.median(spacings)) if spacings else np.nan
    dom = in_domain & reported
    seal = sealed & reported
    n_dom = int(dom.sum())
    n_sealed = int(seal.sum())
    # Compartment SIZES are the whole compartment's, in the section; the count is
    # over the compartments that hold a reported cell. Mixing the two is what
    # produced "157 cells ... in 910 separate compartments" under a lasso.
    comp_all = np.concatenate(comp_sizes) if comp_sizes else np.zeros(0)
    null_sealed_n = np.array([int((null_sealed[r] & reported).sum()) for r in range(n_null)],
                             dtype=np.float64)
    null_sealed_m = float(null_sealed_n.mean())

    values, label, units, domain_rng, cmap = _paint(paint, reported, comp, extra, tau,
                                                    seal, ratio_ok)
    series = _series(barrier_name, tau, tau_free, null_tau, ratio_ok & reported,
                     sealed & ratio_ok & reported, null_sealed, n_null)
    table, ranked = _table(levels, codes, reported, in_domain, sealed, null_sealed,
                           extra, n_null)
    summary = _summary(ds, field, barrier_name, source_mode, source_name,
                       n_barrier_tot, n_source_tot, n_sec_used, spacing_med,
                       width_mult, gap_mult, n_null, n_dom, n_sealed, comp_all,
                       null_sealed_n, null_sealed_m, null_comps,
                       extra, null_extra, dom, ranked, paint, subset_on)

    return {"summary": summary,
            "cells": {"values": values, "label": label, "colormap": cmap,
                      "domain": domain_rng, "units": units},
            "series": series,
            "table": table}


# ---------------------------------------------------------------------------
# the four panels
# ---------------------------------------------------------------------------
def _paint(paint, reported, comp, extra, tau, seal, ratio_ok):
    """One field, and a domain that does not spend its ramp on the floor.

    Every mode leaves the cells the barrier costs nothing BLANK rather than
    painting them the bottom of the ramp. On a real section that is most of the
    tissue, and a floor value there flattens the picture into one colour with a
    few specks on it — the opposite of what the statistic found.
    """
    if paint == "detour":
        v = np.where(reported & np.isfinite(extra) & (extra > 0.0), extra, np.nan)
        label, units, cmap, lo = ("Extra path forced by the barrier", "cell diameters",
                                  "inferno", 0.0)
    elif paint == "tau":
        v = np.where(reported & ratio_ok & np.isfinite(tau) & (tau > 1.0), tau, np.nan)
        label, units, cmap, lo = ("Detour factor (geodesic / straight line)", "x",
                                  "inferno", 1.0)
    else:
        v = np.where(seal, comp, np.nan)
        label, units, cmap, lo = "Sealed compartment size", "cells", "inferno", 1.0
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return v, label, units, None, cmap
    # p95 rather than the whole range: every one of these fields is heavy-tailed,
    # and a domain set by its own maximum spends the ramp on a handful of cells.
    hi = float(np.percentile(fin, 95))
    if not (hi > lo):
        hi = lo + max(1.0, abs(lo) * 0.05)
    return v, label, units, [lo, hi], cmap


def _quantiles(fin, n_sealed):
    """Detour quantiles over a population whose top `n_sealed` cells have no route.

    The sealed cells are ranked above every finite detour and returned as NaN,
    so the line stops at the percentile where enclosure begins instead of
    pretending the distribution ends there.
    """
    fin = np.sort(fin[np.isfinite(fin)])
    total = fin.size + int(n_sealed)
    if total == 0 or fin.size == 0:
        return np.full(_QS.size, np.nan)
    pos = _QS / 100.0 * (total - 1)
    out = np.where(pos <= fin.size - 1,
                   np.interp(np.clip(pos, 0, fin.size - 1), np.arange(fin.size), fin),
                   np.nan)
    return out


def _series(barrier_name, tau, tau_free, null_tau, pop, sealed_pop, null_sealed, n_null):
    y_obs = _quantiles(tau[pop], int(sealed_pop.sum()))
    y_free = _quantiles(tau_free[pop], 0)
    per = [_quantiles(null_tau[r][pop], int((null_sealed[r] & pop).sum()))
           for r in range(n_null)]
    stack = np.vstack(per) if per else np.full((1, _QS.size), np.nan)
    # A COLUMN of `stack` is all-NaN whenever every null draw's enclosure begins
    # below that percentile, which is the normal state of affairs for any wall
    # thick enough to seal (observed on the ring fixture from `barrier_width`
    # 1.25 up). `np.nanmin` on such a column emits numpy's "All-NaN slice
    # encountered", a warnings-module RuntimeWarning that `np.errstate` — a
    # floating-point flag context — cannot suppress; the earlier version of this
    # block wrapped it in one and the warning went to stderr anyway. So take the
    # min only over the columns that have something to take a min of.
    have = np.isfinite(stack).any(axis=0)
    y_lo = np.full(_QS.size, np.nan)
    y_hi = np.full(_QS.size, np.nan)
    if have.any():
        y_lo[have] = np.nanmin(stack[:, have], axis=0)
        y_hi[have] = np.nanmax(stack[:, have], axis=0)
    return {
        "title": f"Detour distribution — {barrier_name} as barrier",
        "x_label": "cells ranked by detour (percentile)",
        "y_label": "detour factor (geodesic / straight line)",
        "log_x": False,
        "rules": [1.0],
        "lines": [
            {"label": f"Random barrier, same size (n={n_null}) — low",
             "x": list(_QS), "y": list(y_lo), "color": "#c9ced6"},
            {"label": "Random barrier — high",
             "x": list(_QS), "y": list(y_hi), "color": "#c9ced6", "band": True},
            {"label": f"{barrier_name} (observed)",
             "x": list(_QS), "y": list(y_obs), "color": "#c0392b"},
            {"label": "Open tissue (barrier removed)",
             "x": list(_QS), "y": list(y_free), "color": "#2c3e50", "dashed": True},
        ],
    }


def _table(levels, codes, reported, in_domain, sealed, null_sealed, extra, n_null):
    rows = []
    for i, lv in enumerate(levels):
        m = (codes == i) & reported
        dom = m & in_domain
        n_dom = int(dom.sum())
        if n_dom < _MIN_TABLE_CELLS:
            continue
        n_seal = int((m & sealed).sum())
        nseal = float(np.mean([int((m & null_sealed[r]).sum()) for r in range(n_null)]))
        ex = extra[dom]
        ex = ex[np.isfinite(ex)]
        rows.append([lv, int(m.sum()), n_seal,
                     round(100.0 * n_seal / n_dom, 2),
                     round(100.0 * nseal / n_dom, 2),
                     round(100.0 * float(np.mean(ex > 0.0)), 2) if ex.size else None,
                     round(float(np.median(ex[ex > 0.0])), 2) if np.any(ex > 0.0) else None])
    rows.sort(key=lambda r: (-r[3], -r[2], r[0]))
    ranked = [r for r in rows if r[2] > 0]
    return ({"title": "Enclosure and detour, by cell type",
             "columns": ["Cell type", "Cells", "Enclosed", "% enclosed",
                         "% enclosed, random barrier", "% detoured",
                         "Extra path where detoured"],
             "rows": rows[:5000],
             "sort_by": "% enclosed"}, ranked)


def _pl(n, word):
    """`1 cell`, `910 separate compartments` — a count with a noun that agrees."""
    return f"{int(n):,} {word}" + ("" if int(n) == 1 else "s")


def _summary(ds, field, barrier_name, source_mode, source_name, n_barrier, n_source,
             n_sec, spacing, width_mult, gap_mult, n_null, n_dom, n_sealed, comp_all,
             null_sealed_n, null_sealed_m, null_comps, extra, null_extra,
             dom, ranked, paint, subset_on):
    """The findings of THIS run: the numbers a reader needs to trust the map."""
    ex = extra[dom]
    ex = ex[np.isfinite(ex)]
    nulls = [null_extra[r][dom] for r in range(null_extra.shape[0])]
    nulls = [a[np.isfinite(a)] for a in nulls]
    det = 100.0 * float(np.mean(ex > 0.0)) if ex.size else float("nan")
    ndet = float(np.mean([100.0 * float(np.mean(a > 0.0)) for a in nulls if a.size])) \
        if any(a.size for a in nulls) else float("nan")
    # The median over ALL routed cells is 0 by construction here — nine cells in
    # ten keep their open-tissue path whatever the barrier — so the magnitude is
    # quoted over the cells that are actually pushed off it.
    p50 = float(np.median(ex[ex > 0.0])) if np.any(ex > 0.0) else float("nan")
    np50 = float(np.mean([np.median(a[a > 0.0]) for a in nulls if np.any(a > 0.0)])) \
        if any(np.any(a > 0.0) for a in nulls) else float("nan")
    src = ("the exposed tissue surface" if source_mode == "surface"
           else f"the nearest {source_name}")
    name = field or (ds.meta.get("celltype_obs") or "cell type")

    # A barrier that encloses LESS than its own null is the interesting failure,
    # not a rounding case, so the three directions get three sentences.
    def _fx(x):
        return f"{x:.0f}x" if x >= 10 else f"{x:.1f}x"
    if null_sealed_m <= 0:
        fold_clause = "cells no random barrier of its size encloses at all."
    elif n_sealed == 0:
        # Unused — an empty result takes the other branch — but a ratio of zero
        # must not reach the reciprocal below.
        fold_clause = "nothing, where a random barrier of its size encloses some."
    else:
        fold = n_sealed / null_sealed_m
        if fold >= 1.05:
            fold_clause = f"{_fx(fold)} more than its abundance alone would."
        elif fold > 0.95:
            fold_clause = "about what its abundance alone would."
        else:
            fold_clause = (f"{_fx(1.0 / fold)} LESS than its abundance alone would — "
                           "at these settings this population is not a wall.")
    null_txt = (f"{null_sealed_m:.1f}" if 0 < null_sealed_m < 1 else f"{null_sealed_m:.0f}")
    n_comp = int(comp_all.size)
    biggest = int(comp_all.max()) if n_comp else 0
    med = float(np.median(comp_all)) if n_comp else 0.0
    frac2 = float(np.mean(comp_all <= 2)) if n_comp else 0.0
    # "the largest 72 cells" alone reads as architecture when the median
    # compartment is one cell, which on real tissue it usually is.
    med_txt = f"{med:g} cell" + ("" if med == 1 else "s")
    shape = (f" — median {med_txt}, {100.0 * frac2:.0f}% of them two cells or "
             f"fewer, largest {biggest:,}" if n_comp >= 5
             else f", the largest {_pl(biggest, 'cell')}")
    lines = [
        f"**{barrier_name}** as barrier over `{name}` — {n_barrier:,} barrier cells, "
        f"{n_dom:,} cells routed to {src} ({n_source:,} sources) across {n_sec} section(s).",
        "",
    ]
    if n_sealed:
        lines.append(
            f"**{_pl(n_sealed, 'cell')} ({100.0 * n_sealed / max(1, n_dom):.2f}%) "
            f"{'is' if n_sealed == 1 else 'are'} enclosed** — no route to any source — in "
            f"**{_pl(n_comp, 'separate compartment')}**{shape}. "
            f"Same number of cells relabelled at random: {null_txt} enclosed "
            f"(range {null_sealed_n.min():.0f}-{null_sealed_n.max():.0f} over {n_null} draws"
            + (f", {_pl(round(float(null_comps.mean())), 'compartment')}"
               if round(float(null_comps.mean())) >= 1 else "")
            + f"), so this barrier encloses {fold_clause}")
        if subset_on:
            lines.append("")
            lines.append(
                "Only the selection is counted and painted; the graph is still the whole "
                "section, so a compartment's SIZE is its full size there even when part of "
                "it lies outside the lasso.")
    else:
        lines.append(
            f"**Nothing is enclosed**: every cell keeps a route to {src}. Random barriers of "
            f"the same size enclosed {null_txt}. Enclosure is sensitive to the barrier "
            f"half-width, and this run used {width_mult:.2f} spacings: on a synthetic wall "
            "exactly one cell thick, nothing seals below about 1.25.")
    if ranked:
        top = "; ".join(f"{r[0]} ({r[3]:.1f}%)" for r in ranked[:5])
        lines.append("")
        lines.append(f"Most enclosed: **{top}**.")
    lines.append("")
    # Every one of det / p50 / ndet / np50 is NaN when the population behind it
    # is empty, and an empty population is a FINDING here — "nothing in the
    # reported set is detoured at all" — not a number to print. The pairs move
    # together: a positive detoured FRACTION guarantees there is a detour to
    # take a median of, so p50 is NaN exactly when det is 0.0 and np50 exactly
    # when ndet is, and branching on the two fractions covers all four. Both
    # cases reproduce on the ring fixture — a coherent wall detours nobody, and
    # a lasso of its interior leaves nothing routable at all — which is how a
    # literal "median nan extra cell diameters" reached the prose.
    n_routed = int(ex.size)
    if not np.isfinite(det):
        lines.append(
            f"**Detour is not measured for this selection**: not one cell in it keeps a route "
            f"to {src} — every one is enclosed — so there is no path length to hold against "
            f"the open tissue. "
            + (f"Random barriers of the same size detoured **{ndet:.1f}%** of the cells they "
               f"left routable (median {np50:.1f} extra cell diameters)."
               if np.isfinite(ndet) and ndet > 0.0 else
               "Random barriers of the same size detoured none of it either."
               if np.isfinite(ndet) else
               "Random barriers of the same size left nothing routable here either.")
            + " The enclosure count above is the whole finding for this selection.")
    elif det <= 0.0:
        lines.append(
            f"**Not one of the {n_routed:,} routed cells is detoured at all**: every route "
            f"that survives the barrier is exactly the length it has with the barrier "
            f"removed, so the extra path has no distribution and no median to quote. "
            + (f"Random barriers of the same size detour **{ndet:.1f}%** of the same cells "
               f"(median {np50:.1f} extra cell diameters) — which is the wall working as a "
               f"wall: it leaves the open tissue alone and seals a compartment instead, where "
               f"the same cells scattered are a fine maze that slows everyone a little. Read "
               f"the enclosure count, not the detour."
               if np.isfinite(ndet) and ndet > 0.0 else
               "Random barriers of the same size detour no cell either."
               if np.isfinite(ndet) else
               "Random barriers of the same size left nothing routable to compare against."))
    elif np.isfinite(ndet) and ndet > det:
        lines.append(
            f"Detour, by contrast, is UNREMARKABLE: only **{det:.1f}%** of routed cells are "
            f"pushed off their open-tissue path (median {p50:.1f} extra cell diameters where "
            f"they are), against **{ndet:.1f}%** (median {np50:.1f}) under the random barrier. "
            f"It scores BELOW its own null because "
            f"a coherent wall leaves most of the tissue untouched and seals a compartment "
            f"instead, where the same cells scattered are a fine maze that slows everyone a "
            f"little. Read the enclosure count, not the median.")
    elif np.isfinite(ndet) and ndet > 0.0:
        lines.append(
            f"**{det:.1f}%** of routed cells are pushed off their open-tissue path (median "
            f"{p50:.1f} extra cell diameters where they are), against **{ndet:.1f}%** (median "
            f"{np50:.1f}) under a random barrier of the same size, so this barrier obstructs "
            f"more than its abundance alone would.")
    else:
        # det > 0 with a null that has no detour to report at all: the fraction
        # stands, the comparison does not, and neither does np50.
        lines.append(
            f"**{det:.1f}%** of routed cells are pushed off their open-tissue path (median "
            f"{p50:.1f} extra cell diameters where they are), against "
            + ("no detoured cell at all under a random barrier of the same size"
               if np.isfinite(ndet) else
               "a random barrier of the same size that left nothing routable to compare")
            + ", so this barrier obstructs more than its abundance alone would.")
    if paint == "compartment" and not n_sealed:
        lines.append("")
        lines.append("The map is blank: with nothing enclosed there are no compartments to "
                     "paint. Switch it to the detour field.")
    lines.append("")
    lines.append(
        f"Method: pruned Delaunay graph of cell centroids, median spacing {spacing:.1f} units; "
        f"a path keeps {width_mult:.2f} spacings ({width_mult * spacing:.1f} units) clear of a "
        f"barrier centre and may cross gaps up to {gap_mult:.1f} spacings. Ratios are left "
        f"blank within one spacing of a source. One barrier type per run.")
    return "\n".join(lines)
