"""Ripley's cross-L function: at what DISTANCE does cell type B sit around A?

What this computes
------------------
The bivariate (cross) K-function counts, for every cell of type A, how many
cells of type B lie within distance r, and normalises that count by what a
uniform B population of the same size would have put there:

    K_AB(r) = (|W| / (n_A n_B)) * sum_i sum_j 1[ d(a_i, b_j) <= r ]

Under complete spatial randomness E[K_AB(r)] = pi r^2 at every r, so K is
reported in Besag's variance-stabilising L form, centred:

    L_AB(r) - r = sqrt(K_AB(r) / pi) - r

which sits on zero for an unstructured pattern, rises where B crowds around A,
and falls where B avoids it. The transform matters for reading a plot, not for
the test: sqrt flattens the r^2 growth so the eye is not looking at a parabola,
and the variance of L is roughly constant in r so a deviation at r = 20 and one
at r = 200 are visually comparable.

**The product is the peak, not the value.** The radius at which L - r is
largest is the scale over which B accumulates around A; a single-radius
statistic (PointillViz's built-in colocalisation, and every "fraction of A with a B within
50 um" number) has to be told that scale in advance and cannot discover it. On
a synthetic pattern where every B is dropped uniformly inside a disc of radius
R about some A, the peak of L - r lands exactly at R: below R the curve grows
like r*(sqrt(1 + c) - 1) and above R like sqrt(r^2 + c) - r, and the two
branches meet at R. `tests/tool_ripley_cross_l_test.py` asserts that recovery,
which is the sharpest available check that the normalisation is right.

The three things a textbook formula leaves out
----------------------------------------------
**1. Edge correction — border ("reduced sample") method.** A cell 10 um from
the section edge has only part of its r-disc inside the tissue, so an
uncorrected count is short by the missing wedge, K is biased DOWN, and every
real section looks repulsive at large r. This module uses the border correction:
at radius r, only A cells whose distance to the window boundary EXCEEDS r
contribute, and the normaliser counts only those cells:

    K_hat(r) = sum_s [ sum_{i in A_s, b_i > r} n_i(r) ]
             / sum_s [ (n_B,s / |W_s|) * m_s(r) ]        m_s(r) = #{i: b_i > r}

Every contributing disc is then wholly inside the window, so the estimator is
ratio-unbiased with no weighting at all. What it costs: cells near the edge are
discarded rather than reweighted, so the effective sample shrinks as r grows
(at r = a quarter of the shorter side of a square window, ~ 3/4 of the cells
remain; at half, none do) and the curve gets noisier at the right-hand end.
Ripley's isotropic correction keeps every cell by weighting each pair by the
reciprocal of the fraction of its circle inside W, which is strictly more
efficient — it was rejected here because that weight has a closed form only for
a rectangle or a disc, and this tool must also serve a convex hull; a wrong
weight is worse than an honest smaller sample. The isotropic and translation
corrections agree with the border correction to within their standard errors
whenever r stays well inside the window, which is the regime this tool asks the
user to stay in (see the r_max warning in `_window`).

**2. The window |W|.** It is not recorded anywhere in the dataset, so it is
estimated from the cells themselves, per section, from ALL cells in play — not
from A and B alone, or picking a rare cell type would shrink the window and
inflate its own K. `window="hull"` takes the convex hull of those cells,
`window="bbox"` their axis-aligned bounding box. Both assume a CONVEX window: a
tissue with a lumen (a colon ring, a ventricle) has its hole counted as
observation area, |W| comes out too large, and every K with it. That bias is
absolute, not relative — see the README — and it is the largest uncontrolled
term in any number this tool prints.

**3. The null.** The envelope comes from randomly relabelling which of the
OBSERVED cells are B, never from scattering uniform random points. Real tissue
is nowhere near uniform: cells sit in crypts, follicles and vessel walls, so a
uniform-CSR null is rejected by essentially every pair of cell types at
essentially every radius, and a test that always rejects has no content. Random
labelling holds the tissue's own inhomogeneous density fixed and asks the
question a biologist actually means: given where cells are AT ALL, is B
specifically near A? A is held fixed and n_B cells are drawn without
replacement, per section, from every in-play cell that is not A (unlabelled
cells included: they are real positions, and the null is about positions).

Holding A fixed is also what makes the envelope affordable: the border
eligibility mask and the per-section normaliser depend only on A and on n_B,
so they are computed ONCE and the entire denominator is shared by the observed
curve and every simulation. A simulation therefore costs exactly one KD-tree
build over n_B points plus one dual-tree count.

Speed
-----
Pair counting is `cKDTree.count_neighbors` between a type-A tree and a type-B
tree over the whole radius vector in one call, never a distance matrix: the
heaviest pair below is 15,060 x 14,613 = 2.2e8 distances, 1.8 GB as float64
for one pair of cell types out of 73, and the whole section is 1.8e10.

Measured on datasets/co_m3_r2 (135,232 cells, 73 cell types, one section),
8 threads, shipped defaults (30 radii to 150 units, 39 simulations, hull
window), warm page cache:

    ds.kdtrees()                                        0.41 s   (paid once)
    IgA plasma cells (188) x Fcrl5+ B cells (170)       0.05 s
    Stem/TA cells (4,034) x Pericytes (537)             0.13 s
    Smooth muscle (15,060) x Int. cells of Cajal (2,962) 0.54 s
    Smooth muscle (15,060) x Mature enterocytes I (14,613) 1.52 s
      the same pair at 199 simulations                  6.94 s
      the same pair out to r = 500                      4.22 s

So the cost is set by n_A * n_B and by r_max (pairs grow like r^2), not by the
size of the dataset, and the two most abundant cell types in a 135k-cell
section stay inside two seconds with the defaults.

Determinism
-----------
The simulation seed is derived from the parameters (SHA-1 of a canonical
string), NOT from `random.seed()` on the clock and NOT from Python's `hash()`,
which is salted per process by PYTHONHASHSEED and would make two runs of the
same tool disagree. So the same form gives the same envelope, which is what a
figure in a paper needs: a reviewer re-running the tool sees the plot from the
paper. The cost is real and worth stating — the envelope is one draw of a
random object, and a user who wants to know how stable a marginal result is has
to change the simulation count and look again, because they cannot re-roll it.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from pointillviz_tool import PluginError

#: Below this many cells of either type the curve is a handful of points'
#: worth of noise dressed as a statistic: at n_A = 10 the observed L is
#: driven by which side of a crypt three cells happened to fall on, and the
#: relabelling envelope is so wide that nothing can leave it. Refuse instead
#: of drawing it.
MIN_CELLS_PER_TYPE = 20

#: Ripley's rule of thumb, as stated in Baddeley et al. (2015): beyond about a
#: quarter of the shorter side of the window, the border correction has thrown
#: away most of the data and the estimate is unstable. Exceeding it is a
#: warning, not a refusal — a user studying a long-range gradient may mean it.
R_MAX_WINDOW_FRACTION = 0.25

#: Convex-hull border distances are a (points x facets) reduction. Blocked so a
#: 60k-cell section against a 1,000-facet hull is 8k x 1k floats at a time
#: rather than a 480 MB temporary.
_HULL_BLOCK = 8192

_OBS_COLOR = "#4363d8"
_ENV_COLOR = "#9aa5b1"


# ===========================================================================
# Entry point
# ===========================================================================
def run(ds, params):
    if ds.is_3d:
        raise PluginError(
            "not_2d",
            "This tool is the 2D cross-L function (L = sqrt(K/pi) - r). On a 3D "
            "dataset both that transform and the area-based window estimate are "
            "wrong; use it on a 2D section.")

    field = (params.get("field") or "").strip() or None
    levels, codes = ds.categorical(field)
    a_label, b_label = _resolve_pair(levels, codes, params, ds)
    code_a = levels.index(a_label)
    code_b = levels.index(b_label)

    # Indexed, not `.get(..., default)`: the manifest owns the defaults and the
    # host has already coerced and clamped them, so a second copy here would be
    # a second place for them to drift. The window IS re-checked, because an
    # unrecognised value would otherwise fall through to the bounding box and
    # answer a question nobody asked.
    n_sims = int(params["n_sims"])
    n_steps = int(params["n_steps"])
    r_max = float(params["max_radius"])
    window = str(params["window"])
    if window not in ("hull", "bbox"):
        raise PluginError("bad_window", f"window must be 'hull' or 'bbox', not {window!r}.")

    # Radii start at r_max/n_steps, not at 0: K(0) = 0 by construction and
    # L(0) - 0 = 0 carries no information, it just pins the plot to the origin.
    radii = np.linspace(r_max / n_steps, r_max, n_steps, dtype=np.float64)

    sections = _sections(ds, codes, code_a, code_b, window, radii)
    n_a = sum(s["a"].shape[0] for s in sections)
    n_b = sum(s["b"].shape[0] for s in sections)
    if n_a < MIN_CELLS_PER_TYPE or n_b < MIN_CELLS_PER_TYPE:
        raise PluginError(
            "too_few_cells",
            f"{a_label} has {n_a} cell(s) and {b_label} has {n_b} with usable "
            f"coordinates; this tool needs at least {MIN_CELLS_PER_TYPE} of each. "
            "Below that the cross-L curve is noise and its envelope covers "
            "everything. Pick more abundant cell types, or a coarser annotation.")

    # The null draws B from the OTHER cells present. When there are no other
    # cells -- a dataset (or a lasso) holding only these two categories -- that
    # draw has exactly one outcome, the envelope collapses onto the observed
    # curve, and every result reads "inside the envelope". Silently returning a
    # test that cannot reject is worse than not running.
    pool_total = sum(s["pool"].shape[0] for s in sections)
    if pool_total <= n_b:
        raise PluginError(
            "no_null_room",
            f"Every cell that is not {a_label} is already {b_label} "
            f"({n_b:,} of them), so relabelling can only reproduce the observed "
            "pattern and the envelope would be the curve itself. This tool's "
            "null draws B from the other cells in the section; pick a column "
            "with more categories, or widen the selection to include them.")

    ds.log(f"{a_label}: {n_a:,} cells   {b_label}: {n_b:,} cells   "
           f"{len(sections)} section(s)")
    ds.log(f"window = {window}, total area = {sum(s['area'] for s in sections):,.0f} "
           f"sq. units, radii {radii[0]:.3g} to {radii[-1]:.3g}")
    notes = _notes(sections, r_max, n_b, b_label)
    for msg in notes:
        ds.log(msg)

    # The denominator is a property of A, of n_B and of the window — none of
    # which the null disturbs — so it is built once and reused by every
    # simulation. sum over sections of (n_B,s / |W_s|) * m_s(r).
    denom = np.zeros(radii.size, dtype=np.float64)
    for s in sections:
        denom += (s["b"].shape[0] / s["area"]) * s["m"]
    estimable = denom > 0
    if not estimable.any():
        raise PluginError(
            "window_too_small",
            f"No {a_label} cell lies further than {radii[0]:.3g} units from the "
            "section boundary, so the border-corrected estimator has nothing to "
            "average. Reduce the maximum radius.")

    observed = _centred_l(_pair_counts(sections, radii, [s["b_tree"] for s in sections]),
                          denom, radii, estimable)

    ds.progress(0.05, "counting observed pairs")
    lo, hi = _envelope(ds, sections, radii, denom, estimable, n_sims,
                       (field or "cell_type", a_label, b_label, r_max, n_steps, window))
    ds.progress(1.0, "done")

    return _report(a_label, b_label, radii, observed, lo, hi, estimable,
                   sections, n_a, n_b, n_sims, window, r_max, field, notes)


# ===========================================================================
# Which two types
# ===========================================================================
def _resolve_pair(levels, codes, params, ds):
    """(label_a, label_b), refusing the degenerate cases in the user's words."""
    counts = np.bincount(codes[codes >= 0], minlength=len(levels))
    want_a = (params.get("type_a") or "").strip()
    want_b = (params.get("type_b") or "").strip()

    # No choice made (the host passes None for an unset optional parameter):
    # fall back to the two most abundant labelled types, ties broken by name so
    # that "what the tool does by default" is a fixed answer and not a
    # dictionary-ordering accident.
    if not want_a or not want_b:
        ranked = [levels[i] for i in
                  sorted(range(len(levels)), key=lambda i: (-int(counts[i]), levels[i]))
                  if counts[i] >= MIN_CELLS_PER_TYPE]
        if len(ranked) < 2:
            raise PluginError(
                "too_few_types",
                f"Fewer than two categories have {MIN_CELLS_PER_TYPE}+ cells, so "
                "there is no pair to test. Choose a coarser annotation column.")
        if not want_a:
            want_a = next(lv for lv in ranked if lv != want_b)
        if not want_b:
            want_b = next(lv for lv in ranked if lv != want_a)
        ds.log("no cell types given; defaulting to the most abundant available: "
               f"{want_a} and {want_b}")

    for label, which in ((want_a, "A"), (want_b, "B")):
        if label not in levels:
            near = ", ".join(sorted(levels)[:6])
            raise PluginError(
                "unknown_type",
                f"Cell type {which} ({label!r}) is not a level of this column. "
                f"It has {len(levels)} levels, e.g. {near}.")
    if want_a == want_b:
        # The univariate K is a different estimator, not a special case: the
        # sum must drop the n self-pairs at distance 0 and the normaliser
        # becomes n(n-1) rather than n_A n_B, or K is inflated by |W|/n at
        # every radius. Worse, the random-labelling null collapses — there is
        # no second label left to permute — so the honest null for A-vs-A is a
        # point-process null (CSR or a fitted inhomogeneous Poisson), which is
        # exactly the null this tool argues is meaningless on tissue. Refusing
        # is more useful than silently answering a different question.
        raise PluginError(
            "same_type",
            f"Cell type A and B are both {want_a!r}. The univariate K-function "
            "needs a different normalisation (self-pairs excluded, n(n-1)) and a "
            "different null — random labelling has nothing to permute when there "
            "is only one label. Pick two different cell types.")
    return want_a, want_b


# ===========================================================================
# Per-section geometry
# ===========================================================================
def _sections(ds, codes, code_a, code_b, window, radii):
    """One record per section: A points, B tree, window area, border groups.

    `ds.kdtrees()` is the source of truth for which cells are in which section
    and which have usable coordinates — one tree per section because sections
    share a canvas and a single tree would hand a cell a neighbour from another
    specimen. Its trees are over ALL cell types, so they are not the trees this
    tool counts with; what is taken from it is the per-section `rows`.
    """
    xy = ds.coords[:, :2].astype(np.float64)
    keep = None
    if ds.subset is not None:
        # A lasso redefines the observation window as well as the sample: the
        # window below is estimated from whatever cells are in play, so a
        # selection is honoured by restriction alone.
        keep = np.zeros(ds.n_cells, dtype=bool)
        keep[ds.subset] = True

    out = []
    for part in ds.kdtrees():
        rows = part["rows"]
        if keep is not None:
            rows = rows[keep[rows]]
        if rows.size < 3:
            continue
        a_rows = rows[codes[rows] == code_a]
        b_rows = rows[codes[rows] == code_b]
        if a_rows.size == 0 or b_rows.size == 0:
            continue                      # a section holding only one of the two
        area, border = _window(xy[rows], window)
        if area <= 0:
            continue
        a_pts = xy[a_rows]
        # Border distance of each A cell, from the section-wide window.
        b_dist = border(a_pts)
        groups, m = _border_groups(a_pts, b_dist, radii)
        out.append({
            "section": part["section"],
            "a": a_pts, "b": xy[b_rows],
            "b_tree": _tree(xy[b_rows]),
            "area": float(area), "groups": groups, "m": m,
            "pool": _null_pool(xy, rows, codes, code_a),
            "extent": (float(xy[rows, 0].max() - xy[rows, 0].min()),
                       float(xy[rows, 1].max() - xy[rows, 1].min())),
        })
    if not out:
        raise PluginError(
            "no_shared_section",
            "No section contains cells of both types, so there is no pair of "
            "cells to count. Cross-L is computed within a section — two "
            "specimens never share a distance.")
    return out


def _window(pts, mode):
    """(area, border_distance_fn) for one section's observation window.

    Both windows are convex, which is what makes "the whole disc is inside W"
    testable with a single number per cell: for a convex window the distance
    from an interior point to the boundary is the minimum distance to the
    bounding lines, because the window is the intersection of half-planes.
    """
    if mode == "hull":
        try:
            from scipy.spatial import ConvexHull
        except ImportError:                              # pragma: no cover
            raise PluginError("no_scipy", "scipy is required for the hull window.")
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Degenerate input (all cells collinear, or three cells). The
            # bounding box is always defined, so fall back rather than fail a
            # whole run on one thin section.
            return _window(pts, "bbox")
        eq = np.asarray(hull.equations, dtype=np.float64)   # [nx, ny, offset], outward unit normals

        def border(q):
            out = np.empty(q.shape[0], dtype=np.float64)
            for i in range(0, q.shape[0], _HULL_BLOCK):
                blk = q[i:i + _HULL_BLOCK]
                d = -(blk @ eq[:, :2].T + eq[:, 2][None, :])
                out[i:i + blk.shape[0]] = d.min(axis=1)
            return np.maximum(out, 0.0)

        return float(hull.volume), border          # 2D "volume" is the area

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)

    def border(q):
        return np.maximum(np.minimum(q - lo, hi - q).min(axis=1), 0.0)

    return float((hi[0] - lo[0]) * (hi[1] - lo[1])), border


def _border_groups(a_pts, b_dist, radii):
    """Trees of A cells bucketed by how many radii they are eligible for.

    The border correction needs, at each radius, the pair count over a
    DIFFERENT subset of A — those further than r from the edge — and
    `count_neighbors` counts one whole tree against another. Rebuilding an A
    tree per radius per simulation is the obvious implementation and is
    n_steps times too slow. The subsets are nested, so instead each A cell is
    bucketed by g = #{k : radii[k] < b_dist}, the number of leading radii it
    survives; a bucket's tree is then counted once against B over radii[:g] and
    added into those radii. Buckets are built once and reused by every
    simulation, since the null never moves an A cell.
    """
    g = np.searchsorted(radii, b_dist, side="left")     # radii strictly below b_dist
    groups = []
    for value in np.unique(g):
        value = int(value)
        if value == 0:
            continue                     # eligible at no radius: contributes nothing
        sel = np.flatnonzero(g == value)
        groups.append((value, _tree(a_pts[sel])))
    m = np.zeros(radii.size, dtype=np.float64)
    for value, tree in groups:
        m[:value] += tree.n
    return groups, m


def _null_pool(xy, rows, codes, code_a):
    """The positions a simulated B cell may occupy in this section.

    Every in-play cell that is not type A — including unlabelled cells, which
    are real cells at real positions and belong in a null about positions.
    Type A is excluded so that a simulated pattern keeps the observed one's
    property that a cell is not both A and B; letting a pseudo-B land on an A
    would put a pair at distance zero and inflate the envelope at every radius.
    """
    pool = rows[codes[rows] != code_a]
    return xy[pool]


def _tree(pts):
    from scipy.spatial import cKDTree
    return cKDTree(pts)


# ===========================================================================
# The estimator
# ===========================================================================
def _pair_counts(sections, radii, b_trees):
    """Border-corrected pair count, summed over sections.

    Sections are pooled by summing numerators and summing normalisers, never by
    averaging per-section L values: sections differ in n_A, n_B and area, so a
    mean of L curves weights a 900-cell section like a 90,000-cell one and is
    not an estimator of anything. The ratio of the sums is the pooled
    estimator, and it reduces exactly to the single-section formula when there
    is one section.
    """
    total = np.zeros(radii.size, dtype=np.float64)
    for sec, b_tree in zip(sections, b_trees):
        for value, a_tree in sec["groups"]:
            counts = a_tree.count_neighbors(b_tree, radii[:value], cumulative=True)
            total[:value] += np.asarray(counts, dtype=np.float64)
    return total


def _centred_l(numer, denom, radii, estimable):
    """L_AB(r) - r, NaN where the border correction left nothing to divide by."""
    k = np.full(radii.size, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=k, where=estimable)
    return np.sqrt(np.maximum(k, 0.0) / math.pi) - radii


def _envelope(ds, sections, radii, denom, estimable, n_sims, key):
    """Pointwise min/max of the centred L over `n_sims` random relabellings."""
    rng = np.random.default_rng(_seed(key))
    sims = np.empty((n_sims, radii.size), dtype=np.float64)
    for t in range(n_sims):
        trees = []
        for sec in sections:
            pool = sec["pool"]
            # Per section, and with the observed n_B held fixed: the envelope
            # is then a test of ARRANGEMENT only. Redrawing n_B across the
            # whole slide instead would let a simulation move B cells between
            # sections, so the envelope would also absorb differences in
            # abundance between specimens — a real effect, but not the one the
            # curve is about.
            pick = rng.choice(pool.shape[0], size=sec["b"].shape[0], replace=False)
            trees.append(_tree(pool[pick]))
        sims[t] = _centred_l(_pair_counts(sections, radii, trees),
                             denom, radii, estimable)
        ds.progress(0.05 + 0.9 * (t + 1) / n_sims,
                    f"simulation {t + 1} of {n_sims}")
    return sims.min(axis=0), sims.max(axis=0)


def _seed(key):
    """A seed that is a function of the parameters and of nothing else.

    SHA-1 rather than `hash()`: PYTHONHASHSEED randomises str hashing per
    process, so `hash()` would give a different envelope on every run, for a
    reason nobody would find.
    """
    text = "|".join(str(k) for k in key)
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")


def _notes(sections, r_max, n_b, b_label):
    """Caveats that apply to THIS run. Logged and repeated in the summary,
    because a caveat only the run log knows is a caveat nobody reads."""
    out = []
    shortest = min(min(s["extent"]) for s in sections)
    if r_max > R_MAX_WINDOW_FRACTION * shortest:
        out.append(
            f"Maximum radius {r_max:.3g} is over {R_MAX_WINDOW_FRACTION:.0%} of "
            f"the shortest section extent ({shortest:.3g}); the right-hand end "
            "of the curve is unstable.")
    pool_total = sum(s["pool"].shape[0] for s in sections)
    if pool_total < 2 * n_b:
        out.append(
            f"{b_label} is {n_b / pool_total:.0%} of the cells the null draws "
            "from, so the test is conservative.")
    return out


# ===========================================================================
# Reporting
# ===========================================================================
def _report(a_label, b_label, radii, observed, lo, hi, estimable, sections,
            n_a, n_b, n_sims, window, r_max, field, notes):
    above = estimable & (observed > hi)
    below = estimable & (observed < lo)

    rows = []
    for k in range(radii.size):
        rows.append([
            round(float(radii[k]), 4),
            _num(observed[k]), _num(lo[k]), _num(hi[k]),
            "above" if above[k] else ("below" if below[k] else ""),
        ])

    series = {
        "title": f"Cross-L: {b_label} around {a_label}",
        "x_label": "radius r (dataset units)",
        "y_label": "L(r) - r",
        # Band lines first and adjacent, so a renderer that fills between
        # consecutive `band` lines gets the envelope and not the observation.
        "lines": [
            {"label": f"Envelope low ({n_sims} relabellings)",
             "x": radii.tolist(), "y": _jsonable(lo),
             "color": _ENV_COLOR, "band": True},
            {"label": "Envelope high",
             "x": radii.tolist(), "y": _jsonable(hi),
             "color": _ENV_COLOR, "band": True},
            {"label": f"{b_label} around {a_label}",
             "x": radii.tolist(), "y": _jsonable(observed),
             "color": _OBS_COLOR},
        ],
        "rules": [0.0],
    }
    return {
        "summary": _summary(a_label, b_label, radii, observed, lo, hi, estimable,
                            above, below, sections, n_a, n_b, n_sims, window,
                            r_max, field, notes),
        "series": series,
        "table": {"title": f"Cross-L of {b_label} around {a_label}, per radius",
                  "columns": ["r", "L(r) - r", "envelope low", "envelope high",
                              "outside envelope"],
                  "rows": rows, "sort_by": "r"},
    }


def _summary(a_label, b_label, radii, observed, lo, hi, estimable, above, below,
             sections, n_a, n_b, n_sims, window, r_max, field, notes):
    usable = np.flatnonzero(estimable)
    peak = int(usable[np.argmax(observed[usable])])
    peak_r, peak_v = float(radii[peak]), float(observed[peak])

    # Cell-type labels routinely end in "cells" ("Fcrl5+ B cells"), so nothing
    # here appends the word: "Fcrl5+ B cells cells" is how a generated summary
    # announces that nobody read its output.
    where = ("one section" if len(sections) == 1
             else f"{len(sections)} pooled sections")
    lines = [
        f"**{b_label} around {a_label}** — Ripley's cross-L over {where}, "
        f"{n_a:,} {a_label} against {n_b:,} {b_label}.",
        "",
    ]
    if above.any():
        r0, r1 = float(radii[above][0]), float(radii[above][-1])
        lines.append(
            f"**Attraction** over r = {r0:.3g} to {r1:.3g} "
            f"({int(above.sum())} of {radii.size} radii above the envelope).")
    if below.any():
        r0, r1 = float(radii[below][0]), float(radii[below][-1])
        lines.append(
            f"**Exclusion** over r = {r0:.3g} to {r1:.3g} "
            f"({int(below.sum())} of {radii.size} radii below it).")
    if not above.any() and not below.any():
        lines.append(
            f"**No detected association** — the curve stays inside the envelope "
            f"at all {radii.size} radii. A weak or very-short-range effect can "
            "hide inside an envelope this wide.")
    if below.any():
        trough = int(usable[np.argmin(observed[usable])])
        lines.append(f"Deepest L(r) - r = **{float(observed[trough]):.3g}** at "
                     f"r = {float(radii[trough]):.3g}.")
    if below[peak]:
        tail = " — below the envelope even at its highest."
    elif not above[peak]:
        tail = " — inside the envelope there, so read it as the shape of the curve."
    elif peak == radii.size - 1:
        tail = (" — still rising at the largest radius asked for; raise the "
                "maximum radius until the curve turns over.")
    else:
        tail = " — the scale the association acts over."
    lines.append(f"Peak L(r) - r = **{peak_v:.3g}** at **r = {peak_r:.3g}**" + tail)
    # K = 0 gives L - r = -r exactly, and that is worth saying in words: it is
    # the difference between "fewer than expected" and "none at all".
    hard = np.flatnonzero(observed + radii != 0.0)
    n_hard = int(hard[0]) if hard.size else radii.size
    if n_hard:
        lines.append("")
        lines.append(
            f"No {b_label} lies within {float(radii[n_hard - 1]):.3g} units of "
            f"any {a_label}, so the curve is exactly -r out to there.")
    lines.append("")
    lines.append(
        f"Method: border-corrected, {window} window "
        f"({sum(s['area'] for s in sections):,.0f} sq. units), {radii.size} radii "
        f"to {r_max:.3g}; envelope = min/max of {n_sims} random relabellings, a "
        f"{2.0 / (n_sims + 1):.1%} two-sided test at one radius"
        + (f", grouping column `{field}`." if field else ", cell-type column."))
    if not estimable.all():
        first = float(radii[np.flatnonzero(~estimable)[0]])
        lines.append("")
        lines.append(
            f"Radii from {first:.3g} up are blank: no {a_label} lies that far "
            "from a section boundary for the border correction to use.")
    for note in notes:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


def _num(v):
    return None if not np.isfinite(v) else round(float(v), 4)


def _jsonable(arr):
    return [None if not np.isfinite(v) else float(v) for v in arr]
