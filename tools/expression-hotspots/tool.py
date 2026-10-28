"""Expression Hotspots — the local Getis-Ord Gi* statistic, one z-score per cell.

    run(ds, params) -> {"summary": <markdown>, "cells": {...}, "table": {...}}

WHAT THE STATISTIC SAYS
-----------------------
For one gene, Gi* asks of every cell: does the neighbourhood around **and
including** this cell carry more of the gene, or less, than a neighbourhood of
that size drawn at random from this tissue would?

    Gi*(i) = [ SUM_j w_ij x_j  -  Xbar * SUM_j w_ij ]
             / ( S * sqrt( ( n * SUM_j w_ij^2 - (SUM_j w_ij)^2 ) / (n - 1) ) )

    Xbar = SUM_j x_j / n        S = sqrt( SUM_j x_j^2 / n - Xbar^2 )

The neighbour set contains cell i itself. That star is not a detail — it is the
difference between two different questions. Without i, Gi measures the tissue
AROUND a cell; with i, Gi* measures the PLACE the cell sits in. So a single
bright cell surrounded by dead tissue does not read as a hotspot here, and a
dim cell inside a bright region does not read as a coldspot: both are places,
and the statistic answers about the place. That is what makes the output a map
rather than an outlier list.

WHY THERE IS NO PERMUTATION
---------------------------
Under the randomisation null — the gene's values reshuffled at random over the
locations — the numerator is a sum of m_i draws without replacement from a
population of n, whose exact mean and variance are the two terms written above.
Standardised by them, Gi* is asymptotically N(0, 1). So the z-score IS the
significance and a permutation loop would spend minutes re-estimating a
quantity already known in closed form. This is the same reasoning
PointillViz's built-in Moran's I gives for the global statistic and Anselin's local moments, and
the reason is the same: at 135k cells, 999 shuffles per gene is hours of
arithmetic for a number that has an exact expression.

AND ITS ASSUMPTION, HONESTLY
----------------------------
"Asymptotically" is doing real work in that sentence. The normal approximation
is a central-limit statement about a sum of m_i values, and it is poor when
those values are dominated by a handful of non-zeros — which is exactly the
regime of a sparse gene in single-cell data. A gene expressed in 0.5% of cells
gives a 16-cell neighbourhood a sum that is 0 or one spike; its distribution is
visibly discrete and its upper tail is heavier than a Gaussian's, so a p-value
read off the normal curve is optimistic. This module therefore measures the
gene's expressing fraction before it reports anything: it REFUSES a gene with
no expression at all (below, S = 0 and every Gi* would be 0/0), refuses one
seen in fewer than a handful of cells, and WARNS in the summary the user reads
whenever the typical neighbourhood holds fewer than
SPARSE_NEIGHBOURHOOD_HITS expressing cells. It never quietly reports a q-value
it does not believe.

THE WEIGHTS: BINARY, AND WHY THE CHOICE IS FREE
-----------------------------------------------
w_ij = 1 for j in the k-NN set of i plus i itself, 0 otherwise. Then
SUM_j w_ij = SUM_j w_ij^2 = m_i, the neighbourhood size, and the denominator
collapses to S * sqrt( m_i (n - m_i) / (n - 1) ).

Row-standardising instead (w_ij = 1/m_i, which PointillViz's built-in LISA uses)
would change nothing: it scales the numerator by 1/m_i and, through
SUM w^2 = 1/m_i, the denominator by exactly the same 1/m_i. Gi* is invariant —
tests/tool_expression_hotspots_test.py asserts that numerically rather than
taking it on faith. Binary is used because it makes both weight sums exact
small integers, so the variance term carries no division at all. moran.py
row-standardises for a reason that does not apply here: its spatial lag has to
be a neighbour MEAN for the LISA quadrants to be read against, while Gi*
standardises the weights away by construction.

The k-NN set is NOT symmetrised. The formula only ever touches row i, so
symmetry buys nothing statistically, and symmetrising a 10.4M-cell graph means
sorting ~150M index pairs — real memory for an identical answer. Using
ds.knn(k) exactly as it comes also means the graph is the one cached on disk
and shared with every other tool, so this tool pays nothing to build it.

THE POPULATION — WHICH CELLS ARE IN n, Xbar AND S
-------------------------------------------------
Every cell the neighbour graph placed, and no others. `ds.knn` returns -1 for a
cell with no coordinates, for a cell outside an isolated section, and for a
cell alone in its section; none of those has a neighbourhood, so none of them
can carry a Gi*, and all of them are NaN in the output.

The half of that which is easy to get wrong is the OTHER half: they must also
be out of n, Xbar and S. The null Gi* is tested against is "these values,
reshuffled over these locations", so Xbar has to be the mean of the same
population the neighbours are drawn from. Taking Xbar over all n_cells while
summing only over graph cells biases every numerator by m_i times the
difference — a silent, uniform tint over the whole map that looks like a
finding. The test file plants that bug deliberately (a dataset with extra
coordinate-less cells at a wildly different expression level) and asserts that
the Gi* of every shared cell is unchanged to 1e-9.

When the user has lassoed a subset, the subset IS the population: n, Xbar, S
and the neighbour sums are all restricted to it, and a listed neighbour outside
the selection is dropped from the neighbourhood. A hotspot is then hot relative
to the selection, which is the only null the numbers support; the summary says
so.

NOT LOCAL MORAN'S I
-------------------
PointillViz already has LISA, built in. It is a different statistic and
answers a different question, and the README spells the difference out. In one
line: LISA is a local COVARIANCE — z_i times the mean of its neighbours' z —
so it is large when a cell agrees with its neighbours, and it splits into four
quadrants including two kinds of outlier. Gi* is a local MAGNITUDE — the
neighbourhood total against the tissue mean — so it is large when a place is
bright and has no outlier quadrants at all. A gene can be strongly
autocorrelated with no hotspot (a smooth gradient) and can have a violent
hotspot with modest autocorrelation (one small bright island in noise).

SPEED
-----
One gather over the (n, k) neighbour matrix and then arithmetic; no Python loop
over cells. Measured on datasets/co_m3_r2 (135,232 cells, 5,106 genes, k = 15,
OMP_NUM_THREADS=8): 30-56 ms for `_gi_star` itself, and 0.07-0.31 s end to end
including the gene read, the two-sided p-values, BH over 135k tests and the
73-row cell-type table. Loading the k-NN cache adds ~1 s to the first call.

The same formula written as a per-cell Python loop measures 1.1 s on that input
— only ~20x slower, so the loop is not really what would sink this at scale.
Memory is. `x[idx]` on a 10.4M x 15 graph is a 1.2 GB float64 temporary and the
neighbour mask is a second array beside it, on top of the gene, the graph
and the p-values. Hence the blocked gather in _neighbourhood: same arithmetic,
peak bounded by _GATHER_BYTES.

REFERENCES
----------
Getis, A. & Ord, J.K. (1992). The analysis of spatial association by use of
    distance statistics. Geographical Analysis 24(3): 189-206.
Ord, J.K. & Getis, A. (1995). Local spatial autocorrelation statistics:
    distributional issues and an application. Geographical Analysis 27(4):
    286-306.  (Gi* and the moments used here.)
Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.
    JRSS-B 57(1): 289-300.
"""

from __future__ import annotations

import numpy as np

from pointillviz_tool import PluginError

# ---------------------------------------------------------------------------
# Constants. Every threshold the numbers or the wording depend on lives here.
# ---------------------------------------------------------------------------
DEFAULTS = {"gene": "", "k": 15, "normalize": "log1p", "alpha": 0.05, "fdr": True}

#: Below this many cells in the graph the asymptotic null is not worth
#: reporting at all — the finite-population correction (n - m)/(n - 1) is doing
#: more work than the CLT is.
MIN_GRAPH_CELLS = 20

#: A gene seen in fewer cells than this cannot support a normal approximation
#: anywhere on the slide, so it is refused rather than warned about. Both an
#: absolute floor and a fraction, because either alone is too permissive at
#: some scale: 0.1% of the 600-cell fixture is 0.6 cells, and 10 cells out of
#: a 135,000-cell section is 0.007%.
MIN_EXPRESSING_CELLS = 10
MIN_EXPRESSING_FRACTION = 0.001

#: Warn (do not refuse) when the typical neighbourhood holds fewer expressing
#: cells than this. Below ~3 the neighbourhood sum is a count of a couple of
#: events: its distribution is discrete and right-skewed, and a normal-tail
#: p-value taken from it is optimistic. Actionable in two ways — raise k, or
#: pick a denser gene — so the summary says both.
SPARSE_NEIGHBOURHOOD_HITS = 3.0

#: Peak size of the neighbour-value temporary, in bytes. 64 MB splits a
#: 10.4M x 15 gather into 19 passes of 559k rows instead of one 1.2 GB
#: allocation, at no measurable cost (the arithmetic is memory-bound either
#: way; measured identical to the unblocked version at 135k cells).
_GATHER_BYTES = 64 << 20


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------
def _neighbourhood(x: np.ndarray, idx: np.ndarray,
                   eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SUM_j w_ij x_j and m_i = SUM_j w_ij, with cell i inside its own set.

    Blocked over rows rather than one whole-array gather: `x[idx]` on a
    10.4M x 15 graph is a 1.2 GB float64 temporary, and a second one for the
    mask. Blocking keeps the peak at _GATHER_BYTES and stays fully vectorised —
    the loop runs a handful of times, never once per cell.
    """
    n_all, k = idx.shape
    total = np.zeros(n_all, dtype=np.float64)
    size = np.zeros(n_all, dtype=np.int64)
    rows = max(1, _GATHER_BYTES // (8 * max(k, 1)))
    for lo in range(0, n_all, rows):
        hi = min(lo + rows, n_all)
        nb = idx[lo:hi]
        ok = nb >= 0
        # -1 is not an index: clamp first, then mask. Indexing with -1 would
        # silently pull the LAST cell's expression into every unfilled slot.
        safe = np.where(ok, nb, 0)
        # A listed neighbour can still be outside the population when the user
        # has lassoed a subset; it leaves the neighbourhood and m_i with it.
        ok &= eligible[safe]
        vals = x[safe]
        vals[~ok] = 0.0
        total[lo:hi] = vals.sum(axis=1)
        size[lo:hi] = ok.sum(axis=1)
    # The star. Written separately because it is the definition, not an
    # optimisation: cell i is in its own neighbourhood exactly when it is in
    # the population at all.
    total += np.where(eligible, x, 0.0)
    size += eligible
    return total, size


def _gi_star(x: np.ndarray, idx: np.ndarray, eligible: np.ndarray,
             gene: str) -> tuple[np.ndarray, dict]:
    """Gi* z-score per cell (NaN off the graph) plus the population it used."""
    n = int(eligible.sum())
    if n < MIN_GRAPH_CELLS:
        raise PluginError(
            "too_few_cells",
            f"Only {n} cells have usable coordinates here, and a neighbourhood "
            f"statistic needs at least {MIN_GRAPH_CELLS}. Clear the section "
            "filter or widen the selection.")

    xe = x[eligible]
    xbar = float(xe.mean())
    # S is the POPULATION standard deviation (divided by n, not n - 1) — that
    # is the Getis-Ord definition and it is what makes the ratio a z.
    #
    # Computed from deviations, not from the textbook's sum(x^2)/n - Xbar^2.
    # Those two are the same number in algebra and not in float: the shortcut
    # subtracts two nearly equal large quantities, so a gene with the SAME
    # value in every cell comes out at ~1e-12 instead of exactly 0 and sails
    # through the guard below with a meaningless map. PointillViz's built-in
    # Moran's I declines the identical shortcut for z'z for the identical reason.
    dev = xe - xbar
    s = float(np.sqrt(float((dev * dev).sum()) / n))

    n_expr = int((xe > 0).sum())
    if n_expr == 0:
        raise PluginError(
            "silent_gene",
            f"{gene} has no expression in any of the {n:,} cells with "
            "coordinates here, so the tissue-wide standard deviation is zero "
            "and every Gi* would be 0/0. There is no hotspot map to draw for a "
            "gene that was never detected.")
    if s <= 0.0:
        raise PluginError(
            "constant_gene",
            f"{gene} has the same value in every cell here, so no "
            "neighbourhood can differ from the tissue mean. Nothing to map.")
    floor = max(MIN_EXPRESSING_CELLS, int(round(MIN_EXPRESSING_FRACTION * n)))
    if n_expr < floor:
        raise PluginError(
            "too_sparse",
            f"{gene} is detected in only {n_expr} of {n:,} cells "
            f"({100.0 * n_expr / n:.3f}%). Gi* reads its significance off a "
            "normal curve that a sum of that few non-zeros does not follow, so "
            f"any hotspot it drew would be an artefact. At least {floor} "
            "expressing cells are needed.")

    total, size = _neighbourhood(x, idx, eligible)
    m = size.astype(np.float64)

    # Binary weights: SUM_j w_ij = SUM_j w_ij^2 = m_i, so the bracketed
    # variance term is m(n - m)/(n - 1) exactly. It vanishes when a
    # neighbourhood is the entire population (m == n), which is a real case on
    # a tiny section and is left as NaN rather than divided by.
    num = total - xbar * m
    den = s * np.sqrt(np.maximum(m * (n - m) / (n - 1.0), 0.0))

    z = np.full(x.shape[0], np.nan, dtype=np.float64)
    testable = eligible & (den > 0)
    if not testable.any():
        # m == n for every cell: k is at least as large as the section, so each
        # "neighbourhood" is the entire population and the variance term
        # m(n - m)/(n - 1) is exactly zero. Left alone this returns a map of
        # NaN under a summary reading "0 cells sit in a significant hotspot
        # across 0 tests", which is a confident sentence about nothing —
        # precisely what the silent-gene branch above exists to prevent, so it
        # is refused the same way rather than painted.
        raise PluginError(
            "k_too_large",
            f"With k = {idx.shape[1]}, every neighbourhood covers all {n:,} "
            "cells that have coordinates here, so there is no surrounding "
            "tissue left to compare a place against and the null variance is "
            "exactly zero. Lower k, or clear the section filter so there are "
            "more cells to compare against.")
    z[testable] = num[testable] / den[testable]
    # How hot and how cold a neighbourhood COULD possibly be here. The numerator
    # is bounded by m*(x_min - Xbar) below and m*(x_max - Xbar) above, so these
    # are the extreme Gi* the data admit at the largest neighbourhood on the
    # slide. They matter because expression is bounded below by zero: on a
    # skewed gene the coldest conceivable patch — every one of its k+1 cells at
    # zero — often scores only -2.4, and no coldspot can then exist at ANY
    # threshold. Reporting that is the difference between "no depleted region
    # was found" and "no depleted region was reachable"; both are true and only
    # the second is informative. Measured on datasets/co_m3_r2: Muc2 floors at
    # -2.41 and Lgr5 at -0.29, against a threshold of 3.26.
    # Over the TESTABLE cells, not all eligible ones: a cell whose
    # neighbourhood is the whole population has no z, and letting its m into
    # this bound divides by (n - m) = 0 and reports an attainable Gi* of 1e12.
    m_max = float(m[testable].max())
    span = np.sqrt(m_max * (n - 1.0) / (n - m_max)) / s
    return z, {"n": n, "xbar": xbar, "s": s, "n_expr": n_expr,
               "testable": testable, "m_typical": float(np.median(m[eligible])),
               "z_floor": float((float(xe.min()) - xbar) * span),
               "z_ceiling": float((float(xe.max()) - xbar) * span)}


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR, as the differential-expression tab computes it.

    Standard step-up procedure: sort ascending, scale each p by n/rank, then
    enforce monotonicity from the largest p downwards so a q-value is never
    smaller than one belonging to a larger p. Clipped to 1.
    """
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def _population(ds, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(neighbour indices, eligibility mask) for this run.

    Eligible means "the graph placed this cell": ds.knn hands back -1 for a
    cell with no coordinates, for a cell outside an isolated section, and for a
    cell alone in its own section. Reading eligibility off the graph rather
    than re-deriving it from coordinates is deliberate — it cannot drift out of
    step with whatever ds.knn decides, and a mismatch between "who has a
    neighbourhood" and "who is in the mean" is this statistic's classic silent
    bug.
    """
    idx, _dist = ds.knn(k)
    eligible = (idx >= 0).any(axis=1)
    if ds.subset is not None:
        keep = np.zeros(ds.n_cells, dtype=bool)
        keep[ds.subset] = True
        eligible &= keep
    return idx, eligible


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _cell_type_table(ds, z: np.ndarray, hot: np.ndarray,
                     cold: np.ndarray, eligible: np.ndarray):
    """Per cell type: who is in the hotspots. None when there are no labels."""
    try:
        labels, codes = ds.cell_types()
    except PluginError:
        return None                      # not every dataset carries annotation
    rows = []
    # Unlabelled cells are a row of their own rather than dropped, so the
    # column totals still add up to the counts the summary quotes. A table that
    # does not reconcile with the paragraph above it invites the reader to
    # believe whichever number suits them.
    groups = [(lab, codes == i) for i, lab in enumerate(labels)]
    unl = eligible & (codes < 0)
    if unl.any():
        groups.append(("Unlabelled", codes < 0))
    for label, is_type in groups:
        mask = eligible & is_type
        n_t = int(mask.sum())
        if n_t == 0:
            continue                     # a level with no cells here says nothing
        n_hot = int((mask & hot).sum())
        n_cold = int((mask & cold).sum())
        zt_ = z[mask]
        zt_ = zt_[np.isfinite(zt_)]
        mean_z = float(zt_.mean()) if zt_.size else float("nan")
        rows.append([label, n_hot, n_cold, round(n_hot / n_t, 4),
                     round(mean_z, 3), n_t])
    # By hotspot COUNT, not by hotspot fraction, and the difference is not
    # cosmetic: sorting by fraction puts "Ribo hi B cells, 1 of 4 cells, 25%"
    # above "Immature enterocytes II, 547 cells" on Lgr5 in datasets/co_m3_r2,
    # which answers "which type is most enriched" when the question the reader
    # asked was "what ARE the hotspots". The fraction stays as a column so the
    # enrichment reading is one click away. Count, then fraction, then the name:
    # a total order, so two runs of the same input give byte-identical rows.
    rows.sort(key=lambda r: (-r[1], -r[3], r[0]))
    return {"title": "Who the hotspots are",
            "columns": ["Cell type", "In hotspot", "In coldspot",
                        "Hotspot fraction", "Mean Gi*", "Cells"],
            "rows": rows, "sort_by": "In hotspot"}


def _summary(gene: str, params: dict, stats: dict, counts: dict,
             table, ds) -> str:
    n = stats["n"]
    n_hot, n_cold, n_test = counts["hot"], counts["cold"], counts["tested"]
    alpha = params["alpha"]
    thresh = (f"BH-corrected q < {alpha:g}" if params["fdr"]
              else f"uncorrected p < {alpha:g}")
    # Blocks joined by a BLANK line, never by one newline: the viewer splits
    # paragraphs on /\n{2,}/, and a single newline runs
    # them together into one paragraph whose emphasis markers then pair across
    # sentences that are not one span.
    #
    # And `Gi*` inside backticks, never Gi\*: that markdown subset has no escape
    # handling, so the backslash reaches the screen and the bare asterisk still
    # pairs with the next one on the line.
    blocks = [
        f"**{gene} — {n_hot:,} cells ({100.0 * n_hot / n:.1f}% of the {n:,} mapped "
        f"cells) sit in a significant hotspot** and {n_cold:,} "
        f"({100.0 * n_cold / n:.1f}%) "
        f"in a coldspot, at {thresh} across {n_test:,} tests "
        f"({params['k']}-nearest neighbours, {params['normalize']} values).",
    ]

    facts = []
    if counts["z_cut"] is not None:
        facts.append(f"Threshold |`Gi*`| = {counts['z_cut']:.2f}; the map runs "
                     f"{counts['z_min']:.1f} to {counts['z_max']:.1f}.")
    top = [r for r in (table["rows"] if table else []) if r[1] > 0][:3]
    if n_hot + n_cold == 0:
        facts.append(
            f"**Nothing reaches significance** at this threshold: no "
            f"neighbourhood of {params['k']}+1 cells departs from the tissue "
            f"mean by more than chance. A smaller k reads a finer pattern.")
    elif top:
        facts.append("The hotspots are mostly " + ", ".join(
            f"{r[0]} ({r[1]:,} cell{'' if r[1] == 1 else 's'}, "
            f"{100.0 * r[3]:.0f}% of that type)" for r in top) + ".")
        share = top[0][1] / max(n_hot, 1)
        if share >= 0.4:
            facts.append(
                f"**{100.0 * share:.0f}% of all hot cells are {top[0][0]}** — "
                f"largely a map of where that population lives.")
    if facts:
        blocks.append(" ".join(facts))

    # Who the hot cells ARE and which population is ENRICHED are two different
    # questions, and on real tissue they usually have two different answers:
    # Muc2's hotspots on datasets/co_m3_r2 are led by Immature enterocytes I
    # simply because that is the biggest population living where the goblet
    # cells are. The size floor keeps "1 of 4 cells, 25%" off this line.
    if top and n_hot:
        floor = max(20, int(round(0.001 * n)))
        sizable = [r for r in table["rows"] if r[5] >= floor and r[1] > 0]
        if sizable:
            best = max(sizable, key=lambda r: (r[3], r[1], r[0]))
            if best[0] != top[0][0]:
                blocks.append(
                    f"Most *enriched* population: **{best[0]}**, with "
                    f"{100.0 * best[3]:.0f}% of its {best[5]:,} cells hot "
                    f"against {100.0 * n_hot / n:.1f}% of the tissue.")

    caveats = []
    # BH's cutoff is never looser than the uncorrected one, so the nominal |z|
    # is a safe stand-in when nothing was significant.
    cut = counts["z_cut"] if counts["z_cut"] is not None else counts["z_nominal"]
    if n_cold == 0 and abs(stats["z_floor"]) < cut:
        caveats.append(
            f"**No coldspot was reachable.** The coldest neighbourhood this "
            f"tissue admits — all {int(stats['m_typical'])} cells at the "
            f"minimum — scores `Gi*` = {stats['z_floor']:.2f} against a "
            f"{cut:.2f} threshold, so the absence of blue is the scale, not "
            f"the tissue.")
    if n_hot == 0 and stats["z_ceiling"] < cut:
        caveats.append(
            f"**No hotspot was reachable either.** {int(stats['m_typical'])} "
            f"cells all at this gene's maximum would score `Gi*` = "
            f"{stats['z_ceiling']:.2f}, short of the {cut:.2f} threshold.")
    if counts["sparse_hits"] < SPARSE_NEIGHBOURHOOD_HITS:
        caveats.append(
            f"**Sparse gene — the p-values are optimistic.** {gene} is detected "
            f"in {100.0 * stats['n_expr'] / n:.1f}% of these cells, so a typical "
            f"{int(stats['m_typical'])}-cell neighbourhood holds "
            f"{counts['sparse_hits']:.1f} expressing cells. Raise k, or read "
            f"the map rather than the test.")
    if ds.subset is not None:
        caveats.append(
            f"Restricted to your selection of {int(ds.subset.size):,} cells: hot "
            f"means hot **relative to the selection**, not to the slide.")
    elif ds.section_filter is None and len(ds.sections) > 1:
        caveats.append(
            f"The reference mean is pooled over all {len(ds.sections)} sections "
            f"while neighbourhoods stay inside one, so a globally brighter "
            f"section reads hot throughout.")
    if not params["fdr"]:
        caveats.append(
            f"**Uncorrected.** {n_test:,} tests at p < {alpha:g}; about "
            f"{int(round(alpha * n_test)):,} clear it on noise alone.")
    if caveats:
        blocks.append("\n".join(f"- {c}" for c in caveats))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(ds, params):
    p = {**DEFAULTS, **{k: v for k, v in (params or {}).items() if v is not None}}
    gene = str(p["gene"]).strip()
    if not gene:
        raise PluginError("no_gene", "Choose a gene to map hotspots for.")
    k = int(p["k"])
    alpha = float(p["alpha"])
    use_fdr = bool(p["fdr"])
    normalize = str(p["normalize"])

    # ds.gene raises unknown_gene with the panel size in the message, which is
    # a better sentence than anything this module could write.
    x = ds.gene(gene, normalize=normalize).astype(np.float64)
    ds.progress(0.2, "building the neighbour graph")
    idx, eligible = _population(ds, k)

    ds.progress(0.5, f"Gi* over {int(eligible.sum()):,} cells")
    z, stats = _gi_star(x, idx, eligible, gene)
    testable = stats["testable"]

    # Two-sided, because a coldspot is as much a finding as a hotspot and the
    # tool paints both. One-sided p would halve the coldspot p-values.
    from scipy.stats import norm
    pvals = 2.0 * norm.sf(np.abs(z[testable]))
    # n cells is n tests. Without a correction a 135k-cell section paints ~6,700
    # cells significant at 0.05 with no structure whatever — the same arithmetic
    # PointillViz's built-in LISA cites. _benjamini_hochberg is a copy of the
    # differential-expression tab's Benjamini-Hochberg, not a rewrite, so a q
    # here and a q in that tab mean the same thing.
    qvals = _benjamini_hochberg(pvals) if use_fdr else pvals
    sig = qvals < alpha

    hot = np.zeros(ds.n_cells, dtype=bool)
    cold = np.zeros(ds.n_cells, dtype=bool)
    rows = np.flatnonzero(testable)
    hot[rows[sig & (z[testable] > 0)]] = True
    cold[rows[sig & (z[testable] < 0)]] = True

    ds.progress(0.85, "summarising by cell type")
    table = _cell_type_table(ds, z, hot, cold, eligible)
    zt = z[testable]
    counts = {
        "hot": int(hot.sum()), "cold": int(cold.sum()), "tested": int(sig.size),
        # The effective cutoff, reported rather than assumed: BH's threshold is
        # data-dependent, so "q < 0.05" corresponds to a different |z| on every
        # gene and the reader deserves the number that actually applied.
        "z_cut": float(np.abs(zt[sig]).min()) if sig.any() else None,
        "z_min": float(zt.min()) if zt.size else float("nan"),
        "z_max": float(zt.max()) if zt.size else float("nan"),
        "sparse_hits": stats["m_typical"] * stats["n_expr"] / stats["n"],
        "z_nominal": float(norm.isf(alpha / 2.0)),
    }
    ds.log(f"{gene}: n={stats['n']:,} in the graph, mean={stats['xbar']:.4g}, "
           f"SD={stats['s']:.4g}, expressed in {stats['n_expr']:,} "
           f"({100.0 * stats['n_expr'] / stats['n']:.1f}%)")
    ds.log(f"{counts['hot']:,} hotspot cells, {counts['cold']:,} coldspot cells "
           f"at {'q' if use_fdr else 'p'} < {alpha:g}")

    # A symmetric domain, explicitly. Left to the viewer's default p1..p99 the
    # ends would be unequal and the ramp's neutral colour would land somewhere
    # other than Gi* = 0 — which paints a mildly positive neighbourhood blue.
    # Floored at the significance cutoff so the cells the tool calls hot are
    # always inside the coloured range rather than clipped at the end of it.
    lim = float(np.nanpercentile(np.abs(zt), 99)) if zt.size else 1.0
    if counts["z_cut"] is not None:
        lim = max(lim, counts["z_cut"] * 1.05)
    lim = max(lim, 1.0)

    return {
        "summary": _summary(gene, {"k": k, "alpha": alpha, "fdr": use_fdr,
                                   "normalize": normalize},
                            stats, counts, table, ds),
        "cells": {
            "values": z,
            "label": f"{gene} Gi* (hotspot z)",
            # icefire is PointillViz's own diverging colour map;
            # `diverging` is the load-bearing flag — it is what centres the
            # ramp on zero, and zero is the only meaningful centre for a z.
            "colormap": "icefire",
            "diverging": True,
            "domain": [-lim, lim],
            "units": "z",
        },
        "table": table,
    }
