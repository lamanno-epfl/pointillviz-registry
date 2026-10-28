"""Conditional contact odds: the change in log-odds of being level A per extra
neighbour of level B, holding every other level's neighbour count fixed.

WHAT IS FITTED
--------------
Build the undirected union of the kNN graph, and let N (n x K) hold each cell's
neighbour counts per level. Fit the K x K symmetric matrix J and the per-level
offsets h that maximise the Besag pseudo-likelihood of the observed labels
given that graph,

    P(z_i = a | the labels around i) = softmax_a( h_a + sum_b J_ab N_ib ),

which is one multinomial logistic regression on a design matrix that never
changes. So J_ab is a REGRESSION COEFFICIENT: swap one of cell i's neighbours
for a level-b cell and the log-odds that i is level a move by J_ab, with every
other level's count held where it was. `celltype-cooccurrence` and
`ripley-cross-l` both answer the marginal version of that question -- how often
do A and B end up together -- and the two answers differ whenever a third level
mediates the contact. Separating those is the whole product; README section 4
demonstrates it on a generator where the marginal statistic reports a 4x
enrichment for a pair the generator never linked.

WHAT IS NOT CLAIMED, AND WHY THE README SAYS SO FIRST
-----------------------------------------------------
Pseudo-likelihood is consistent for the CONDITIONALS. The joint distribution
they imply is a different object, and on real tissue it is wrong: start
chromatic Gibbs sampling AT the observed colon labelling under the fitted
(J, h) and the field leaves the data inside 25 sweeps -- neighbour agreement
0.3214 (the data) -> 0.4823 by sweep 10 -> 0.8022 by 25 -> 0.9197 by 50, and
the largest level's share 0.1114 -> 0.2531. The chain starts at the data, so no
equilibration argument applies. A translation-invariant pairwise model has one
offset per level and cannot represent crypt, muscularis and submucosa, so the
section's composition gradient is loaded onto J. The regression reading survives
that; the generative reading does not, so nothing here is reported as a physical
quantity and the sign of J is a direction of association, not a force.

DESIGN, WITH WHAT EACH PART COST
--------------------------------
Everything below is measured on `datasets/co_m3_r2` (135,232 cells, 73 levels)
through PointillViz's tool runner, the path the browser uses. The
estimator lives in `potts_core.py`, whose docstring carries the per-optimisation
before/after; the short form is float32 for the two (n x K)(K x K) GEMMs and the
softmax with float64 for the two reductions that can lose digits, row blocking
at 16,384 because the elementwise passes cost more than the GEMMs, symmetry
imposed in the parametrisation rather than by penalty, exact column centring of
N absorbed into h, and L-BFGS-B stopped on max|gradient| per cell rather than on
step size.

  ALL LEVELS ARE FITTED, then masked for display. Dropping the levels under 100
  cells before fitting takes K from 73 to 59 and the fit from 26.1 to 18.7 s,
  and moves well-observed entries by up to 0.046 -- two and a half times the
  whole remaining convergence error of the full fit, bought for 7.4 s. The gate
  is applied to the OUTPUT instead: an entry is shown when both levels clear
  `min_cells` and the pair has at least 100 observed contacts. That is where the
  estimate is worth anything -- against a float64 reference solved to
  max|g|/n = 3e-3 on a GPU, this fit's worst entry is 0.018 out over the 1,698
  entries with at least 100 contacts and 0.87 out over the 3,631 without.

  THE PER-CELL CHANNEL IS log P(z_i | its neighbours), not the model's
  predictive entropy. Both are one line of the same softmax; only one of them is
  new. Against the shipped `neighborhood-composition` mixing channel on this
  section, the entropy reads Spearman +0.921 -- a restatement -- and the fitted
  log-probability reads -0.577.

  NO SERIES. Nothing computed here is a function of a scale parameter, and
  padding one in by refitting at a ladder of k would multiply the cost by the
  length of the ladder to draw a curve with no reading.

  NO RADIUS OPTION. Marginal enrichment on the kNN-15 union and on a radius-60
  graph over the same section rank-correlate at +0.980 over the pairs whose two
  levels both have at least 500 cells (+0.970 over all 2,628 off-diagonal
  pairs), so a second neighbourhood knob buys nothing here, and
  `celltype-cooccurrence` already owns the radius axis.

DETERMINISM
-----------
There is no stochastic step anywhere in this tool -- no sampling, no shuffled
null, no random initialisation (L-BFGS-B starts at the origin) -- so there is no
seed to derive, and two runs at the same parameters on the same machine return
byte-identical results.

The iteration cap is a count (MAX_ITER), never a clock, so where it binds it
binds at the same iteration on every machine. A fit that reaches it says so in
bold in the summary.

One caveat, stated rather than hidden: the float32 arithmetic is thread-count
sensitive in the last digits. The same fit takes 374 iterations at 8 BLAS
threads and 348 at 4, so two machines with different thread counts can
disagree there.
"""

import numpy as np

from pointillviz_tool import PluginError

#: A pair needs this many observed contacts before its entry is shown. Below it
#: the entry is set by the ridge rather than by the data: measured against a
#: float64 reference, worst-entry error is 0.018 above this floor and 0.62
#: below it.
MIN_CONTACTS = 100

#: `check_result` caps a matrix at 10,000 cells, so 100 levels is the largest
#: square matrix that can be returned at all, whatever `max_levels` says. The
#: fit itself has been run at K = 290; the viewer has not.
MATRIX_LEVEL_CEILING = 100

#: One objective evaluation touches every cell K times, so the row block is
#: sized for L3 rather than for the matrix. See potts_core.
ROW_BLOCK = 16384

#: The optimiser's iteration cap. The iteration count needed is not a function
#: of anything the tool can see in advance -- at k=15 and the shipped ridge the
#: colon section takes 374 iterations, at ridge 0.01 it takes 776, and asking
#: for a tolerance float32 cannot reach (1e-6) makes it run until the line
#: search fails. One iteration costs about one objective evaluation: 0.070 s at
#: k=15 and at k=30 on that section (the graph is built once, so k moves the
#: ITERATION COUNT and not the cost of one), and 0.091 s at 100 levels. A fit
#: that reaches the cap returns what it has, with the summary saying it did not
#: converge.
MAX_ITER = 3000


def _contact_counts(edges, codes, n_levels):
    """(contacts, n_edges_labelled): undirected contacts per unordered pair.

    `contacts[a, b]` is the number of graph edges whose two endpoint labels are
    {a, b}, counted once each. The diagonal is the number of homotypic edges.
    """
    ca, cb = codes[edges[:, 0]], codes[edges[:, 1]]
    keep = (ca >= 0) & (cb >= 0)
    ca, cb = ca[keep].astype(np.int64), cb[keep].astype(np.int64)
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    upper = np.bincount(lo * n_levels + hi,
                        minlength=n_levels * n_levels).reshape(n_levels, n_levels)
    upper = upper.astype(np.float64)
    contacts = upper + upper.T - np.diag(np.diag(upper))
    return contacts, int(ca.size)


def _marginal_enrichment(contacts, n_edges, codes, n_levels):
    """log2(observed / expected) on the SAME graph, for the reference column.

    This is `celltype-cooccurrence`'s statistic recomputed here rather than
    quoted from it, so the table's two columns differ by the ESTIMATOR and not
    by the neighbourhood definition. Same convention as that tool: `observed`
    counts ordered endpoint pairs (so the diagonal is doubled), `expected` is
    the multinomial mean under a relabelling of the same cells, and both get a
    +1 so an unseen pair is strongly negative rather than -inf.
    """
    observed = contacts + np.diag(np.diag(contacts))
    counts = np.bincount(codes[codes >= 0].astype(np.int64),
                         minlength=n_levels).astype(np.float64)
    share = counts / counts.sum() if counts.sum() else np.zeros(n_levels)
    expected = 2.0 * n_edges * np.outer(share, share)
    return np.log2((observed + 1.0) / (expected + 1.0)), counts


def _log_conditional(N, J, h, codes):
    """log P(z_i = its observed label | the labels around i), NaN if unlabelled.

    Blocked over rows for the same reason the objective is: the softmax passes
    dominate, and one block of logits fits in cache.
    """
    n, n_levels = N.shape
    out = np.full(n, np.nan, dtype=np.float64)
    Jf = np.ascontiguousarray(J, dtype=np.float32)
    hf = np.asarray(h, dtype=np.float32)
    for start in range(0, n, ROW_BLOCK):
        stop = min(start + ROW_BLOCK, n)
        logits = (N[start:stop] @ Jf) + hf
        top = logits.max(axis=1, keepdims=True)
        logits -= top
        chosen = codes[start:stop]
        ok = chosen >= 0
        lse = np.log(np.exp(logits, dtype=np.float64).sum(axis=1))
        picked = logits[np.arange(stop - start), np.maximum(chosen, 0)]
        out[start:stop] = np.where(ok, picked.astype(np.float64) - lse, np.nan)
    return out


def _ranks(values):
    """Dense competition-free ranks, 1 = largest. Ties broken by position, which
    is stable because both rankings are taken over the same ordered pair list."""
    order = np.argsort(-values, kind="stable")
    out = np.empty(values.size, dtype=np.int64)
    out[order] = np.arange(1, values.size + 1)
    return out


def run(ds, params):
    import time

    import potts_core as pc               # sibling module: not importable at
                                          # manifest-probe time, only at run time

    field = params.get("field") or None
    labels, codes = ds.categorical(field)
    n_levels = len(labels)
    codes = np.asarray(codes).copy()

    # A lasso is the tissue the fit describes. Setting the cells outside it
    # unlabelled rebuilds the graph inside the selection: no contact with an
    # excluded cell enters N, none of them contributes to a neighbour's counts,
    # and each is NaN on the map because `_log_conditional` already returns NaN
    # for an unlabelled cell. Masking a J fitted on everything would be the
    # other choice and it is the wrong one -- the coefficients would then come
    # from contacts with tissue the user excluded. Same rule, and the same four
    # lines, as the lasso handling in interface-coherence/tool.py.
    if ds.subset is not None and ds.subset.size:
        chosen = np.zeros(ds.n_cells, dtype=bool)
        chosen[ds.subset] = True
        codes[~chosen] = -1

    if n_levels < 2:
        raise PluginError(
            "one_level",
            "This annotation has only one level. A contact matrix needs at "
            "least two things to be in contact.")
    ceiling = min(int(params["max_levels"]), MATRIX_LEVEL_CEILING)
    if n_levels > ceiling:
        why = (f"the level ceiling is set to {ceiling}"
               if ceiling < MATRIX_LEVEL_CEILING else
               f"{MATRIX_LEVEL_CEILING} levels is the hard ceiling — a "
               f"{n_levels}x{n_levels} matrix has {n_levels * n_levels:,} "
               f"entries and the viewer caps a matrix at 10,000")
        raise PluginError(
            "too_many_levels",
            f"This annotation has {n_levels} levels and {why}. Pick a coarser "
            f"grouping. (Cost climbs steeply with the level count as well: on a "
            f"135k-cell section 73 levels took 26 s at k=15 and 146 levels took "
            f"108 s at k=30.)")

    n_labelled = int((codes >= 0).sum())
    if n_labelled < 2 * n_levels:
        raise PluginError(
            "too_few_labelled",
            f"Only {n_labelled:,} cells carry a level of "
            f"{field or 'this annotation'}, against {n_levels} levels. There "
            f"is not enough labelled tissue to fit one coefficient per pair.")

    k = int(params["k"])
    t0 = time.time()
    idx, _ = ds.knn(k)
    t_knn = time.time() - t0

    t0 = time.time()
    edges = pc.undirected_edges(idx)
    if edges.shape[0] == 0:
        raise PluginError(
            "no_edges",
            "The neighbour graph came back empty — no cell in this dataset has "
            "a neighbour with coordinates. There are no contacts to model.")
    N = pc.neighbour_counts(edges, codes, n_levels, ds.n_cells)
    contacts, n_edges = _contact_counts(edges, codes, n_levels)
    marginal, level_counts = _marginal_enrichment(contacts, n_edges, codes, n_levels)
    t_graph = time.time() - t0
    ds.log(f"graph: k={k}, {edges.shape[0]:,} undirected edges "
           f"({n_edges:,} between labelled cells) in {t_knn + t_graph:.2f}s")

    # The display gate is checked BEFORE the fit, so a gate nothing can clear
    # costs a sentence rather than half a minute.
    min_cells = int(params["min_cells"])
    level_ok = level_counts >= min_cells
    pair_ok = (level_ok[:, None] & level_ok[None, :]) & (contacts >= MIN_CONTACTS)
    if not pair_ok.any():
        raise PluginError(
            "nothing_reportable",
            f"No pair of levels clears the display gate: {min_cells:,} cells in "
            f"each level and {MIN_CONTACTS} observed contacts between them. The "
            f"largest level here has {int(level_counts.max()):,} cells and the "
            f"busiest pair has {int(contacts.max()):,} contacts. Lower the "
            f"minimum, or group the annotation more coarsely.")

    t0 = time.time()
    J, h, info = pc.fit(N, codes, n_levels, lam=float(params["lam"]),
                        gtol_rel=float(params["gtol"]), chunk=ROW_BLOCK,
                        maxiter=MAX_ITER, precondition=False, center=True,
                        progress=ds.progress)
    t_fit = time.time() - t0
    hit_cap = not info["gtol_met"] and info["nit"] >= MAX_ITER
    ds.log(f"fit: {t_fit:.2f}s, {info['nit']} iterations, {info['nfev']} "
           f"evaluations, max|g|/n = {info['gmax_per_cell']:.2e} "
           f"({'converged' if info['gtol_met'] else 'did not converge'})")

    # Gauge. Adding a constant to every entry of J shifts each cell's logits by
    # that constant times its degree, which the softmax removes, so only
    # DIFFERENCES within J are identified. Subtracting the mean picks the
    # representative in which 0 reads "no conditional preference relative to the
    # average pair"; the ridge already chose a nearby one, and this makes the
    # choice explicit rather than a side effect of the penalty.
    Jc = np.asarray(J, dtype=np.float64)
    Jc = Jc - Jc.mean()

    t0 = time.time()
    logp = _log_conditional(N, J, h, codes)
    t_cells = time.time() - t0

    shown = np.where(pair_ok, Jc, np.nan)

    # --- the table: conditional beside marginal, on the same graph ----------
    iu, ju = np.triu_indices(n_levels)
    sel = pair_ok[iu, ju]
    ia, ib = iu[sel], ju[sel]
    j_vals = Jc[ia, ib]
    e_vals = marginal[ia, ib]

    # Rank shift is computed WITHIN kind — homotypic entries against homotypic
    # entries, cross-level pairs against cross-level pairs. A level's marginal
    # enrichment with itself is dominated by its own clustering and sits far
    # above every cross-level pair, so ranking the two kinds together makes the
    # column a measure of the diagonal: on the colon test section the twelve
    # largest shifts in a pooled ranking are all diagonal entries, and the
    # largest of them (Immature enterocytes III with itself, +3.83 marginal
    # against -0.125 conditional) buries every cross-level disagreement in the
    # table.
    homotypic = ia == ib
    shift = np.zeros(ia.size, dtype=np.int64)
    for group in (~homotypic, homotypic):
        if group.any():
            shift[group] = _ranks(e_vals[group]) - _ranks(j_vals[group])
    order = np.argsort(-np.abs(j_vals), kind="stable")
    rows = [[str(labels[ia[o]]), str(labels[ib[o]]),
             round(float(j_vals[o]), 4), round(float(e_vals[o]), 3),
             int(contacts[ia[o], ib[o]]), int(shift[o])]
            for o in order[:200]]

    n_pairs_total = n_levels * (n_levels + 1) // 2
    n_pairs_shown = int(sel.sum())
    n_levels_masked = int(n_levels - level_ok.sum())
    cross = np.nonzero(~homotypic)[0]
    biggest = int(cross[np.argmax(np.abs(shift[cross]))]) if cross.size else None

    if biggest is not None:
        a, b = labels[ia[biggest]], labels[ib[biggest]]
        moved = ("up" if shift[biggest] > 0 else "down")
        disagreement = (
            f"The pair the two columns disagree about most is "
            f"**{a} ↔ {b}**, which conditioning moves {moved} "
            f"{abs(int(shift[biggest])):,} places out of {cross.size:,} "
            f"cross-level pairs: {e_vals[biggest]:+.2f} marginal log2 obs/exp "
            f"against {j_vals[biggest]:+.3f} conditional log-odds over "
            f"{int(contacts[ia[biggest], ib[biggest]]):,} contacts.")
    else:
        disagreement = ""

    n_flip = int(np.sum(np.sign(j_vals) != np.sign(e_vals)))
    if info["gtol_met"]:
        how = "converged"
    elif hit_cap:
        how = (f"**stopped after {MAX_ITER:,} iterations without "
               f"converging** — lower k, or loosen the tolerance, or raise "
               f"the ridge")
    else:
        how = ("stopped where float32 arithmetic can no longer improve the "
               "objective, short of the tolerance asked for")
    if ds.subset is not None and ds.subset.size:
        selection = (f" Restricted to your selection of {int(ds.subset.size):,} "
                     f"cells: the graph was rebuilt inside it, so contacts with "
                     f"the tissue around the selection are not in this fit.")
    else:
        selection = ""
    summary = (
        f"**{n_levels} levels fitted jointly**, {info['nit']} iterations to "
        f"max gradient {info['gmax_per_cell']:.1e} per cell ({how}), "
        f"over {edges.shape[0]:,} undirected k={k} contacts.{selection}\n\n"
        f"{n_pairs_shown:,} of {n_pairs_total:,} pairs are shown; the rest are "
        f"blank because a level has fewer than {min_cells:,} cells "
        f"({n_levels_masked} of {n_levels} levels) or the pair has fewer than "
        f"{MIN_CONTACTS} observed contacts. Every level was fitted regardless — "
        f"the gate is on what is worth reading, not on what was estimated. "
        f"{n_flip:,} of the shown pairs ({100.0 * n_flip / max(1, n_pairs_shown):.1f}%) "
        f"take the opposite sign from the marginal enrichment of the same "
        f"contacts.\n\n{disagreement}\n\n"
        f"The matrix is a coefficient, not a count: 0 is no conditional "
        f"preference relative to the average pair, and the fit is only "
        f"consistent for these conditional odds — it is not a generative model "
        f"of the section, and the store page says why in full.")

    ds.log(f"timing: knn {t_knn:.2f}s, graph {t_graph:.2f}s, fit {t_fit:.2f}s, "
           f"per-cell channel {t_cells:.2f}s")

    return {
        "summary": summary,
        "cells": {"values": logp, "label": "Neighbourhood predictability",
                  "colormap": "viridis", "diverging": False,
                  "units": "log probability"},
        "matrix": {"title": f"Conditional contact log-odds (k={k})",
                   "rows": list(labels), "cols": list(labels),
                   "values": shown.tolist(), "diverging": True},
        "table": {"title": "Strongest conditional contacts",
                  "columns": ["A", "B", "conditional log-odds",
                              "marginal log2 obs/exp", "contacts", "rank shift"],
                  "rows": rows},
    }
