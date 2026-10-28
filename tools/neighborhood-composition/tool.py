"""Neighbourhood composition and mixing — one k-NN pass, three readings of it.

For every cell: its k nearest cells IN THE SAME SECTION, and what categories
(cell type by default) those neighbours are. Everything below is a different
projection of that one (n, k) table of neighbour labels.

  cells   a per-cell mixing index — the Shannon entropy of the neighbourhood's
          composition divided by log(T), T = categories present in the data.
          That ratio is Pielou's evenness (Pielou 1966), computed over a
          spatial neighbourhood rather than an ecological quadrat.
  matrix  for every ordered pair of types, log2 of observed over expected
          neighbour counts, expected under independence WITHIN EACH SECTION.
  table   per type: cells scored, mean mixing, self-enrichment, and the other
          type it sits beside more than chance.

The denominator is log(T over the whole dataset), never log(T in this cell's
own neighbourhood)
------------------------------------------------------------------------------
Dividing by the count of types a cell happens to see would give every cell its
own y-axis: a cell with two types split evenly and a cell with fifteen types
split evenly would both score exactly 1.0, and the map would say they are
equally mixed when one of them is fifteen times more heterogeneous. A single
dataset-wide denominator keeps one meaning for one colour, which is the only
thing that makes a painted tissue readable. It costs the top of the scale: with
k neighbours drawn from T types no cell can exceed log(min(k, T))/log(T), so
k=15 on a 73-type section tops out at 0.63. That ceiling is computed, put in
the summary, and used as the colour domain, rather than being hidden by
quietly swapping the denominator for log(min(k, T)).

What counts as a neighbour
--------------------------
An UNLABELLED neighbour (code -1) is not a category, it is a hole. Counting it
as a thirteenth type would make the index rise wherever segmentation or
annotation failed, so the tool would be measuring annotation coverage and
reporting it as tissue biology. Unlabelled neighbours are dropped and the
composition is taken over the labelled ones that remain.

An unlabelled CELL still gets a value. Its own missing label says nothing about
what surrounds it, and the neighbourhood is exactly as measurable as any
other's; refusing to score it would leave holes in the map for a reason the
reader cannot see. Those cells contribute to no per-type row, because there is
no type to attribute them to.

A cell with fewer than MIN_LABELLED_NEIGHBOURS labelled neighbours gets NaN.
With one neighbour the entropy is exactly 0, which paints identically to the
interior of a homogeneous domain while meaning "nothing was measured" — the
distinction pointillviz_tool/dataset.py exists to preserve.

A type with zero members is dropped from T, from the matrix and from the table.
Leaving it in the denominator would make the maximum unreachable for every cell
(log of a count that includes a category nothing can ever be), and leaving it
in the matrix would print a row of numbers for a measurement that never
happened.

Expectation is computed per section, then summed
------------------------------------------------
Neighbours never cross sections — `ds.knn` builds one tree per section, because
two sections can overlap on the shared canvas. The enrichment has to respect
that too. Pooling all sections and comparing against one global composition
makes any two types that happen to be abundant in the same section look
mutually enriched, whatever their arrangement inside it: a section is a
confounder and pooling over it is Simpson's paradox with a colour ramp. So the
expected count is sum over sections of (row total x column total / grand
total), the stratified expectation, and the observed count is the plain sum.
On a one-section dataset this reduces to the ordinary independence expectation.

log2((observed + 1) / (expected + 1)), matching PointillViz's Niches tab,
so the number a reader learns in the Niches tab means the same thing here. The
pseudocount is what makes a rare pair report a small ratio instead of a
spectacular one built on four observations, and it is why a never-observed pair
that COULD have been observed is a finite strong depletion rather than -inf
(which check_result rejects) or NaN (which reads as "not measured" when it was).

A pair no section could ever have shown — type i present only in one section,
type j only in another — is the one case that gets NaN, and the viewer draws it
as absence. Its expected count is exactly zero, so "observed 0 where 0 was
expected" carries no information about arrangement; printing the strong
depletion that a pooled expectation would give it would be reporting a
difference between specimens as a spatial finding.

Raw row fractions were the alternative and are worse: with a type at 40% of the
tissue, every row's largest entry is that type, and the matrix reports the
global composition 73 times over instead of anything spatial.

Cost
----
One blocked pass over the (n, k) neighbour matrix; no Python loop over cells.
Measured on datasets/co_m3_r2 (135,232 cells, 73 types, 1 section, k = 15):
**0.21 s** with the neighbour graph cached on disk, **1.1 s** including a cold
cKDTree build (0.94 s of it). The statistics are 0.17 s of that, and the
neighbour counting inside them is 0.07 s. The same counting written as a
per-cell Python loop is 1.4 s, which extrapolates to ~104 s on the 10.4 M-cell
atlas against ~5 s here; see _sweep for the two other candidates and why they
lost.
"""

from __future__ import annotations

import math

import numpy as np

from pointillviz_tool import PluginError

DEFAULT_K = 15
DEFAULT_MIN_TYPE_CELLS = 25

#: Below this many labelled neighbours a neighbourhood is not a sample. Two is
#: the smallest count that can distinguish "all the same" from "mixed" at all.
MIN_LABELLED_NEIGHBOURS = 2

#: 100 x 100 is exactly MAX_MATRIX_CELLS in pointillviz_tool/results.py. A
#: dataset with more types than this still gets every one of them in the
#: table; the heatmap shows the most abundant, which is why its axes are
#: ordered by abundance — everything displayed outranks everything dropped.
MAX_MATRIX_TYPES = 100
#: Ceiling on the types the K x K accumulator is built over. 2,000 types is
#: 32 MB of int64 and more rows than the 5,000-row table limit leaves room to
#: read; the largest reference we have used (Yao 2023 supertypes) has 1,197.
MAX_ANALYSED_TYPES = 2000

#: Neighbour pairs handled per block. The whole point of blocking is that peak
#: memory stops depending on n: 4 M pairs is ~130 MB of int64 scratch, whether
#: the dataset is the 600-cell fixture or the 10.4 M-cell atlas (which at k=15
#: is 156 M pairs and would otherwise need 2.5 GB inside one np.unique).
PAIR_BLOCK = 4_000_000


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _resolve_field(ds, name):
    """(levels, codes) for the categorical the user picked, or the default one.

    `field` is declared optional in the manifest instead of carrying a default
    string, because every dataset spells its cell-type column differently and a
    hardcoded default would name a column that exists on one of them.
    `ds.categorical(None)` resolves to whatever THIS dataset calls its cell
    types, which is the documented meaning of "no choice made".
    """
    levels, codes = ds.categorical(name or None)
    if len(levels) < 2:
        raise PluginError(
            "one_category",
            f"{name or 'the cell-type column'} has only {len(levels)} level, so "
            "there is no composition to measure. Pick a categorical column with "
            "at least two categories.")
    # int64 throughout: the neighbour codes get folded into flat bincount keys
    # of size row * T + code, which overflows int32 on a dataset of any size.
    return list(levels), np.asarray(codes, dtype=np.int64)


# ---------------------------------------------------------------------------
# The one pass
# ---------------------------------------------------------------------------
def _sweep(ds, idx, codes, mcode, centre, sec_dense, n_types, n_kept, n_sections):
    """Everything the three outputs need, in one block-wise sweep of `idx`.

    Returns (clogc, n_valid, pair, row_marg, col_marg):
      clogc      (n,)  sum over types of c*log(c) for that cell's neighbourhood
      n_valid    (n,)  labelled neighbours found
      pair       (K,K) observed neighbour counts, pooled over sections
      row_marg   (S,K) neighbour observations per (section, centre type)
      col_marg   (S,K) neighbour observations per (section, neighbour type)

    The per-cell composition is NEVER materialised as a dense (n, T) count
    matrix, which is the obvious way to write this and the reason it would not
    run: 135k x 73 is 39 MB at int32, but 10.4 M x 73 is 3.0 GB and 10.4 M x
    1,197 (the brain reference's supertypes) is 50 GB. A neighbourhood holds at
    most k counts however many types exist, so sorting the flat (cell, type)
    keys and taking run lengths — which is all np.unique(return_counts=True)
    does — carries the same information in O(n*k) and never mentions T except
    as a stride.

    np.add.at into that dense matrix was the other candidate and is 5x slower
    even where it fits (measured on 135k x 15: 0.35 s against 0.07 s), because
    it takes the unbuffered-ufunc path instead of a counting sort.
    """
    n, k = idx.shape
    clogc = np.zeros(n, dtype=np.float64)
    n_valid = np.zeros(n, dtype=np.int64)
    pair = np.zeros(n_kept * n_kept, dtype=np.int64)
    row_marg = np.zeros(n_sections * n_kept, dtype=np.int64)
    col_marg = np.zeros(n_sections * n_kept, dtype=np.int64)
    block = max(1, PAIR_BLOCK // max(1, k))

    for a in range(0, n, block):
        b = min(n, a + block)
        m = b - a
        sub = idx[a:b]
        # idx is -1 in every slot a section boundary or a missing coordinate
        # left unfillable, so the gather is done on a clipped copy and the
        # invalid slots are excluded by the mask rather than by pointing at
        # cell 0 and hoping. A partial neighbourhood is a smaller sample, not
        # a wrong one.
        filled = sub >= 0
        nbr = codes[np.clip(sub, 0, None)]
        ok = filled & (nbr >= 0)
        if centre is not None:
            ok &= centre[a:b, None]
        n_valid[a:b] = ok.sum(axis=1)

        rows = np.broadcast_to(np.arange(m, dtype=np.int64)[:, None], (m, k))
        flat_rows = rows[ok]
        if flat_rows.size:
            keys = flat_rows * n_types + nbr[ok]
            uniq, count = np.unique(keys, return_counts=True)
            # sum(c*log c) per cell. Entropy follows as log(N) - that / N,
            # which is the algebraically identical form that never divides by
            # N inside a log and so never turns a count of 1 into -0.0 noise.
            clogc[a:b] = np.bincount(uniq // n_types,
                                     weights=count * np.log(count),
                                     minlength=m)

        # The composition matrix needs BOTH ends labelled and both ends inside
        # the analysed set of types; the entropy above needs only the far end.
        # Two masks, one gather.
        # mcode carries one slot past the last type, holding -1, so that the
        # -1 of an unlabelled cell indexes it (Python's negative indexing) and
        # comes back as "not in the analysed set" instead of raising.
        own = mcode[codes[a:b]]
        nbr_m = mcode[nbr]
        okm = ok & (own[:, None] >= 0) & (nbr_m >= 0)
        if okm.any():
            own_b = np.broadcast_to(own[:, None], (m, k))[okm]
            nbr_b = nbr_m[okm]
            sec_b = np.broadcast_to(sec_dense[a:b, None], (m, k))[okm]
            pair += np.bincount(own_b * n_kept + nbr_b, minlength=n_kept * n_kept)
            row_marg += np.bincount(sec_b * n_kept + own_b, minlength=n_sections * n_kept)
            col_marg += np.bincount(sec_b * n_kept + nbr_b, minlength=n_sections * n_kept)
        ds.progress(0.3 + 0.6 * (b / n), f"{b:,} of {n:,} cells")

    return (clogc, n_valid,
            pair.reshape(n_kept, n_kept),
            row_marg.reshape(n_sections, n_kept),
            col_marg.reshape(n_sections, n_kept))


def _expected(row_marg, col_marg):
    """Stratified independence expectation: sum over sections of R_i K_j / G.

    A plain loop over sections rather than the equivalent (R/G).T @ K, for two
    reasons: there are at most a few hundred sections so the matmul buys
    nothing, and a BLAS product's summation order depends on the thread count,
    so its last digits could differ from one machine to the next.
    """
    n_sections, n_kept = row_marg.shape
    exp = np.zeros((n_kept, n_kept), dtype=np.float64)
    for s in range(n_sections):
        grand = float(row_marg[s].sum())
        if grand <= 0:
            continue                    # a section where nothing was labelled
        exp += np.outer(row_marg[s].astype(np.float64),
                        col_marg[s].astype(np.float64)) / grand
    return exp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(ds, params):
    # The host has already coerced and clamped every declared parameter, so the
    # defaults here are only for a direct call (a test, a notebook) — they are
    # the manifest's defaults and must stay in step with it.
    k = int(params.get("k") or DEFAULT_K)
    field = params.get("field") or None
    raw_floor = params.get("min_type_cells")
    # max(1, ...) rather than the raw floor: at the minimum of the declared
    # range the parameter is 0, and a floor of 0 would let a type with no
    # members become a row of numbers about nothing.
    floor = max(1, int(DEFAULT_MIN_TYPE_CELLS if raw_floor is None else raw_floor))

    levels, codes = _resolve_field(ds, field)
    n_types = len(levels)
    n = ds.n_cells

    # Which cells can be a neighbour of anything: `ds.knn` only ever returns
    # cells that have coordinates and, when the user has isolated a section,
    # only cells from it. The set of types PRESENT has to be counted over the
    # same population or the denominator promises a maximum that no
    # neighbourhood in this run could reach.
    placed = np.isfinite(ds.coords).all(axis=1)
    sect = np.asarray(ds.section, dtype=np.int64)
    if ds.section_filter is not None:
        placed &= (sect == int(ds.section_filter))
    if not placed.any():
        raise PluginError("no_coordinates",
                          "No cell in this view has spatial coordinates, so no "
                          "neighbourhood can be built.")
    placed_counts = np.bincount(codes[placed & (codes >= 0)], minlength=n_types)
    present = placed_counts > 0
    n_present = int(present.sum())
    if n_present < 2:
        raise PluginError(
            "one_category",
            f"Only {n_present} of the {n_types} levels of this annotation have "
            "any cell with coordinates here, so every neighbourhood is trivially "
            "uniform. Try a different column, or clear the section filter.")

    # The user's lasso restricts what is REPORTED, never the graph. Cutting the
    # neighbour search at the selection boundary would invent a tissue edge
    # where the user drew one, and every selected cell near that edge would
    # come back artificially unmixed.
    centre = None
    if ds.subset is not None and ds.subset.size:
        centre = np.zeros(n, dtype=bool)
        centre[ds.subset] = True

    # Types the matrix and table are built over. Below the floor a row is a
    # handful of neighbour observations and its enrichments are noise with a
    # confident-looking colour; the cap keeps the K x K accumulator bounded.
    order = np.argsort(-placed_counts, kind="stable")
    analysed = order[(placed_counts[order] >= floor)][:MAX_ANALYSED_TYPES]
    analysed = np.sort(analysed)
    if analysed.size < 2:
        raise PluginError(
            "no_types_pass_floor",
            f"No two categories have at least {floor} cells with coordinates, so "
            "there is no composition matrix to build. Lower 'Minimum cells per "
            "category'.")
    # One slot longer than the type list: codes == -1 lands on that last slot
    # (Python negative indexing) and reads back as "not analysed", so the
    # unlabelled case needs no branch inside the sweep.
    mcode = np.full(n_types + 1, -1, dtype=np.int64)
    mcode[analysed] = np.arange(analysed.size)
    n_kept = int(analysed.size)

    sec_levels, sec_dense = np.unique(sect, return_inverse=True)
    sec_dense = np.asarray(sec_dense, dtype=np.int64).reshape(n)
    n_sections = int(sec_levels.size)

    ds.log(f"{n:,} cells, {n_present} of {n_types} categories present, "
           f"{n_sections} section(s), k = {k}")
    ds.progress(0.05, "building the neighbour graph")
    idx, _ = ds.knn(k)

    ds.progress(0.3, "counting neighbourhoods")
    clogc, n_valid, obs, row_marg, col_marg = _sweep(
        ds, idx, codes, mcode, centre, sec_dense, n_types, n_kept, n_sections)

    # H = log(N) - sum(c log c)/N, then Pielou-normalised. The log base cancels
    # in the ratio, so the result is dimensionless and NOT in bits — calling it
    # bits would invite a reader to compare it with an unnormalised entropy.
    with np.errstate(divide="ignore", invalid="ignore"):
        counted = n_valid.astype(np.float64)
        mixing = (np.log(counted) - clogc / counted) / math.log(n_present)
    mixing[n_valid < MIN_LABELLED_NEIGHBOURS] = np.nan
    # Floating point can leave a uniform neighbourhood at -1e-17, which is
    # below the declared domain and paints as an out-of-range value.
    np.clip(mixing, 0.0, 1.0, out=mixing, where=np.isfinite(mixing))

    scored = np.isfinite(mixing)
    n_scored = int(scored.sum())
    if n_scored == 0:
        raise PluginError(
            "no_neighbourhoods",
            f"No cell has at least {MIN_LABELLED_NEIGHBOURS} labelled "
            f"neighbours among its {k} nearest. Either almost nothing in this "
            "view is annotated, or its sections are smaller than the "
            "neighbourhood.")

    typed = scored & (codes >= 0)
    per_n = np.bincount(codes[typed], minlength=n_types)
    per_sum = np.bincount(codes[typed], weights=mixing[typed], minlength=n_types)
    with np.errstate(invalid="ignore"):
        per_mean = per_sum / per_n

    exp = _expected(row_marg, col_marg)
    enrich = np.log2((obs + 1.0) / (exp + 1.0))
    # See the header: zero expected means no section held both categories, so
    # the zero observed is not a depletion, it is an absence of any test.
    enrich[exp <= 0] = np.nan

    # A type that survived the floor but was never observed next to anything —
    # every one of its cells lost its neighbourhood to a section edge, say —
    # has no measurement, and a row of pseudocount-only zeros would read as
    # "present everywhere at exactly chance".
    alive = obs.sum(axis=1) > 0
    if not alive.any():
        raise PluginError("no_neighbourhoods",
                          "No labelled cell has a labelled neighbour in this view.")
    analysed = analysed[alive]
    enrich = enrich[np.ix_(alive, alive)]
    obs = obs[np.ix_(alive, alive)]
    n_kept = int(analysed.size)

    labels = [levels[t] for t in analysed]
    kept_n = per_n[analysed]
    kept_mean = per_mean[analysed]

    # argmax over everything except the diagonal. Self is left out because in
    # real tissue almost every type is most enriched next to itself, so the
    # column would repeat the type name and say nothing; the self value gets
    # its own column instead, where it answers "is this type clustered".
    # -inf, not NaN, for the cells argmax must ignore: np.argmax returns the
    # position of a NaN rather than skipping it, so a type that never shared a
    # section with its neighbour would win every row.
    off = np.where(np.isfinite(enrich), enrich, -np.inf)
    np.fill_diagonal(off, -np.inf)
    top_j = np.argmax(off, axis=1) if n_kept > 1 else np.zeros(n_kept, dtype=int)
    has_partner = np.isfinite(off[np.arange(n_kept), top_j]) if n_kept > 1 \
        else np.zeros(n_kept, dtype=bool)

    by_count = np.argsort(-kept_n, kind="stable")
    rows = []
    for i in by_count:
        j = int(top_j[i])
        rows.append([
            labels[i],
            int(kept_n[i]),
            None if not np.isfinite(kept_mean[i]) else round(float(kept_mean[i]), 4),
            None if not np.isfinite(enrich[i, i]) else round(float(enrich[i, i]), 3),
            labels[j] if has_partner[i] else None,
            round(float(enrich[i, j]), 3) if has_partner[i] else None,
        ])

    show = by_count[:MAX_MATRIX_TYPES]
    ceiling = math.log(min(k, n_present)) / math.log(n_present)

    ds.progress(1.0, "done")
    return {
        "summary": _summary(ds, labels, kept_n, kept_mean, enrich, k, field,
                            n_present, n_types, n_scored, n, n_sections, floor,
                            ceiling, show.size),
        "cells": {
            "values": mixing,
            "label": "Neighbourhood mixing",
            "colormap": "viridis",
            "diverging": False,
            # Fixed to the range this run can actually produce, not [0, 1]:
            # with k < T the top of the scale is unreachable and a [0, 1] ramp
            # would spend a third of its colours on values that cannot occur.
            "domain": [0.0, round(ceiling, 4)],
            "units": "",
        },
        "matrix": {
            "title": f"Neighbour enrichment, log2 observed / expected (k = {k})",
            "rows": [labels[i] for i in show],
            "cols": [labels[i] for i in show],
            "values": enrich[np.ix_(show, show)],
            "diverging": True,
            "colormap": "icefire",
        },
        "table": {
            "title": "Per category",
            "columns": ["Category", "Cells scored", "Mean mixing",
                        "Self log2 O/E", "Most enriched neighbour", "log2 O/E"],
            "rows": rows,
            "sort_by": "Mean mixing",
        },
    }


def _summary(ds, labels, kept_n, kept_mean, enrich, k, field,
             n_present, n_types, n_scored, n, n_sections, floor, ceiling, n_shown):
    """The numbers a reader needs before the three panels mean anything."""
    finite = np.isfinite(kept_mean)
    ranked = np.argsort(np.where(finite, kept_mean, np.nan), kind="stable")
    ranked = [int(i) for i in ranked if finite[i]]
    low = ranked[0] if ranked else None
    high = ranked[-1] if ranked else None
    diag = float(np.nanmean(np.diag(enrich)))
    name = field or (ds.meta.get("celltype_obs") or "cell type")

    # `ceiling` states the top of the scale as a number, so the summary never
    # has to walk through the arithmetic behind it: at k >= the categories
    # present it is 1.00, below that it is what a cell scores when all k
    # neighbours differ.
    parts = [
        f"**Neighbourhood mixing** over `{name}` at k = {k} — **{n_scored:,} of "
        f"{n:,} cells** scored across {n_sections} section(s), the rest NaN. "
        f"Entropy over log({n_present}): {n_present} of {n_types} categories are "
        f"present here, so the scale tops out at **{ceiling:.2f}**.",
    ]
    if high is not None and low is not None and high != low:
        parts.append(
            f"Most mixed: **{labels[high]}** ({kept_mean[high]:.2f}, "
            f"{int(kept_n[high]):,} cells). Least mixed: **{labels[low]}** "
            f"({kept_mean[low]:.2f}, {int(kept_n[low]):,} cells).")
    parts.append(
        f"Matrix: log2 observed/expected neighbour counts, expected within each "
        f"section, so **0 is chance**; the diagonal averages **{diag:+.2f}**. "
        + (f"Showing all {n_shown} categories with at least {floor} cells."
           if n_shown == len(labels) else
           f"Showing the {n_shown} most abundant of {len(labels)} categories "
           f"with at least {floor} cells; the table lists them all."))
    blank = int(np.isnan(enrich).sum())
    if blank:
        parts.append(
            f"{blank} of the {enrich.size} matrix entries are blank: no section "
            f"contains both of those categories.")
    return "\n\n".join(parts)
