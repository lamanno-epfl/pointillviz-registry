"""Pairwise adjacency enrichment: log2(observed / expected) per type pair.

`observed` is how many within-radius pairs of cells carry each pair of labels.
`expected` is what the same number of pairs would give if the labels were
scattered at random over the same cells. The ratio is what makes the matrix
readable across datasets — raw adjacency counts mostly measure how common each
type is, which the user already knows.

The expectation is closed-form rather than permuted, for the reason
PointillViz's built-in ligand-receptor Moran's R gives: a permutation null costs seconds per run and
buys nothing here, because the quantity being nulled is a multinomial pair
count with a known mean.

Pairs are counted WITHIN A SECTION. Sections are laid out side by side on a
shared canvas, so a radius query over everything at once would invent
neighbours across specimens — `ds.kdtrees()` returns one tree per section
precisely so a tool cannot make that mistake by accident.
"""

import numpy as np

from pointillviz_tool import PluginError

#: The result contract caps a matrix at 10,000 cells, and a 100x100 heatmap is
#: already past what anyone reads. Refusing with a sentence beats returning a
#: matrix the viewer will reject with a limit message the author cannot act on.
MAX_LEVELS = 100


def run(ds, params):
    radius = float(params["radius"])
    labels, codes = ds.categorical(params.get("field") or None)
    n_levels = len(labels)
    if n_levels < 2:
        raise PluginError("one_level", "This annotation has only one level; a "
                                       "co-occurrence matrix needs at least two.")
    if n_levels > MAX_LEVELS:
        raise PluginError(
            "too_many_levels",
            f"This annotation has {n_levels} levels, and the matrix would have "
            f"{n_levels * n_levels:,} cells. Pick a coarser grouping "
            f"({MAX_LEVELS} levels is the ceiling).")

    ds.progress(0.1, f"finding pairs within {radius:g}")
    obs = np.zeros((n_levels, n_levels), dtype=np.float64)
    n_pairs = 0
    labelled_total = 0
    for part in ds.kdtrees():
        rows = part["rows"]
        # output_type="ndarray" instead of the default set(): a set of tuples
        # is a different iteration order every process.
        pairs = part["tree"].query_pairs(radius, output_type="ndarray")
        a = codes[rows[pairs[:, 0]]] if pairs.size else np.empty(0, dtype=np.int32)
        b = codes[rows[pairs[:, 1]]] if pairs.size else np.empty(0, dtype=np.int32)
        keep = (a >= 0) & (b >= 0)
        a, b = a[keep], b[keep]
        n_pairs += int(a.size)
        flat = np.bincount(a.astype(np.int64) * n_levels + b.astype(np.int64),
                           minlength=n_levels * n_levels).reshape(n_levels, n_levels)
        # query_pairs reports each unordered pair once, in whichever order the
        # tree happened to store it, so the counts are symmetrised here rather
        # than left as a triangular matrix the reader has to mentally fold.
        obs += flat + flat.T
        labelled_total += int((codes[rows] >= 0).sum())

    ds.progress(0.7, "comparing against a label-shuffled expectation")
    counts = np.bincount(codes[codes >= 0].astype(np.int64), minlength=n_levels).astype(np.float64)
    share = counts / counts.sum() if counts.sum() else np.zeros(n_levels)
    # Under a random relabelling, an ordered pair of endpoints is (i, j) with
    # probability share_i * share_j, and `obs` counts every unordered pair
    # twice (once per direction) — so both sides are in the same units.
    expected = 2.0 * n_pairs * np.outer(share, share)

    if n_pairs == 0:
        # A radius that finds nothing is a legitimate answer to a legitimate
        # question, not a crash: the user asked "who touches whom within 5 µm"
        # and the answer is "nobody". NaN keeps it out of the colour ramp.
        values = np.full((n_levels, n_levels), np.nan)
    else:
        # +1 in both places is a Laplace smoothing that keeps an unseen pair at
        # a finite, strongly negative log-ratio instead of -inf, which the
        # result contract rejects outright (and rightly — an infinity is not a
        # measurement).
        values = np.log2((obs + 1.0) / (expected + 1.0))

    ds.progress(0.9, "ranking the strongest pairs")
    table_rows = []
    for i in range(n_levels):
        for j in range(i, n_levels):
            if not np.isfinite(values[i, j]):
                continue
            table_rows.append([labels[i], labels[j], int(obs[i, j] / (1 if i != j else 2)),
                               round(float(expected[i, j] / (1 if i != j else 2)), 1),
                               round(float(values[i, j]), 3)])
    table_rows.sort(key=lambda r: (-r[4], r[0], r[1]))
    table_rows = table_rows[:200]

    if n_pairs:
        top = table_rows[0]
        summary = (
            f"**{top[0]} ↔ {top[1]}** is the most enriched contact at "
            f"{top[4]:+.2f} log2 over chance.\n\n"
            f"{n_pairs:,} within-radius pairs over {labelled_total:,} labelled "
            f"cells, counted per section at r = {radius:g}. The matrix is log2 "
            f"observed/expected across {n_levels} groups, 0 = chance; the table "
            f"ranks the {len(table_rows)} strongest pairs.")
    else:
        # A radius that finds nothing gets no legend for a colour ramp that is
        # entirely NaN — it gets the one number the user can act on.
        summary = (
            f"**No two cells are within {radius:g} of each other.**\n\n"
            f"{labelled_total:,} labelled cells, 0 pairs, so every matrix entry "
            f"is blank. Raise the radius.")

    return {
        "summary": summary,
        "matrix": {"title": f"Adjacency enrichment (log2 obs/exp, r={radius:g})",
                   "rows": list(labels), "cols": list(labels),
                   "values": values.tolist(), "diverging": True,
                   "colormap": "icefire"},
        "table": {"title": "Strongest contacts",
                  "columns": ["Group A", "Group B", "Pairs", "Expected", "log2 obs/exp"],
                  "rows": table_rows, "sort_by": "log2 obs/exp"},
    }
