"""Local cell density from the k-th nearest-neighbour distance.

The k-NN estimator, not a grid histogram: a grid needs a bin size the user has
to guess, and it puts a visible lattice into a map of something continuous. The
k-th neighbour radius adapts on its own — dense tissue gets a small radius,
sparse tissue a large one — and the only knob is how much smoothing you want.

    density = (neighbours found) / (area or volume of the enclosing ball)

Needs no annotation at all, which makes it the first thing worth running on a
dataset nobody has looked at yet: it shows tissue borders, folds, section edges
and segmentation failures in one pass.
"""

import numpy as np


def run(ds, params):
    k = int(params["k"])
    log_scale = str(params.get("scale") or "log10") == "log10"

    ds.progress(0.15, f"building the {k}-nearest-neighbour graph")
    idx, dist = ds.knn(k)

    # A slot is usable only if the tree actually filled it. `dist` is inf and
    # `idx` is -1 for a cell with no coordinates and for the tail slots of a
    # section smaller than k+1, and inf would poison the max below.
    usable = (idx >= 0) & np.isfinite(dist)
    found = usable.sum(axis=1)
    radius = np.where(usable, dist, -np.inf).max(axis=1)

    ds.progress(0.6, "converting radius to density")
    dims = ds.coords.shape[1]
    # Volume of the enclosing ball in whatever dimensionality the geometry was
    # done in: pi r^2 in 2D, 4/3 pi r^3 in 3D. Using the 2D formula on a 3D
    # dataset would report a density in the wrong units and quietly rank cells
    # by the wrong thing.
    if dims >= 3:
        volume = (4.0 / 3.0) * np.pi * np.power(np.maximum(radius, 0.0), 3)
        units = "cells / unit³"
    else:
        volume = np.pi * np.square(np.maximum(radius, 0.0))
        units = "cells / unit²"

    # radius == 0 happens for real: duplicated coordinates from a segmentation
    # that emitted two cells at one centroid. That is an infinite density,
    # which is not a measurement — and the result contract rejects infinities
    # outright — so those cells come back blank.
    ok = (found > 0) & np.isfinite(radius) & (radius > 0)
    values = np.full(ds.n_cells, np.nan)
    values[ok] = found[ok] / volume[ok]

    label = "Local density"
    if log_scale:
        values[ok] = np.log10(values[ok])
        label = "log10 local density"

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        # Every cell blank is a real outcome (a dataset with no usable
        # coordinates at all) and the tool still has to return the outputs it
        # declared, or the result gate is right to fail it.
        return {
            "summary": "**No cell had a usable neighbourhood.** Every cell is "
                       "either missing coordinates or sits in a section with "
                       f"fewer than {k + 1} positioned cells.",
            "cells": {"values": values.tolist(), "label": label, "colormap": "viridis",
                      "units": units},
            "series": {"title": "Density distribution", "x_label": "percentile",
                       "y_label": label,
                       "lines": [{"label": "no data", "x": [0.0, 100.0], "y": [np.nan, np.nan]}]},
        }

    ds.progress(0.85, "building the distribution curve")
    # The sorted curve, resampled onto a fixed 201-point percentile grid. Fixed
    # because the series ceiling is 5,000 points and a 10 M-cell dataset would
    # otherwise have to be thinned by a rule invented here; a percentile grid is
    # also directly readable ("the top 5% of cells are above this line").
    pct = np.linspace(0.0, 100.0, 201)
    curve = np.percentile(finite, pct)
    lo, med, hi = (float(np.percentile(finite, 1)), float(np.median(finite)),
                   float(np.percentile(finite, 99)))
    blank = int(ds.n_cells - finite.size)

    summary = (
        f"**Median {label.lower()} {med:.3g}** ({units}), 1st–99th percentile "
        f"{lo:.3g} to {hi:.3g}, from the {k}-th nearest neighbour of "
        f"{finite.size:,} cells in {dims}D."
        # "1 cells ... are blank" is how a generated line announces that nobody
        # read it, and a single blank cell is the common case on a clean section.
        + ("" if not blank else
           f" {blank:,} cell had no usable neighbourhood and is blank." if blank == 1
           else f" {blank:,} cells had no usable neighbourhood and are blank."))

    return {
        "summary": summary,
        "cells": {"values": values.tolist(), "label": label,
                  "colormap": "viridis", "units": units,
                  "domain": [lo, hi]},
        "series": {"title": "Density distribution",
                   "x_label": "percentile of cells", "y_label": label,
                   "lines": [{"label": label, "x": pct.tolist(), "y": curve.tolist()}],
                   "rules": [med]},
    }
