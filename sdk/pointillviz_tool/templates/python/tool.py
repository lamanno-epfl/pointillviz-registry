"""{{NAME}}: per-cell transcript totals, the starting point.

`run(ds, params)` is the only symbol PointillViz looks for. `ds` is the dataset
the user is looking at (`pointillviz_tool.Dataset`); `params` holds the values
of the parameters declared in plugin.json, already coerced to the right Python
type and clamped into their declared range.
"""

import numpy as np

from pointillviz_tool import PluginError


def run(ds, params):
    counts = np.asarray(ds.total_counts(), dtype=np.float64)
    floor = float(params["min_counts"])

    # NaN rather than 0 for the cells below the floor. The viewer DIMS a NaN
    # cell and paints a 0 at the bottom of the colour ramp, and "not counted
    # here" is a different statement from "counted zero".
    values = np.where(counts >= floor, counts, np.nan)
    shown = int(np.isfinite(values).sum())
    ds.log(f"{shown} of {ds.n_cells} cells are at or above {floor:g} transcripts")

    # THE BRANCH TO COPY INTO YOUR OWN TOOL. An all-NaN map and a real result
    # are the same picture once they are painted, so a tool that has nothing to
    # say must SAY so rather than hand back a blank slide: the user cannot tell
    # "the threshold excluded everything" from "this tool is broken" by looking.
    # PluginError is the way to refuse an input in words the user sees: name
    # the number that failed and the move that fixes it. Anything else you
    # raise is reported as a crash with a traceback, which is right for a bug
    # and wrong for "nothing qualified".
    if shown == 0:
        highest = float(np.nanmax(counts)) if counts.size else 0.0
        raise PluginError(
            "threshold_unreachable",
            f"No cell reaches {floor:,.0f} transcripts; the highest here has "
            f"{highest:,.0f}. Lower the minimum count and run again.")

    median = float(np.nanmedian(values))
    return {
        "summary": (f"**{shown:,} of {ds.n_cells:,} cells** carry at least "
                    f"{floor:g} transcripts. The median among them is "
                    f"{median:,.0f}."),
        # Exactly one value per cell, in dataset order: the one convention the
        # whole tool contract rests on.
        "cells": {
            "values": values,
            "label": "Total counts",
            "colormap": "viridis",
            "units": "transcripts",
        },
    }
