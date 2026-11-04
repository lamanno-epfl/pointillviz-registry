"""Ridge map: a tissue section as a relief, one ridge per horizontal strip.

The section is cut into horizontal strips. Each strip is drawn, from the back of the figure to its
front, as a ridge along x: its height is a gene's local expression, or the local cell density when
no gene is chosen, and each stretch takes the colour of the cell type most common there. A ridge
hides what stands behind it, on black, so the tissue's architecture reads as a landscape.

Height and colour come from one round Gaussian kernel, so the strips are cross-sections of one
smooth surface and a peak lines up from one strip to the next. The figure keeps the section's
proportions and the map's orientation: the viewer draws small y at the top, and so does SVG.
"""

import colorsys
import html
import math
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d, label

from pointillviz_tool import PluginError, download_file, image

MIN_CELLS = 30          # cells a section needs to be drawn
MAX_FIGURES = 8         # sections drawn at most: the largest
MIN_PER_STRIP = 15      # fewer strips than asked when they would hold fewer cells than this
MIN_UNDER_KERNEL = 5    # the kernel widens when it would hold fewer cells than this, on average
TISSUE_MIN = 1.5        # a place is tissue when the kernel holds at least this many cells ...
TISSUE_SHARE = 0.1      # ... and this share of what it holds around the section's typical cell
MIN_PATCH = 0.0005      # a patch of tissue holds this share of the section's cells, or it is strays
MIN_PATCH_CELLS = 3     # ... and at least this many
RAMP = 3.0              # a ridge comes down to its baseline over this many σ where the tissue ends
OUTPUT_LIMIT = 8 * 1024 * 1024  # images and their downloads share this, in bytes on the wire
MAX_KEY = 60            # cell types named in a key
MAX_NAMED = 5           # cell types named in a sentence of the summary
NOTE_UNLABELLED = 0.01  # the summary speaks of unlabelled cells from this share of those drawn

# The figure, in px.
MAP_W, MAP_H = 980, 800             # the box the largest section fits in
PAD, KEY_GAP = 34, 36
KEY_LINE, SCALE_BAR = 21, 180
MAX_RELIEF = MAP_H / 2              # the tallest ridge
MIN_SPACING = 2.5                   # between strips
MIN_STEP, MAX_STEP = 1.6, 2.5       # between points on a ridge, unless the output limit asks more
MIN_STRETCH = 5.0                   # a stretch of one colour, whatever the number of strips
LINE = 0.2                          # a ridge's line, in strip heights ...
LINE_MIN, LINE_MAX = 0.5, 1.5       # ... within these
HALO = 1.8                          # the black a ridge clears around itself
BODY = 0.34                         # the body under a ridge: its colour at this share over black
GROUND, INK, MUTED = "#000000", "#F2F2F2", "#8C8C96"
UNLABELLED = MUTED
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
# The viewer's own palette, for a dataset with no colours; past it, hues a golden angle apart.
PALETTE = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
           "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff", "#1abc9c", "#ff6348", "#7bed9f",
           "#ff4757", "#70a1ff", "#ffa502", "#2ed573", "#5352ed", "#ff6b81", "#dfe6e9", "#fd79a8",
           "#00cec9", "#e17055", "#0984e3", "#6c5ce7", "#fdcb6e", "#fab1a0", "#74b9ff", "#a29bfe")


def run(ds, params):
    gene = params.get("gene") or None
    levels, codes = ds.cell_types()
    values = _expression(ds, gene) if gene else None
    parts, scope, left = _sections(ds, values)
    if gene and not any((values[p["rows"]] > 0).any() for p in parts):
        n = sum(p["rows"].size for p in parts)
        raise PluginError("gene_not_detected",
                          f"{gene} is not detected in any of the {n:,} cells drawn here. "
                          "Choose another gene, or clear the gene to draw cell density.")

    # ---- one scale for every figure: px per unit, strip height, kernel, height per value ----
    strips = int(params["strips"])
    max_w = max(np.ptp(p["x"]) for p in parts)
    max_h = max(np.ptp(p["y"]) for p in parts)
    scale = min(MAP_W / max_w, MAP_H / max_h)
    caps = {f"each strip needs at least {MIN_PER_STRIP} cells":
            max(p["rows"].size for p in parts) // MIN_PER_STRIP,
            "more would not separate at this figure's size": int(max_h * scale / MIN_SPACING)}
    why = min(caps, key=caps.get)
    n_strips = max(2, min(strips, caps[why]))
    area = sum(np.ptp(p["x"]) * np.ptp(p["y"]) for p in parts)
    units = str((ds.meta.get("spatial") or {}).get("units") or "")
    colours, lightened = _palette(levels, ds.cell_type_colors)
    frame = {"gene": gene, "levels": levels, "colours": colours, "units": units, "scale": scale,
             "max_h": float(max_h), "smoothing": float(params["smoothing"]),
             "relief": float(params["relief"]),
             "sparse": math.sqrt(MIN_UNDER_KERNEL * area
                                 / (2 * math.pi * sum(p["rows"].size for p in parts))),
             "names": {int(s.get("index", i)): str(s.get("section_id") or "")
                       for i, s in enumerate(ds.sections)}}

    # What fits the output limit is measured on the figures themselves. When they run over: fewer
    # points on each ridge, as far as the kernel's width; then longer stretches of one colour; and
    # only then fewer strips, the most that fit.
    coarse, longer, limited = 0.0, 1, False
    ds.progress(0.1, "measuring and drawing the ridges")
    shot = _draw(parts, values, codes, frame, n_strips, coarse, longer)
    while shot["wire"] > OUTPUT_LIMIT and shot["step"] < shot["sigma"] * scale:
        coarse = shot["step"] * shot["wire"] / OUTPUT_LIMIT * 1.02
        shot = _draw(parts, values, codes, frame, n_strips, coarse, longer)
    if shot["wire"] > OUTPUT_LIMIT:
        longer = 2
        shot = _draw(parts, values, codes, frame, n_strips, coarse, longer)
    if shot["wire"] > OUTPUT_LIMIT:
        limited = True
        n_strips, shot = _most_strips(
            lambda n: _draw(parts, values, codes, frame, n, coarse, longer), n_strips, shot)
    shown, drawn, sigma, step, top = (shot[k] for k in ("shown", "drawn", "sigma", "step", "top"))
    many = len(shown) > 1

    images, files, used = [], [], set()
    for p, (svg, _, _) in zip(shown, drawn):
        title = f"Ridge map: {p['name']}" if many else "Ridge map"
        name = f"ridge-map-{gene or 'density'}" + (f"-{p['name']}" if many else "")
        name = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)[:120]
        if name in used:
            name += f"-{p['section'] + 1}"
        used.add(name)
        images.append(image(svg, title))
        files.append(download_file(name + ".svg", svg, media_type="image/svg+xml"))

    # ---- the summary: the numbers of this run, and the caveats that depend on it ----
    length = sum(d[1] for d in drawn)
    height = sum(d[2] for d in drawn)
    unit, factor = _height_unit(gene, units)
    inside = np.concatenate([codes[p["drawn"]] for p in shown])
    n_cells = inside.size
    n_ridges = sum(int(p["on"].any(axis=1).sum()) for p in shown)
    whose = "selected cells" if ds.subset is not None and ds.subset.size else "cells"
    held = f"{n_cells:,} {whose}" if n_cells == scope else f"{n_cells:,} of {scope:,} {whose}"
    across = f" across {len(shown)} sections" if many else ""
    lines = [f"**{gene or 'Cell density'}** over {held}: {n_ridges:,} strips of "
             f"{_length(shot['thick'], units)}{across}. Height: {unit} within a Gaussian of "
             f"σ = {_length(sigma, units)}, reaching {_value(top * factor)}."]
    long_enough = length[:-1] >= 0.01 * length.sum()
    mean = np.where(long_enough, height[:-1] / np.maximum(length[:-1], 1e-9), -1.0)
    tallest = [i for i in np.argsort(-mean)[:3] if mean[i] > 0]
    if tallest:
        over_types = ", ".join(f"{levels[i]} ({_value(mean[i] * factor)})" for i in tallest)
        lines.append(f"Tallest on average where the most common cell type is {over_types}.")

    notes = []
    sparse_or_stray = scope - n_cells - sum(n for n, _ in left)
    if sparse_or_stray:
        left.append((sparse_or_stray, "sparse or stray"))
    if left:
        notes.append("Cells not drawn: " + ", ".join(f"{n:,} {r}" for n, r in left) + ".")
    if n_strips < strips:
        if limited:
            why = "more would not fit the output limit"
            why += "; isolate a section for more" if many else ""
        each = " a section" if many else ""
        notes.append(f"{n_strips} strips{each} cut rather than {strips}: {why}.")
    notes += shot["notes"]
    shortest = _shortest_bins(longer, step) * step
    if longer > 1:
        notes.append(f"Stretches of one colour are at least {shortest:.0f} px long here rather "
                     f"than {MIN_STRETCH:g}: the figure would not fit the output limit otherwise.")
    n_blank = int((inside < 0).sum())
    if n_blank and (length[-1] > 0 or n_blank >= NOTE_UNLABELLED * n_cells):
        grey = " Grey stretches hold no labelled cell." if length[-1] > 0 else ""
        notes.append(f"{n_blank:,} of the cells drawn ({100 * n_blank / n_cells:.1f}%) have no "
                     f"cell type: they count toward the height, not the colours.{grey}")
    present = np.unique(inside[inside >= 0]).tolist()
    unseen = [levels[i] for i in present if length[i] <= 0]
    if unseen:
        verb = "is" if len(unseen) == 1 else "are"
        notes.append(f"{len(unseen)} of {len(present)} cell types {verb} present but not drawn: "
                     f"none is the most common over a stretch of {_length(shortest / scale, units)}"
                     f"{_named(unseen)}.")
    lifted = [levels[i] for i in lightened if length[i] > 0]
    if lifted:
        notes.append(f"{len(lifted)} colour{'s' if len(lifted) > 1 else ''} lightened to show on "
                     f"black{_named(lifted)}.")
    if notes:
        lines.append(" ".join(notes))
    return {"summary": "\n\n".join(lines), "images": images, "files": files}


def _most_strips(draw, too_many, shot):
    """The most strips whose figures fit the output limit, and their figures, found by trying:
    `too_many` strips, drawn as `shot`, did not fit."""
    fits, n = None, too_many
    while fits is None or too_many - fits > 1:
        if fits is not None:
            n = (fits + too_many) // 2
        elif too_many > 2:
            n = max(2, min(too_many - 1, int(too_many / math.sqrt(shot["wire"] / OUTPUT_LIMIT))))
        else:
            raise PluginError("figure_too_large", "These cells make a figure too large to return. "
                              "Isolate a section, or select fewer cells.")
        shot = draw(n)
        if shot["wire"] > OUTPUT_LIMIT:
            too_many = n
        else:
            fits = n
    return fits, shot if n == fits else draw(fits)


def _draw(parts, values, codes, frame, n_strips, coarse, longer):
    """Measure and draw every section at this number of strips. `coarse` asks for fewer points on
    each ridge, `longer` for longer stretches of one colour. Says how many bytes the figures take on
    the wire, each counted as an image and again as a base64 download."""
    scale = frame["scale"]
    thick = frame["max_h"] / n_strips
    spacing = thick * scale
    notes = []
    narrowest = max(frame["sparse"], MIN_STEP / scale)
    sigma = frame["smoothing"] * thick
    if sigma < narrowest:
        cause = ("these cells are sparse" if frame["sparse"] >= MIN_STEP / scale
                 else "the figure draws no finer")
        sigma = narrowest
        notes.append(f"The smoothing widened to {sigma / thick:.2f} strip heights: {cause}.")
    # between the points of a ridge, never more than the kernel's width: σ is what the kernel does
    sigma_px = sigma * scale
    step = max(MIN_STEP, min(sigma_px / 2, MAX_STEP), min(coarse, sigma_px))

    for p in parts:
        if p.get("at") != (thick, step, sigma):
            _measure(p, values, codes, thick, step / scale, sigma)
            p["at"] = (thick, step, sigma)
    shown = [p for p in parts if p["on"].any()]
    if not shown:
        raise PluginError("too_sparse", "No strip holds enough cells close together here. Raise "
                          "the smoothing, or widen the selection.")
    top = max(float(p["value"][p["on"]].max()) for p in shown)
    if top <= 0:
        cause = (f"{frame['gene']} is detected only in stray cells" if frame["gene"]
                 else "these cells are too scattered")
        raise PluginError("flat", f"Nothing rises here: {cause}. Choose another gene, widen the "
                          "selection, or raise the smoothing.")
    relief = frame["relief"]
    if relief * spacing > MAX_RELIEF:
        relief = MAX_RELIEF / spacing
        notes.append(f"The relief held at {relief:.1f} strip heights: half the figure's height.")

    scene = {"gene": frame["gene"], "levels": frame["levels"], "colours": frame["colours"],
             "units": frame["units"], "scale": scale, "spacing": spacing, "top": top, "step": step,
             "per_value": relief * spacing / top}
    for p in shown:
        name = frame["names"].get(p["section"]) or f"section {p['section'] + 1}"
        p["name"] = name[:60] if len(shown) > 1 else ""
    drawn = [_figure(p, scene, longer) for p in shown]
    sizes = [len(d[0].encode("utf-8")) for d in drawn]
    return {"shown": shown, "drawn": drawn, "notes": notes, "thick": thick, "sigma": sigma,
            "step": step, "top": top, "wire": sum(n + 4 * math.ceil(n / 3) for n in sizes)}


def _named(names):
    return ": " + ", ".join(names) if len(names) <= MAX_NAMED else ""


def _expression(ds, gene):
    """log1p(CP10K) per cell, NaN wherever the cell's section did not measure the gene: not measured
    is not zero. An SDK that predates sections of different panels gives an empty column there as
    zeros, so the sections' own lists are read as well."""
    values = np.array(ds.gene(gene, "log1p"), dtype=np.float64)
    position = ds.gene_names.index(gene)
    for i, s in enumerate(ds.sections):
        if position in (s.get("unmeasured_genes") or ()):
            values[ds.section == int(s.get("index", i))] = np.nan
    return values


def _sections(ds, values):
    """The cells to draw, one entry per section; how many cells this run is about; and the cells
    left out so far, as (how many, why). With a gene, only cells whose section measured it: not
    measured is not zero."""
    xy = ds.xy
    sect = ds.section
    scope = np.ones(ds.n_cells, dtype=bool)
    if ds.section_filter is not None:
        scope &= sect == ds.section_filter
    if ds.subset is not None and ds.subset.size:
        chosen = np.zeros(ds.n_cells, dtype=bool)
        chosen[ds.subset] = True
        scope &= chosen
    placed = scope & np.isfinite(xy).all(axis=1)
    left = []
    if (scope & ~placed).any():
        left.append((int((scope & ~placed).sum()), "with no coordinates"))
    if values is not None:
        measured = np.isfinite(values)
        if placed.any() and not (placed & measured).any():
            raise PluginError("gene_not_measured", "This gene was not measured in any of the cells "
                              "here. Isolate another section, choose another gene, or clear the "
                              "gene to draw cell density.")
        unmeasured = int((placed & ~measured).sum())
        if unmeasured:
            ids = {int(s.get("index", i)): str(s.get("section_id") or "")
                   for i, s in enumerate(ds.sections)}
            whole = [ids.get(int(s)) or f"section {int(s) + 1}" for s in np.unique(sect[placed])
                     if not (measured[placed & (sect == s)]).any()]
            where = f" ({', '.join(whole)})" if 0 < len(whole) <= MAX_NAMED else ""
            left.append((unmeasured, f"where this gene was not measured{where}"))
        placed &= measured

    parts, far, small, small_cells = [], 0, 0, 0
    for s in np.unique(sect[placed]):
        every = np.flatnonzero(placed & (sect == s))
        rows = every[_within_reach(xy[every, 0]) & _within_reach(xy[every, 1])]
        far += every.size - rows.size
        x, y = xy[rows, 0].astype(np.float64), xy[rows, 1].astype(np.float64)
        if rows.size < MIN_CELLS or np.ptp(x) <= 0 or np.ptp(y) <= 0:
            small += 1
            small_cells += rows.size
            continue
        parts.append({"section": int(s), "rows": rows, "x": x, "y": y})
    if not parts:
        raise PluginError("too_few_cells",
                          f"A ridge map needs at least {MIN_CELLS} cells with coordinates, spread "
                          "over an area; no section has that here. Widen the selection, or isolate "
                          "another section.")
    if far:
        left.append((far, "far outside the section"))
    if small_cells:
        left.append((small_cells,
                     f"in {small} section{'s' if small > 1 else ''} too small to draw"))
    if len(parts) > MAX_FIGURES:
        kept = sorted(parts, key=lambda p: -p["rows"].size)[:MAX_FIGURES]
        n_over = len(parts) - MAX_FIGURES
        left.append((sum(p["rows"].size for p in parts) - sum(p["rows"].size for p in kept),
                     f"in {n_over} section{'s' if n_over > 1 else ''} beyond the {MAX_FIGURES} "
                     "largest (isolate one to draw it)"))
        parts = sorted(kept, key=lambda p: p["section"])
    return parts, int(scope.sum()), left


def _within_reach(v):
    """False for cells past an empty gap as wide as the middle 90% of the section: too few and too
    far off to be anything but strays, and they would otherwise decide the whole figure's scale."""
    lo, hi = np.percentile(v, [5, 95])
    if hi <= lo:
        return np.ones(v.size, dtype=bool)
    s = np.sort(v)
    gaps = np.flatnonzero(np.diff(s) >= hi - lo)
    below, above = gaps[s[gaps + 1] <= lo], gaps[s[gaps] >= hi]
    first = s[below[-1] + 1] if below.size else s[0]
    last = s[above[0]] if above.size else s[-1]
    return (v >= first) & (v <= last)


def _measure(p, values, codes, thick, bin_w, sigma):
    """One section's strips, per strip and x bin: whether there is tissue (`on`), the height's value
    (`value`) and the cell type most common there (`kinds`, -1 where no cell has a label).

    Every cell weighs by its true distance to the strip's line, and lies between two nodes of a
    grid no wider than σ/4. Dropping cells into their nearest bin instead beats against a regular
    lattice of pixels, and a uniform section then draws waves that are not there."""
    x, y, rows = p["x"], p["y"], p["rows"]
    margin = 4.0 * sigma
    x0, y0 = float(x.min()) - margin, float(y.min())
    n_bins = int(math.ceil((np.ptp(x) + 2 * margin) / bin_w)) + 1
    n_rows = max(1, int(math.ceil(np.ptp(y) / thick - 1e-9)))
    shape = (n_rows, n_bins)

    # the nodes, however far apart the points of a ridge are: a bin's centre is one of them
    per_bin = 2 * max(1, int(math.ceil(2 * bin_w / sigma)))
    gap = bin_w / per_bin
    n_nodes = per_bin * n_bins + 1
    pos = (x - x0) / gap
    node = np.minimum(pos.astype(np.int64), n_nodes - 2)
    frac = pos - node
    width = sigma / gap

    def along_x(where, part, weights, n=1):
        """The kernel along x of what lies at these nodes: n rows of bins, one for each group."""
        grid = np.bincount(where, weights * (1 - part), minlength=n * n_nodes) \
            + np.bincount(where + 1, weights * part, minlength=n * n_nodes)
        return gaussian_filter1d(grid.reshape(n, n_nodes), width, axis=1, mode="constant",
                                 truncate=4.0)[:, per_bin // 2::per_bin]

    order = np.argsort(y)
    y_sorted = y[order]
    labels = codes[rows]
    measure = None if values is None else values[rows]
    smooth = np.zeros(shape)
    value = np.zeros(shape)
    kinds = np.full(shape, -1, dtype=np.int32)
    for r in range(n_rows):
        line = y0 + (r + 0.5) * thick
        lo, hi = np.searchsorted(y_sorted, [line - margin, line + margin])
        near = order[lo:hi]
        if not near.size:
            continue
        w = np.exp(-((y[near] - line) ** 2) / (2 * sigma * sigma))
        smooth[r] = along_x(node[near], frac[near], w)[0]
        if measure is not None:
            value[r] = along_x(node[near], frac[near], w * measure[near])[0]
        have = near[labels[near] >= 0]
        if have.size:
            present, group = np.unique(labels[have], return_inverse=True)
            hist = along_x(group * n_nodes + node[have], frac[have], w[labels[near] >= 0],
                           present.size)
            won = hist.max(axis=0) > 0
            kinds[r, won] = present[hist.argmax(axis=0)[won]]

    # `smooth` is each cell's weight on the strip, at 1 where the cell lies, in cells per node
    under = smooth * width * math.sqrt(2 * math.pi)     # cells under the kernel, at its peak 1 each
    if measure is None:
        # to cells per square unit: per node along x, and the strip's weight integrates to √(2π)σ
        value = smooth / (gap * math.sqrt(2 * math.pi) * sigma)
    else:
        value = value / np.maximum(smooth, 1e-12)

    # each cell's own bin, to count what is drawn
    s = np.minimum(((y - y0) / thick).astype(np.int64), n_rows - 1)
    b = np.minimum(((x - x0) / bin_w).astype(np.int64), n_bins - 1)
    count = np.bincount(s * n_bins + b, minlength=n_rows * n_bins).reshape(shape)

    typical = float((count * under).sum() / count.sum())
    on = under >= max(TISSUE_MIN, TISSUE_SHARE * typical)
    patches, n_patches = label(on, structure=np.ones((3, 3)))
    held = np.bincount(patches.ravel(), weights=count.ravel(), minlength=n_patches + 1)
    on &= held[patches] >= max(MIN_PATCH_CELLS, MIN_PATCH * count.sum())

    ramp = max(2.0, RAMP * sigma / bin_w)
    for r in range(n_rows):
        for a, e in _runs(on[r]):
            if e - a < 2:
                on[r, a:e] = False
                continue
            i = np.arange(e - a, dtype=np.float64)
            t = np.clip(np.minimum(i, e - a - 1 - i) / ramp, 0.0, 1.0)
            value[r, a:e] *= t * t * (3 - 2 * t)
    p.update(on=on, value=value, kinds=kinds, x0=x0, n_rows=n_rows, n_bins=n_bins,
             drawn=rows[on[s, b]])


def _runs(mask):
    """(start, end) of each run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _shortest_bins(longer, step):
    """How many bins the shortest stretch of one colour runs: the same on the figure at any number
    of strips."""
    return max(2, int(math.ceil(longer * MIN_STRETCH / step)))


def _stretches(kinds, shortest):
    """[start, end, kind] of each stretch of one cell type along a run of bins. A stretch shorter
    than `shortest` bins goes to the longer of its neighbours, so that every colour can be read."""
    edges = [0] + (np.flatnonzero(np.diff(kinds)) + 1).tolist() + [kinds.size]
    parts = [[edges[i], edges[i + 1], int(kinds[edges[i]])] for i in range(len(edges) - 1)]
    while len(parts) > 1:
        i = min(range(len(parts)), key=lambda j: parts[j][1] - parts[j][0])
        if parts[i][1] - parts[i][0] >= shortest:
            break
        if i == 0 or (i < len(parts) - 1 and parts[i + 1][1] - parts[i + 1][0]
                      > parts[i - 1][1] - parts[i - 1][0]):
            parts[i + 1][0] = parts[i][0]
            del parts[i]
        else:
            parts[i - 1][1] = parts[i][1]
            del parts[i]
            i -= 1
        for j in (i, i - 1):            # the stretches either side may now be one type
            if 0 <= j < len(parts) - 1 and parts[j][2] == parts[j + 1][2]:
                parts[j][1] = parts[j + 1][1]
                del parts[j + 1]
    return parts


def _luminance(rgb):
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _visible(colour, floor=0.1):
    """The colour itself when a line of it shows on black, at a contrast of 3:1 or more; otherwise
    lightened just enough, keeping its hue."""
    rgb = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    if _luminance(rgb) >= floor:
        return colour.lower()
    h, light, s = colorsys.rgb_to_hls(*rgb)
    lo, hi = light, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if _luminance(colorsys.hls_to_rgb(h, mid, s)) < floor:
            lo = mid
        else:
            hi = mid
    return "#%02x%02x%02x" % tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, hi, s))


def _spare_colour(i):
    """A colour no other level has, for a dataset with no colours of its own."""
    if i < len(PALETTE):
        return PALETTE[i]
    rgb = colorsys.hls_to_rgb((i * 0.618034) % 1.0, (0.58, 0.7, 0.5)[i % 3], 0.75)
    return "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)


def _palette(levels, known):
    """Each cell type's colour, the map's as long as it shows on black, and which were lightened."""
    out, lightened = [], []
    for i, level in enumerate(levels):
        own = re.fullmatch(r"#[0-9a-fA-F]{6}", str(known.get(level) or ""))
        out.append(_visible(own.group() if own else _spare_colour(i)))
        if own and out[-1] != own.group().lower():
            lightened.append(i)
    return out, lightened


def _over_black(colour):
    return "#%02x%02x%02x" % tuple(round(int(colour[i:i + 2], 16) * BODY) for i in (1, 3, 5))


def _num(tenths):
    """A length given in tenths of a px, in as few characters as SVG allows."""
    t = f"{tenths / 10:.1f}"
    if t.endswith(".0"):
        return t[:-2] if t != "-0.0" else "0"
    return t.replace("0.", ".", 1) if t.startswith(("0.", "-0.")) else t


def _line(xs, ys):
    """SVG path data through the points, in relative steps of 0.1 px."""
    X = np.round(np.asarray(xs) * 10).astype(np.int64)
    Y = np.round(np.asarray(ys) * 10).astype(np.int64)
    steps = zip(np.diff(X).tolist(), np.diff(Y).tolist())
    path = " ".join(f"{_num(a)} {_num(b)}" for a, b in steps).replace(" -", "-")
    return f"M{_num(int(X[0]))} {_num(int(Y[0]))}l{path}"


def _figure(p, scene, longer):
    """One section's ridges as one SVG document, then per cell type (the last entry is unlabelled)
    how long its stretches run and the height over them, both in px."""
    levels, palette = scene["levels"], scene["colours"] + [UNLABELLED]
    spacing, step, scale = scene["spacing"], scene["step"], scene["scale"]
    on, kinds = p["on"], p["kinds"]
    heights = p["value"] * scene["per_value"] * on
    n_rows, n_bins = p["n_rows"], p["n_bins"]
    shortest = _shortest_bins(longer, step)

    xs = (p["x0"] + (np.arange(n_bins) + 0.5) * step / scale) * scale
    used = np.flatnonzero(on.any(axis=0))
    xs = xs - xs[used[0]] + PAD
    map_w = float(xs[used[-1]] - xs[used[0]])
    rows = np.flatnonzero(on.any(axis=1))
    bases = (np.arange(n_rows) - rows[0] + 0.5) * spacing
    # how far the peaks rise over the first baseline
    above = max(float((heights[rows].max(axis=1) - bases[rows]).max()), 0.0)
    map_h = float(bases[rows[-1]]) + above

    length, height = np.zeros(len(palette)), np.zeros(len(palette))
    body, washes = [], []
    width = float(np.clip(LINE * spacing, LINE_MIN, LINE_MAX))
    halo = min(HALO, 0.3 * spacing)
    for r in rows.tolist():             # the back of the figure first: nearer ridges hide it
        base = bases[r]
        for a, e in _runs(on[r]):
            x, ys = xs[a:e], base - heights[r, a:e]
            k = kinds[r, a:e].copy()
            k[k < 0] = len(levels)
            stretches = _stretches(k, shortest)
            base10 = int(round(base * 10))
            outline = (f"M{_num(int(round(x[0] * 10)))} {_num(base10)}L{_line(x, ys)[1:]}"
                       f"V{_num(base10)}Z")
            if len(stretches) == 1:
                fill = _over_black(palette[stretches[0][2]])
            else:
                span = float(x[-1] - x[0])
                stops = "".join(f'<stop offset="{(x[(s + t - 1) // 2] - x[0]) / span:.3f}" '
                                f'stop-color="{_over_black(palette[kind])}"/>'
                                for s, t, kind in stretches)
                fill = f"url(#w{len(washes)})"
                washes.append(f'<linearGradient id="w{len(washes)}">{stops}</linearGradient>')
            body.append(f'<path fill="{fill}" stroke="{GROUND}" stroke-width="{halo:.2f}" '
                        f'd="{outline}"/><path fill="url(#fade)" stroke="none" d="{outline}"/>')
            for s, t, kind in stretches:
                last = min(t, e - a - 1)        # a stretch ends where the next one starts
                run_px = float(x[last] - x[s])
                length[kind] += run_px
                height[kind] += float(p["value"][r, a + s:a + t].mean()) * run_px
                body.append(f'<path stroke="{palette[kind]}" '
                            f'd="{_line(x[s:last + 1], ys[s:last + 1])}"/>')

    shown = [i for i in np.argsort(-length) if length[i] > 0]
    key, key_w, key_h = _key(shown, map_h, scene, p["name"])
    inner_h = max(map_h, key_h)
    drop = PAD + above + (inner_h - map_h) / 2
    canvas_w = int(math.ceil(PAD + map_w + KEY_GAP + key_w + PAD))
    canvas_h = int(math.ceil(PAD + inner_h + PAD))
    what = scene["gene"] or "cell density"
    title = html.escape(f"Ridge map of {what}" + (f", {p['name']}" if p["name"] else ""))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" '
           f'viewBox="0 0 {canvas_w} {canvas_h}" font-family="{FONT}"><title>{title}</title>',
           f'<defs><linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" '
           f'stop-color="{GROUND}" stop-opacity="0"/><stop offset="1" stop-color="{GROUND}"/>'
           f'</linearGradient>{"".join(washes)}</defs>',
           f'<rect width="{canvas_w}" height="{canvas_h}" fill="{GROUND}"/>',
           f'<g transform="translate(0 {drop:.1f})" fill="none" stroke-width="{width:.2f}" '
           f'stroke-linejoin="round" stroke-linecap="round">{"".join(body)}</g>',
           f'<g transform="translate({PAD + map_w + KEY_GAP:.1f} {PAD})">{key}</g></svg>']
    return "".join(out), length, height


def _text_px(text, size):
    """About how wide a label is set, erring wide: an SVG cannot measure text as it is written."""
    return size * (0.58 * len(text) + 0.4 * sum(c.isupper() or c in "mw@%" for c in text))


def _key(shown, map_h, scene, section):
    """The figure's furniture, in one column beside the map: what the height means and its scale,
    the cell types, the length scale. Returns the SVG, its width and its height."""
    levels, palette = scene["levels"], scene["colours"] + [UNLABELLED]
    esc = html.escape
    gene = scene["gene"]
    unit, factor = _height_unit(gene, scene["units"])
    style = ' font-style="italic"' if gene else ""
    head = gene or "Cell density"
    height_is = f"height: {unit}" + (", log scale" if gene else "")
    colour_is = "colour: most common cell type here"
    notes = ("Height is lowered at a ridge's ends.", "A peak stands behind its baseline.")
    out = [f'<text y="16" font-size="20" fill="{INK}"{style}>{esc(head)}</text>']
    wide = [_text_px(head, 20), _text_px(height_is, 12), _text_px(colour_is, 12)]
    wide += [_text_px(n, 11) for n in notes]
    y = 35.0
    if section:
        out.append(f'<text y="{y:.1f}" font-size="12.5" fill="{INK}">{esc(section)}</text>')
        wide.append(_text_px(section, 12.5))
        y += 17
    out.append(f'<text y="{y:.1f}" font-size="12" fill="{MUTED}">{esc(height_is)}</text>')
    y += 14

    # a round value at its true height
    peak = scene["top"] * factor
    nice = _round_down(peak)
    bar = nice / peak * scene["top"] * scene["per_value"]
    out.append(f'<path d="M4 {y:.1f}v{bar:.1f}M0 {y:.1f}h8M0 {y + bar:.1f}h8" stroke="{INK}" '
               'fill="none"/>')
    if bar >= 13:
        out.append(f'<text x="14" y="{y + 4:.1f}" font-size="12" fill="{INK}">{_value(nice)}</text>'
                   f'<text x="14" y="{y + bar + 4:.1f}" font-size="12" fill="{INK}">0</text>')
    else:
        out.append(f'<text x="14" y="{y + bar / 2 + 4:.1f}" font-size="12" fill="{INK}">0 to '
                   f'{_value(nice)}</text>')
    y += max(bar, 13) + 30

    out.append(f'<text y="{y:.1f}" font-size="12" fill="{MUTED}">{esc(colour_is)}</text>')
    y += 10
    listed = shown[:MAX_KEY]
    labels = [levels[i] if i < len(levels) else "Unlabelled" for i in listed]
    labels = [s if len(s) <= 30 else s[:29] + "…" for s in labels]
    rest = len(shown) - len(listed)
    n_lines = len(listed) + (1 if rest else 0)
    column = 26 + max(_text_px(s, 12.5) for s in labels) + 16   # swatch, longest name, a gap
    room = max(map_h, MAP_H * 0.75) - y - 86
    per_col = max(1, int(room // KEY_LINE))
    n_cols = int(math.ceil(n_lines / per_col))
    for j, i in enumerate(listed):
        col, line = divmod(j, per_col)
        lx, ly = col * column, y + line * KEY_LINE
        out.append(f'<path d="M{lx:.1f} {ly + 9:.1f}h18" stroke="{palette[i]}" stroke-width="2.4" '
                   f'stroke-linecap="round"/><text x="{lx + 26:.1f}" y="{ly + 13.2:.1f}" '
                   f'font-size="12.5" fill="{INK}">{esc(labels[j])}</text>')
    if rest:
        col, line = divmod(len(listed), per_col)
        out.append(f'<text x="{col * column:.1f}" y="{y + line * KEY_LINE + 13.2:.1f}" '
                   f'font-size="12" fill="{MUTED}">and {rest} more</text>')
    y += min(n_lines, per_col) * KEY_LINE + 22

    # how to read a peak, then the length scale, at the foot of the column
    bar_len = _round_down(SCALE_BAR / scene["scale"])
    bar_px = bar_len * scene["scale"]
    bottom = max(map_h, y + 56)
    for j, note in enumerate(notes):
        out.append(f'<text y="{bottom - 48 + j * 15:.1f}" font-size="11" fill="{MUTED}">{esc(note)}'
                   '</text>')
    out.append(f'<path d="M0 {bottom - 6:.1f}h{bar_px:.1f}" stroke="{INK}" stroke-width="1.6"/>'
               f'<text y="{bottom - 12:.1f}" font-size="12" fill="{INK}">'
               f'{esc(_length(bar_len, scene["units"]))}</text>')
    return "".join(out), max([n_cols * column] + wide), bottom


def _round_down(v):
    """The largest 1, 2 or 5 times a power of ten that is at most v."""
    power = 10.0 ** math.floor(math.log10(v))
    return next((m * power for m in (5, 2) if m * power <= v), power)


def _value(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def _length(v, units):
    if units in ("µm", "um") and v >= 1000:
        return f"{v / 1000:.3g} mm"
    return f"{_value(v)} {units}".strip()


def _height_unit(gene, units):
    """What the height is measured in, and that unit per value of the height."""
    if gene:
        return "mean log1p(CP10K)", 1.0
    if units in ("µm", "um"):
        return "cells per mm²", 1e6
    if units == "mm":
        return "cells per mm²", 1.0
    return (f"cells per 1,000 {units}²" if units else "cells per 1,000 square units"), 1e3
