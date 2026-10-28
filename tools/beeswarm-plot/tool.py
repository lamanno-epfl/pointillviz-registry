"""Beeswarm plot: how each gene is expressed across the groups of an annotation.

One row per group (cell types, niches, any categorical column) and one column per gene. Every glyph
is the swarm of the group's cells along the gene's expression, log1p(CP10K), low at its left end and
high at its right: its height at each level follows how many of the group's cells are there, and its
size the share of the group above the threshold (the square root, as the Marker Dot Plot sizes its
dots), so a gene that few cells of a group detect draws a small glyph. Its colour is the group's mean
expression as a z-score across the groups, in one of the viewer's gene colormaps.

The threshold leaves cells at or below it out of the swarms (0 keeps every cell that detects the
gene). The colour always counts every cell of the group, so the threshold changes what is drawn,
never what a colour means.
"""

import html

import numpy as np

from pointillviz_tool import PluginError, download_file, image

N_PRESET = 20           # genes when none are chosen: plugin.json's preset_n
MIN_CELLS = 10          # cells a group needs for a row
MAX_ROWS = 120          # rows at most: the largest groups
Z_LIM = 2.5             # the colour range, ± z
BINS = 24               # expression levels along a glyph

# The figure, in px.
PAD, ROW_H, COL_W, INSET = 18, 15, 46, 4
STRIP_W, STRIP_GAP, GRID_GAP = 9, 2, 6
HALF = ROW_H * 0.46     # a glyph's largest half-height
SMALLEST = 0.14         # ... and its smallest, as a share of HALF: a few cells still show
GROUND, INK, MUTED, RULE, TICK = "#000000", "#F2F2F2", "#8C8C96", "#24242C", "#3C3C46"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# The viewer's gene colormaps (frontend/js/colors.js, GENE_CMAP_STOPS): 64 anchors of canonical
# Matplotlib each, interpolated into a 256-entry table the way the viewer does, so a colour here is
# the colour the viewer gives the same value.
CMAPS = {
    "viridis": (68,1,84,70,7,90,71,13,96,71,19,101,72,24,106,72,29,111,72,35,116,72,40,120,71,45,123,70,50,126,69,55,129,68,59,132,66,64,134,64,69,136,62,73,137,61,78,138,58,83,139,56,88,140,54,92,141,52,96,141,50,100,142,49,104,142,47,108,142,45,112,142,44,115,142,42,119,142,41,123,142,39,127,142,38,130,142,36,134,142,35,138,141,33,142,141,32,146,140,31,150,139,31,154,138,31,158,137,31,161,135,33,165,133,35,169,131,38,173,129,42,176,127,47,180,124,53,183,121,59,187,117,66,190,113,74,193,109,82,197,105,90,200,100,101,203,94,110,206,88,119,209,83,129,211,77,139,214,70,149,216,64,160,218,57,170,220,50,181,222,43,192,223,37,202,225,31,213,226,26,223,227,24,234,229,26,244,230,30,253,231,37),
    "plasma": (13,8,135,25,6,140,34,6,144,42,5,147,49,5,151,56,4,154,63,4,156,70,3,159,76,2,161,83,2,163,89,1,165,96,1,166,102,0,167,108,0,168,114,1,168,120,1,168,128,4,168,134,6,166,139,10,165,145,14,163,150,19,161,156,23,158,161,27,155,166,32,152,171,36,148,176,41,145,180,46,141,184,50,137,189,55,134,193,59,130,197,64,126,201,68,122,205,74,118,209,78,114,213,83,111,216,87,107,219,92,104,222,97,100,226,101,97,229,106,93,231,111,90,234,116,87,237,121,83,239,126,80,241,131,76,244,136,73,246,141,69,247,147,66,249,154,62,251,159,58,252,165,55,253,171,51,253,177,48,254,183,45,254,189,42,253,195,40,253,202,38,252,208,37,251,215,36,249,221,37,247,228,37,245,235,39,242,242,39,240,249,33),
    "inferno": (0,0,4,2,1,10,4,3,18,7,5,27,11,7,36,16,9,45,21,11,55,27,12,65,33,12,74,40,11,83,47,10,91,54,9,97,61,9,101,68,10,104,74,12,107,81,14,108,89,16,110,95,19,110,101,21,110,108,24,110,114,26,110,120,28,109,127,30,108,133,33,107,140,35,105,146,37,104,152,39,102,159,42,99,165,44,96,171,47,94,177,50,90,183,53,87,191,57,82,196,60,78,202,64,74,207,68,70,212,72,66,217,77,61,222,82,56,226,87,52,230,93,47,234,99,42,237,105,37,240,111,32,243,118,27,245,125,21,247,132,16,249,139,11,250,148,7,251,155,6,252,163,9,252,170,15,252,178,22,251,186,31,250,194,40,249,201,50,247,209,61,245,217,73,244,225,86,242,232,101,241,239,117,243,245,134,246,250,150,252,255,164),
    "magma": (0,0,4,2,1,9,3,3,18,6,5,26,10,8,34,14,11,43,19,13,52,24,15,61,29,17,71,34,17,80,41,17,90,47,17,99,54,16,107,61,15,113,68,15,118,74,16,121,82,19,124,89,21,126,95,24,127,101,26,128,107,29,129,114,31,129,120,34,129,126,36,130,132,38,129,139,41,129,145,43,129,152,45,128,158,47,127,165,49,126,171,51,124,178,53,123,186,56,120,192,58,118,199,61,115,205,64,113,211,67,110,217,70,107,223,74,104,228,79,100,233,84,98,237,90,95,241,96,93,244,103,92,246,110,92,248,118,92,250,125,94,251,133,96,252,142,100,253,150,104,254,157,108,254,165,113,254,172,118,254,180,123,254,187,129,254,194,135,254,202,141,254,209,148,254,216,154,253,224,161,253,231,169,252,238,176,252,246,184,252,253,191),
    "cividis": (0,34,78,0,37,84,0,40,91,0,43,98,0,46,106,0,48,112,5,51,113,18,53,112,26,56,111,33,59,110,39,62,110,45,65,109,50,67,109,54,70,108,59,73,108,63,76,108,68,79,108,72,82,108,76,85,108,80,87,108,84,90,109,87,93,109,91,96,110,94,99,111,98,101,111,101,104,112,105,107,113,108,110,114,112,113,115,115,116,117,119,119,118,122,122,120,126,125,120,130,128,121,134,131,121,138,134,120,142,137,120,146,140,120,149,143,119,153,146,119,157,149,118,161,153,117,165,156,116,169,159,115,173,162,114,177,165,112,182,169,111,186,172,109,191,176,107,195,179,105,199,183,103,204,186,100,208,190,98,212,193,95,217,197,92,221,200,88,225,204,85,230,208,81,234,211,76,239,215,72,243,219,66,248,223,60,253,227,52,254,232,56),
}


def _lut(name):
    stops = np.asarray(CMAPS.get(name) or CMAPS["magma"], dtype=float).reshape(-1, 3)
    f = np.linspace(0, 1, 256) * (len(stops) - 1)
    a = np.floor(f).astype(int)
    b = np.minimum(a + 1, len(stops) - 1)
    m = (f - a)[:, None]
    return np.round(stops[a] * (1 - m) + stops[b] * m).astype(int)


def _hex(lut, z):
    i = int(round((min(max(z, -Z_LIM), Z_LIM) + Z_LIM) / (2 * Z_LIM) * 255))
    return "#%02x%02x%02x" % tuple(lut[i])


def _column(ds, name):
    """(label, levels, codes, colours) of a categorical column; None names the cell types."""
    levels, codes = ds.categorical(name)          # refuses an unknown column in a sentence
    entries = ds.categorical_fields()
    e = entries[0] if name is None else next(
        (e for e in entries if name in (e["name"], e["label"])), entries[0])
    return e["label"], list(levels), codes, dict(e["colors"] or {})


def run(ds, params):
    m = _measure(ds, params)
    ds.progress(0.92, "drawing")
    svg = _figure(m, _lut(m["cmap"]))
    label, levels, rows, genes, thr = m["label"], m["levels"], m["rows"], m["genes"], m["thr"]
    size, means, z, share = m["size"], m["means"], m["z"], m["share"]

    table = {"title": "The numbers behind the plot",
             "columns": ["Group", "Gene", "Cells", "Mean log1p(CP10K)", "z-score",
                         f"% above {thr:g}"],
             "rows": [[levels[i], g, int(size[r]), round(float(means[r, j]), 4),
                       round(float(z[r, j]), 3), round(100 * float(share[r, j]), 2)]
                      for r, i in enumerate(rows) for j, g in enumerate(genes)]}

    names = {"viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno", "magma": "Magma",
             "cividis": "Cividis"}
    summary = (
        f"{len(genes)} genes across the {len(rows)} groups of {label} with at least {MIN_CELLS} "
        f"cells ({int(size.sum()):,} cells). "
        + ("Genes: the 20 detected in the most cells (Top markers). " if m["preset"] else "")
        + "A swarm holds the cells of a group above "
        f"{thr:g} log1p(CP10K), low expression at its left end and high at its right; its size "
        "follows the share of the group's cells there. Its colour is the group's mean log1p(CP10K) "
        f"as a z-score across these groups ({names.get(m['cmap'], m['cmap'])}, ±{Z_LIM:g}).")
    if m["notes"]:
        summary += "\n\n" + " ".join(m["notes"])
    return {"summary": summary,
            "images": [image(svg, "Beeswarm plot")],
            "table": table,
            "files": [download_file("beeswarm-plot.svg", svg.encode("utf-8"),
                                    media_type="image/svg+xml")]}


def _measure(ds, params):
    """Everything the plot draws: the rows, the genes, and per group and gene the mean, its
    z-score, the share above the threshold and the swarm's cells across the levels."""
    thr = float(params.get("threshold") or 0.0)
    notes = []

    # ---- the rows ---------------------------------------------------------
    label, levels, codes, colours = _column(ds, params.get("field") or None)
    counts = np.bincount(codes[codes >= 0], minlength=len(levels))
    rows = [i for i in range(len(levels)) if counts[i] >= MIN_CELLS]
    small = len([i for i in range(len(levels)) if 0 < counts[i] < MIN_CELLS])
    if small:
        notes.append(f"{small} group{'s' if small > 1 else ''} with fewer than {MIN_CELLS} cells "
                     "left out.")
    if not rows:
        raise PluginError("too_few_cells",
                          f"No group of {label} has {MIN_CELLS} cells here.")
    if len(rows) > MAX_ROWS:
        notes.append(f"{len(rows) - MAX_ROWS} of the smallest groups left out: the plot draws the "
                     f"{MAX_ROWS} largest.")
        rows = sorted(sorted(rows, key=lambda i: -counts[i])[:MAX_ROWS])

    coarse = None
    if params.get("coarse"):
        c_label, c_levels, c_codes, c_colours = _column(ds, params["coarse"])
        if c_label != label:              # the same column as the groups: no outer strip
            major = {}                    # each group's commonest coarse group orders the rows
            for i in rows:
                c = c_codes[(codes == i) & (c_codes >= 0)]
                major[i] = int(np.bincount(c, minlength=len(c_levels)).argmax()) if c.size else -1
            rows.sort(key=lambda i: (major[i] < 0, major[i], i))
            coarse = (c_label, c_levels, c_colours, major)

    # ---- the genes --------------------------------------------------------
    asked = list(dict.fromkeys(g for g in (params.get("genes") or []) if g))
    genes = [g for g in asked if ds.has_gene(g)]
    missing = [g for g in asked if not ds.has_gene(g)]
    if asked and not genes:
        raise PluginError("unknown_gene", "None of the chosen genes is in this dataset's panel: "
                          + ", ".join(missing) + ".")
    if missing:
        notes.append("Not in this panel, so not drawn: " + ", ".join(missing) + ".")
    preset = not asked
    if preset:
        genes = ds.top_detected_genes(N_PRESET)
    if not genes:
        raise PluginError("no_genes", "This dataset has no genes to draw.")

    # ---- per group and gene: the mean, the share above the threshold, the swarm ----
    row_of = np.full(len(levels), -1)
    row_of[rows] = np.arange(len(rows))
    cell_row = np.where(codes >= 0, row_of[np.maximum(codes, 0)], -1)
    keep = cell_row >= 0
    rr = cell_row[keep]
    n_rows, n_genes = len(rows), len(genes)
    size = np.bincount(rr, minlength=n_rows).astype(float)
    means = np.zeros((n_rows, n_genes))
    share = np.zeros((n_rows, n_genes))
    hist = np.zeros((n_rows, n_genes, BINS))
    for j, g in enumerate(genes):
        ds.progress(0.05 + 0.85 * j / n_genes, f"reading {g}")
        v = ds.gene(g, "log1p")[keep]
        means[:, j] = np.bincount(rr, weights=v, minlength=n_rows) / size
        above = v > thr
        share[:, j] = np.bincount(rr[above], minlength=n_rows) / size
        va = v[above]
        if va.size:
            hi = thr + max(float(np.percentile(va, 99.5)) - thr, 1e-3)
            b = np.clip(((va - thr) / (hi - thr) * BINS).astype(int), 0, BINS - 1)
            hist[:, j] = np.bincount(rr[above] * BINS + b,
                                     minlength=n_rows * BINS).reshape(n_rows, BINS)
    sd = means.std(axis=0)
    z = np.where(sd > 1e-12, (means - means.mean(axis=0)) / np.where(sd > 1e-12, sd, 1), 0.0)

    return {"label": label, "levels": levels, "colours": colours, "coarse": coarse, "rows": rows,
            "genes": genes, "preset": preset, "thr": thr, "notes": notes,
            "cmap": str(params.get("cmap") or "magma"),
            "size": size, "means": means, "z": z, "share": share, "hist": hist}


def _heights(share, hist, size, half=HALF):
    """Every swarm's half-height at each level, in px: (groups, genes, BINS).

    Its shape is the group's cells across the levels, lightly smoothed. Its tallest point is the
    square root of the share of the group above the threshold, against the 95th percentile of the
    shares, and at least SMALLEST of the full height, so that a few cells still show."""
    frac = hist / size[:, None, None]
    sm = np.zeros_like(frac)
    sm[..., 1:-1] = (frac[..., :-2] + 2 * frac[..., 1:-1] + frac[..., 2:]) / 4
    sm[..., 0] = (2 * frac[..., 0] + frac[..., 1]) / 4
    sm[..., -1] = (frac[..., -2] + 2 * frac[..., -1]) / 4
    present = share[share > 0]
    ref = float(np.percentile(present, 95)) if present.size else 1.0
    peak = sm.max(axis=2, keepdims=True)
    top = half * np.clip(np.sqrt(share / ref), SMALLEST, 1.0)[..., None]
    shown = (share[..., None] > 0) & (peak > 0)
    return np.where(shown, top * sm / np.where(peak > 0, peak, 1.0), 0.0)


def _glyph(h, x0, yc, bw):
    """One swarm's outline as SVG path data, from its half-heights; None when it is empty."""
    on = np.flatnonzero(h > 0.04)
    if not on.size:
        return None
    b0, b1 = on[0], on[-1]
    xs = x0 + (np.arange(b0, b1 + 1) + 0.5) * bw
    hs = h[b0:b1 + 1]
    return (f"M{xs[0] - bw / 2:.1f},{yc:.1f}"
            + "".join(f"L{x:.1f},{yc - y:.1f}" for x, y in zip(xs, hs))
            + f"L{xs[-1] + bw / 2:.1f},{yc:.1f}"
            + "".join(f"L{x:.1f},{yc + y:.1f}" for x, y in zip(xs[::-1], hs[::-1])) + "Z")


def _figure(m, lut):
    """The plot as one SVG document."""
    label, levels, colours, coarse = m["label"], m["levels"], m["colours"], m["coarse"]
    rows, genes, z = m["rows"], m["genes"], m["z"]
    n_rows, n_genes = len(rows), len(genes)
    esc = html.escape

    # the left column: names, then the strips (outer first)
    names = [levels[i] if len(levels[i]) <= 30 else levels[i][:29] + "…" for i in rows]
    name_w = min(190, 8 + 6.1 * max(len(s) for s in names))
    strips = 2 if coarse else 1
    gx = PAD + name_w + strips * (STRIP_W + STRIP_GAP) + GRID_GAP
    gy = PAD
    gw, gh = n_genes * COL_W, n_rows * ROW_H
    columns = [s if len(s) <= 28 else s[:27] + "…" for s in ([coarse[0]] if coarse else []) + [label]]
    gene_h = 8 + max(6.5 * max(len(g) for g in genes), 6.1 * max(len(s) for s in columns))
    bar_y = gy + gh + gene_h + 26
    width = int(gx + gw + PAD + 10)
    height = int(bar_y + 46 + PAD)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
           f'<rect width="{width}" height="{height}" fill="{GROUND}"/>']

    # rows: rules between them, a tick where each gene's axis starts
    rule = "".join(f"M{gx:.1f},{gy + r * ROW_H:.1f}h{gw}" for r in range(1, n_rows))
    out.append(f'<path d="{rule}" stroke="{RULE}" stroke-width="0.6" fill="none"/>')
    ticks = "".join(f"M{gx + j * COL_W + INSET:.1f},{gy + r * ROW_H + ROW_H / 2 - 3:.1f}v6"
                    for r in range(n_rows) for j in range(n_genes))
    out.append(f'<path d="{ticks}" stroke="{TICK}" stroke-width="0.7" fill="none"/>')

    # the swarms
    heights = _heights(m["share"], m["hist"], m["size"])
    bw = (COL_W - 2 * INSET) / BINS
    for r in range(n_rows):
        yc = gy + r * ROW_H + ROW_H / 2
        for j in range(n_genes):
            d = _glyph(heights[r, j], gx + j * COL_W + INSET, yc, bw)
            if d:
                out.append(f'<path d="{d}" fill="{_hex(lut, z[r, j])}"/>')

    # names and strips
    sx = PAD + name_w
    for r, i in enumerate(rows):
        y = gy + r * ROW_H
        out.append(f'<text x="{sx - 6:.1f}" y="{y + ROW_H / 2 + 3.6:.1f}" font-size="10.5" '
                   f'fill="{INK}" text-anchor="end">{esc(names[r])}</text>')
        x = sx
        if coarse:
            c_levels, c_colours, major = coarse[1], coarse[2], coarse[3]
            m = major[i]
            fill = c_colours.get(c_levels[m], "#555560") if m >= 0 else "#555560"
            out.append(f'<rect x="{x:.1f}" y="{y + 0.5:.1f}" width="{STRIP_W}" '
                       f'height="{ROW_H - 1}" fill="{fill}"/>')
            x += STRIP_W + STRIP_GAP
        out.append(f'<rect x="{x:.1f}" y="{y + 0.5:.1f}" width="{STRIP_W}" height="{ROW_H - 1}" '
                   f'fill="{colours.get(levels[i], "#555560")}"/>')

    # under the grid: the strips' columns, the genes, the colour bar
    x = sx
    under = gy + gh + 8
    for name in columns:
        out.append(f'<text transform="translate({x + STRIP_W / 2 + 3.5:.1f},{under:.1f}) rotate(-90)" '
                   f'font-size="11" fill="{MUTED}" text-anchor="end">{esc(name)}</text>')
        x += STRIP_W + STRIP_GAP
    for j, g in enumerate(genes):
        cx = gx + j * COL_W + COL_W / 2 + 4
        out.append(f'<text transform="translate({cx:.1f},{under:.1f}) rotate(-70)" font-size="12.5" '
                   f'font-style="italic" fill="{INK}" text-anchor="end">{esc(g)}</text>')

    bar_w = min(220.0, max(160.0, gw * 0.5))
    bx = gx + (gw - bar_w) / 2
    stops = "".join(f'<stop offset="{k / 16:.4f}" stop-color="{_hex(lut, -Z_LIM + 2 * Z_LIM * k / 16)}"/>'
                    for k in range(17))
    out.append(f'<defs><linearGradient id="zbar">{stops}</linearGradient></defs>')
    out.append(f'<text x="{bx + bar_w / 2:.1f}" y="{bar_y - 8:.1f}" font-size="12" fill="{INK}" '
               'text-anchor="middle">mean expression z-score</text>')
    out.append(f'<rect x="{bx:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="10" fill="url(#zbar)" '
               f'stroke="{MUTED}" stroke-width="0.6"/>')
    for t in range(-2, 3):
        tx = bx + (t + Z_LIM) / (2 * Z_LIM) * bar_w
        out.append(f'<path d="M{tx:.1f},{bar_y + 10:.1f}v4" stroke="{MUTED}" stroke-width="0.8"/>')
        out.append(f'<text x="{tx:.1f}" y="{bar_y + 26:.1f}" font-size="11" fill="{INK}" '
                   f'text-anchor="middle">{"−" if t < 0 else ""}{abs(t)}</text>')
    out.append("</svg>")
    return "".join(out)
