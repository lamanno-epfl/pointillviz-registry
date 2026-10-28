"""Which way a quantity increases across the tissue, drawn as arrows.

Pick any quantity the dataset can reduce to a number per cell — a gene, one
level of a categorical column, a continuous .obs column — and this reports the
SPATIAL GRADIENT of its smoothed local mean, as a lattice of arrows over the
section, plus the smoothed field itself painted on the cells the arrows
differentiate.

Three things make it a measurement rather than a picture, and all three are in
`vfield_core.py` with the numbers behind them:

  1. THE FIELD IS A LOCAL MEAN, NEVER A DENSITY. Numerator and denominator are
     smoothed with the same Gaussian and divided, so a region with twice the
     cells and the same expression reads the same value.

  2. THE GRADIENT DIFFERENTIATES THE ACCUMULATORS, NOT THE FINISHED FIELD. The
     quotient rule cancels the -F d(ln C)/dx term exactly, so a density edge is
     not reported as an expression gradient and a tissue outline is not
     reported as a ring of maximal arrows. Measured: the naive estimator is
     wrong by 258-314% under a 64x density ramp where this one is wrong by
     0.7-2%.

  3. EVERY ARROW CLEARS A SIGNIFICANCE GATE. Gaussian smoothing manufactures a
     beautiful vector field out of pure noise. On a Xenium colon section a
     permuted gene — all spatial structure destroyed — still produces a field
     that looks entirely like biology, and a gene present in 0.2% of cells
     produces a MORE coherent-looking one than a real, broadly expressed gene.
     No global "does this look coherent" summary separates them. Sites are
     therefore reported only at z >= 3 with at least 30 effective cells under
     the kernel: measured survival is 83-92% for a real gene and 1.1-1.9% for
     both nulls, against the 1.11% the Rayleigh null predicts.

A CATEGORY HAS NO GRADIENT. Only its local abundance does. Choosing a level
therefore differentiates the local FRACTION of cells of that level, and every
label this tool writes says so. Cells with no label at all are excluded from
the field rather than counted as "not this level", because those are different
statements.

WHAT THE ARROWS ARE NOT. They are the gradient of a smoothed mean, which is a
description of the picture, not a flow: nothing here claims cells move that
way, and a gradient is not a direction of transport, differentiation or
signalling. Read them as "uphill from here is that way, this steeply".
"""

import numpy as np

from pointillviz_tool import PluginError

# Grid cell size, as a multiple of the typical spacing between cells, at
# detail = 1. Two is the smallest that reliably puts several cells in every
# occupied cell of the lattice, which is what keeps the local mean a mean
# rather than a scatter of single observations.
CELL_SIZE_IN_SPACINGS = 2.0
# Grid ceiling per axis. 512 x 512 is a quarter-million sites, which is already
# far more than the arrow lattice will ever sample; past it the convolutions
# start to cost real time for detail nobody sees.
MAX_GRID = 512
#: Rayleigh null: P(z >= 3) = exp(-4.5). The rate at which pure noise clears
#: the gate, and therefore the number a survival fraction has to beat.
RAYLEIGH_RATE = float(np.exp(-4.5))


def run(ds, params):
    import vfield_core as VC          # sibling module: importable only at run time

    sigma = float(params["sigma"])
    detail = float(params["detail"])
    want_arrows = int(params["arrows"])
    gate_on = bool(params["gate"])

    alpha, label, categorical = _quantity(ds, params)

    # ---- eligibility: the population the field is computed over ------------
    # The lasso and an isolated section are the two ways a user narrows the
    # question, and both must narrow the FIELD, not just the picture — a
    # gradient computed over cells that are not on screen is a measurement of
    # something else.
    eligible = np.ones(ds.n_cells, dtype=bool)
    scope = []
    if ds.section_filter is not None:
        eligible &= (ds.section == ds.section_filter)
        scope.append(f"section {ds.section_filter}")
    if ds.subset is not None:
        sub = np.zeros(ds.n_cells, dtype=bool)
        sub[ds.subset] = True
        eligible &= sub
        scope.append(f"the {int(sub.sum()):,} cells you selected")

    xy = ds.xy
    usable = eligible & np.isfinite(xy).all(axis=1) & np.isfinite(alpha)
    n_usable = int(usable.sum())
    if n_usable < 200:
        raise PluginError(
            "too_few_cells",
            f"Only {n_usable:,} cells have both a position and a value for "
            f"{label}. A smoothed gradient needs a few hundred at the very "
            "least; widen the selection, or pick a quantity more cells carry.")

    ds.progress(0.15, "sizing the grid")
    bbox = (float(xy[usable, 0].min()), float(xy[usable, 1].min()),
            float(xy[usable, 0].max()), float(xy[usable, 1].max()))
    spacing = VC.spacing(xy[usable, 0], xy[usable, 1], bbox)
    cell_size = spacing * CELL_SIZE_IN_SPACINGS / detail
    grid = VC.make_grid(bbox[0], bbox[1], bbox[2], bbox[3], cell_size, MAX_GRID)
    ds.log(f"grid {grid.GX} x {grid.GY}, cell {grid.cell_size:.3g} "
           f"(cell spacing {spacing:.3g})" + ("  [clamped]" if grid.clamped else ""))

    ds.progress(0.35, "binning and smoothing")
    cell_of = VC.assign(xy[:, 0], xy[:, 1], grid)
    acc = VC.accumulate(cell_of, alpha, grid, eligible=eligible)

    ds.progress(0.6, "differentiating")
    field = VC.build_field(grid, acc, sigma,
                           min_neff=30.0 if gate_on else 0.0,
                           z_min=3.0 if gate_on else 0.0)

    if field.n_gated == 0:
        # Painting nothing under a confident sentence is the failure the whole
        # NaN convention exists to prevent, so this refuses in words instead.
        # The two ways to get here mean different things to a reader and lead
        # to different fixes, so they are never merged into one sentence: a
        # grid too coarse or too eroded to measure ANYWHERE is a settings
        # problem, while a grid that measured thousands of sites and found
        # nothing significant is an answer about the quantity.
        if field.n_valid == 0:
            raise PluginError(
                "no_measurable_sites",
                f"Nothing could be measured. Of {field.n_supported:,} grid "
                "sites inside the tissue, none were far enough from a tissue "
                "edge for the smoothing window to sit entirely on real data, "
                f"so no gradient at all was computed for {label}. The grid is "
                "too coarse, or σ too large, for a region this size: lower the "
                "smoothing, raise the grid detail, or widen the selection.")
        raise PluginError(
            "no_signal",
            f"No site cleared the significance gate. Of {field.n_supported:,} "
            f"grid sites inside the tissue, {field.n_valid:,} were far enough "
            "from an edge to measure, and none of those reached z = 3 with 30 "
            f"effective cells under the kernel. For {label} that is the honest "
            "answer: there is no gradient here to draw. Try a larger σ, a "
            "coarser grid, or a quantity more cells carry — or turn the "
            "significance gate off to look at the raw field, knowing that what "
            "it shows may be noise.")

    ds.progress(0.8, "laying out the arrows")
    stride = VC.stride_for(field.n_gated, want_arrows)
    rows, cols = VC.lattice_sites(grid, field, stride, want_arrows)
    ax, ay = VC.site_centres(grid, rows, cols)
    gx = field.gx[rows, cols]
    gy = field.gy[rows, cols]
    fval = field.F[rows, cols]

    # Arrow components are reported PER DISPLAY UNIT rather than per data unit:
    # a gradient per micrometre on a 20,000 um section is a number with four
    # leading zeros, and a reader cannot compare two of those. The direction is
    # untouched — this is one positive scale factor on both components.
    unit_scale, unit_label = _units(ds)
    u = gx * unit_scale
    v = gy * unit_scale

    ds.progress(0.9, "painting the field")
    cells = np.full(ds.n_cells, np.nan)
    on_grid = cell_of >= 0
    krow = np.zeros(ds.n_cells, dtype=np.int64)
    kcol = np.zeros(ds.n_cells, dtype=np.int64)
    krow[on_grid] = cell_of[on_grid] // grid.GX
    kcol[on_grid] = cell_of[on_grid] % grid.GX
    # Only cells that are IN the population and sit on a supported site: a cell
    # excluded from the field must not be painted with the field's value, or
    # the map would claim a measurement that was never taken there.
    paint = on_grid & eligible & field.mask[krow, kcol]
    cells[paint] = field.F[krow[paint], kcol[paint]]

    quantity = f"local fraction of {label}" if categorical else f"local mean {label}"
    survival = field.n_gated / field.n_valid if field.n_valid else 0.0
    z_curve, obs_curve, null_curve = _survival_curve(field)

    return {
        "summary": _summary(ds, label, quantity, categorical, grid, field, spacing,
                            sigma, stride, rows.size, survival, gate_on, scope,
                            unit_label, n_usable),
        "cells": {
            "values": cells.tolist(),
            "label": quantity,
            "colormap": "viridis",
            "units": "fraction" if categorical else "",
        },
        "vectors": {
            "title": f"Gradient of {quantity}",
            "label": f"∇ {quantity}",
            "x": ax.tolist(), "y": ay.tolist(),
            "u": u.tolist(), "v": v.tolist(),
            # Colour by the FIELD, not by the magnitude: length already carries
            # the magnitude, so colouring by it again spends the only free
            # channel on a duplicate. Colouring by F ties every arrow to the
            # scalar it was computed from, and to the same ramp the cells use.
            "value": fval.tolist(),
            "units": unit_label,
            "pitch": float(stride * grid.cell_size),
            "colormap": "viridis",
        },
        "series": {
            "title": "How much of this field could be noise",
            "x_label": "significance threshold z",
            "y_label": "fraction of measurable sites above z",
            "log_x": False,
            "lines": [
                {"label": "this field", "x": z_curve.tolist(), "y": obs_curve.tolist()},
                {"label": "expected by chance (Rayleigh)", "x": z_curve.tolist(),
                 "y": null_curve.tolist(), "dashed": True},
            ],
            "rules": [3.0],
        },
    }


# ---------------------------------------------------------------------------
def _quantity(ds, params):
    """(values, label, is_categorical) for the chosen source.

    Returns a full-length array with NaN where the cell has no value — the one
    convention the whole contract rests on. For a categorical level that means
    three states, not two: 1 at the level, 0 at a different level, NaN with no
    label at all.
    """
    source = str(params["source"])

    if source == "gene":
        name = params.get("gene")
        if not name:
            raise PluginError("no_gene",
                              "Choose a gene. 'Differentiate' is set to a gene "
                              "and the gene box is empty.")
        return np.asarray(ds.gene(str(name), "cp10k"), dtype=np.float64), \
            f"{name} (cp10k)", False

    if source == "continuous":
        name = params.get("column")
        if not name:
            available = ", ".join(c["name"] for c in ds.continuous_fields()) or "none"
            raise PluginError(
                "no_column",
                "Choose a continuous column. 'Differentiate' is set to a "
                f"continuous column and none is selected. This dataset has: {available}.")
        return np.asarray(ds.continuous(str(name)), dtype=np.float64), str(name), False

    field = params.get("field")
    level = params.get("level")
    if not level:
        if not ds.categorical_fields():
            raise PluginError(
                "no_categoricals",
                "This dataset has no categorical columns, so there is no level "
                "to differentiate. Set 'Differentiate' to a gene or to a "
                "continuous column.")
        raise PluginError(
            "no_level",
            "Choose a level. 'Differentiate' is set to a category level, and "
            "one level per run is the whole design: a category has no gradient, "
            "only the local fraction of ONE of its levels does.")
    levels, codes = ds.categorical(str(field) if field else None)
    if str(level) not in levels:
        raise PluginError("unknown_level",
                          f"{level!r} is not a level of {field!r}. "
                          f"Have: {', '.join(map(str, levels[:12]))}"
                          + (" ..." if len(levels) > 12 else "") + ".")
    want = levels.index(str(level))
    out = np.full(ds.n_cells, np.nan)
    labelled = codes >= 0
    out[labelled] = (codes[labelled] == want).astype(np.float64)
    n_at = int((codes == want).sum())
    if n_at == 0:
        raise PluginError("empty_level",
                          f"No cell in this selection is {level!r}, so its local "
                          "fraction is zero everywhere and has no gradient.")
    return out, str(level), True


def _units(ds):
    """(scale, label) turning a per-data-unit gradient into a readable one.

    A pure positive rescaling of both components, so it moves no arrow: it only
    decides whether the reader is shown 0.00042 or 4.2e-2.
    """
    u = ((ds.meta.get("spatial") or {}).get("units")) or "px"
    if u == "px":
        return 1000.0, "per 1000 px"
    return 100.0, f"per 100 {u}"


def _survival_curve(field):
    """P(z >= t) over the measurable sites, against the Rayleigh null.

    The honesty plot. Under the null the two gradient components are
    independent N(0, SE), so |grad|/SE is Rayleigh and P(z >= t) = exp(-t^2/2)
    exactly — a curve with no fitted parameter. A field whose observed curve
    lies on the dashed one is noise, however convincing the arrows look, and
    that is a thing a reader can see rather than take on trust.
    """
    # The MEASURABLE set, not the gated one: the curve has to be computable at
    # every threshold, including thresholds the gate would have rejected, or it
    # could only ever show its own gate.
    mask = field.mask.copy()
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = True
    nbr = np.zeros_like(mask)
    nbr[1:-1, 1:-1] = (mask[1:-1, :-2] & mask[1:-1, 2:] &
                       mask[:-2, 1:-1] & mask[2:, 1:-1])
    measurable = inner & mask & nbr & (field.fill >= field.min_fill)
    z = field.z[measurable]
    t = np.linspace(0.0, 6.0, 121)
    if z.size == 0:
        return t, np.full(t.shape, np.nan), np.exp(-t * t / 2.0)
    obs = (z[None, :] >= t[:, None]).mean(axis=1)
    return t, obs, np.exp(-t * t / 2.0)


def _summary(ds, label, quantity, categorical, grid, field, spacing, sigma,
             stride, drawn, survival, gate_on, scope, unit_label, n_usable):
    u = ((ds.meta.get("spatial") or {}).get("units")) or "px"
    sigma_p = field.sigma_p
    lines = []

    lines.append(
        f"**{drawn:,} arrows**, one every {stride} grid cell"
        f"{'' if stride == 1 else 's'}, showing the gradient of the "
        f"{quantity}. Length is the rate of change; colour is the underlying "
        f"value; the units on the arrows are {unit_label}.")

    if categorical:
        lines.append(
            f"A category has no gradient — only its abundance does — so this is "
            f"the gradient of the local FRACTION of cells that are *{label}*, "
            "not of *{0}* itself. Cells with no label at all were left out of "
            "the field rather than counted as 'not {0}'.".format(label))

    lines.append(
        f"Smoothing σ = {sigma:g} grid cells = **{sigma_p:.1f} {u}** on the "
        f"tissue, over a {grid.GX} × {grid.GY} grid of {grid.cell_size:.1f} {u} "
        f"cells (typical spacing between cells: {spacing:.1f} {u})."
        + ("  The grid hit its 512-cell ceiling, so the cells are coarser than "
           "the detail setting asked for." if grid.clamped else ""))

    pct = 100.0 * survival
    lines.append(
        f"Of {field.n_supported:,} grid sites inside the tissue, "
        f"{field.n_valid:,} were far enough from an edge to measure "
        f"({field.n_edge:,} were not), and **{field.n_gated:,} of those "
        f"({pct:.1f}%) are significant**.")

    if not gate_on:
        lines.append(
            "**The significance gate is OFF, so this map shows noise as signal.** "
            "Gaussian smoothing manufactures a convincing vector field out of "
            "pure noise: on a real section a permuted gene, with every trace of "
            "spatial structure destroyed, still produces arrows that look "
            "exactly like biology. Nothing below is evidence of anything until "
            "the gate is back on.")
    elif pct <= 100 * RAYLEIGH_RATE * 3:
        lines.append(
            f"**Treat this field as noise.** Pure noise clears this gate at "
            f"{100 * RAYLEIGH_RATE:.2f}% by construction, and {pct:.1f}% is not "
            "meaningfully above that. The arrows that survived are the ones "
            "chance was always going to leave behind. For comparison, a broadly "
            "expressed real gene measured on a 135k-cell section survives at "
            "83–92%.")
    else:
        lines.append(
            f"Pure noise clears this gate at {100 * RAYLEIGH_RATE:.2f}% by "
            f"construction, so {pct:.1f}% is {pct / (100 * RAYLEIGH_RATE):.0f}× "
            "the chance rate. The chart shows the whole survival curve against "
            "that null: a field lying on the dashed line is noise however "
            "convincing its arrows look.")

    if scope:
        lines.append(f"Computed over {' in '.join(scope)} — {n_usable:,} cells "
                     "with both a position and a value.")

    lines.append(
        "These arrows describe the picture, not a flow. A gradient says which "
        "way the smoothed quantity increases and how steeply; it is not a "
        "direction of movement, differentiation or signalling.")

    return "\n\n".join(lines)
