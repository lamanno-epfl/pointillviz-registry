"""Smoothie: gene modules that share a spatial pattern, from the authors' published package.

Smoothie (Holdener & De Vlaminck, Commun Biol 9, 459, 2026) smooths every gene over the tissue with
a Gaussian, correlates the smoothed genes, links the pairs above a cutoff and clusters that network
into modules with Infomap.

This file computes none of that. It imports `smoothie` (`smoothie-st` on PyPI) and calls the steps
of the authors' pipeline notebooks in their order, with the values they give: sections are the
datasets of their multi-dataset pipeline, and one section is their single-dataset pipeline. What is
here is the way in and the way out: counts from the dataset into AnnData, a width in µm into
coordinates, sentences where the package raises or crashes, and its panels on one sheet.
"""

import csv
import io
import itertools
import math
import os
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")  # plot_modules calls plt.show(), and a tool opens no window

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import smoothie
from scipy import sparse
from scipy.spatial import cKDTree

from pointillviz_tool import PluginError, download_file, image

# The authors' pipeline notebooks, for cell-sized data.
MIN_COUNTS_CELL = 50    # sc.pp.filter_cells(min_counts)
MIN_COUNTS_GENE = 100   # sc.pp.filter_genes(min_counts)
MIN_CELLS_GENE = 10     # sc.pp.filter_genes(min_cells)
TARGET_SUM = 1e3        # CPT, then log1p
MIN_GENES = 2           # "genes within modules of size 2 or more"
SEED = 0                # the default of make_spatial_network, and what they pass to the shuffle
# The low end of their documented 10-100. A location with fewer cells in reach is set to zero, and
# zeros raise every correlation, so a low value keeps them few: 0.15% of a Xenium section at 20 µm.
MIN_SPOTS = 10
MAX_PROCESSES = 10      # the default of run_parallelized_smoothing
BINNED_UM = 20.0        # their table starts "binned" data here: smoothed in place, the rest on a grid
INPLACE_LIMIT_GB = 100  # the package refuses in-place smoothing above this, and advises the grid

# Coordinates with no known size. Cells sit about 9 µm apart: 7.3 to 12.9 µm, median 9, over the
# cell-segmented datasets this was measured on, as the square root of the area per cell.
CELL_SPACING_UM = 9.0
SPACING_K = 15          # the neighbour that measures the area per cell ...
SPACING_SAMPLE = 20000  # ... around at most this many cells of a section

# The figure: display only, nothing here changes a module.
PANEL_IN = 1.0          # inches a side, the size of the authors' panels ...
PANEL_DPI = 300         # ... at their resolution
MAX_DOTS = 50000        # locations drawn in each panel; a panel has fewer pixels than this
DOT_PX = 2.25           # a dot's smallest diameter, in pixels of the panel ...
DOT_NEIGHBOURS = 1.5    # ... and in spacings between locations: a lattice needs 1.41 to leave no gap
FULL_SIZE_MODULES = 100         # past this many, panels shrink, to keep the time and the size ...
MAX_PANELS = 900                # ... as far as this many, at a third of the resolution
OUTPUT_LIMIT = 8 * 1024 * 1024  # images and downloads share this, in bytes on the wire

# The table.
MAX_ROWS = 5000
MEMBERS_CHARS = 200     # the longest text a table cell may hold


def run(ds, params):
    width = float(params["smoothing_width"])
    cutoff = float(params["correlation_cutoff"])
    power = float(params["clustering_power"])

    spacing = _spacing(ds)
    per_micron, scale_note = _coordinate_units(ds, spacing)
    sd = width * per_micron
    sections, read, kept, notes = _sections(ds)
    total_gb = sum(a.n_obs * a.n_vars * 4 for _, a in sections) / 1e9
    grid = spacing / per_micron < BINNED_UM or total_gb >= INPLACE_LIMIT_GB
    processes = min(MAX_PROCESSES, os.cpu_count() or 1)

    smoothed, used, names = [], [], []
    for i, (name, adata) in enumerate(sections):
        ds.progress(0.06 + 0.34 * i / len(sections), "Smoothing each gene over the tissue")
        result = _when_dense_enough(smoothie.run_parallelized_smoothing, grid, adata,
                                    grid_based_or_not=grid, gaussian_sd=sd,
                                    min_spots_under_gaussian=MIN_SPOTS, num_processes=processes)
        if result is None:
            notes.append(f"Section {name} left out: too sparse for this width.")
            continue
        smoothed.append(result)
        used.append(adata)
        names.append(name)
    if not smoothed:
        _refuse_too_sparse(sections, width, per_micron)

    xy, tiled = _laid_out([np.asarray(s.obsm["spatial"]) for s in smoothed])
    if tiled:
        notes.append("These sections share coordinates, so each panel sets them side by side, "
                     "row by row: " + ", ".join(names) + ".")
    X, genes = smoothie.concatenate_smoothed_matrices(smoothed)
    del smoothed
    silent = int((X.max(axis=1) == 0).sum())
    if silent >= 0.01 * X.shape[0]:
        notes.append(f"The package set {silent:,} of the {X.shape[0]:,} smoothed locations to zero, "
                     f"for fewer than {MIN_SPOTS} cells within {3 * width:g} µm, which raises every "
                     "correlation. Widen the smoothing to reduce them.")

    ds.progress(0.42, "Correlating the smoothed genes")
    pearson, _ = smoothie.compute_correlation_matrix(X)
    _refuse_no_link(pearson, cutoff)
    ds.progress(0.48, "Clustering the gene network")
    _, nodes = smoothie.make_spatial_network(
        pearsonR_mat=pearson, gene_names=genes, pcc_cutoff=cutoff, clustering_power=power)
    del pearson
    modules = nodes.groupby("module_label").filter(lambda m: len(m) >= MIN_GENES)
    if modules.empty:
        raise PluginError("no_module",
                          f"The network at a cutoff of {cutoff:g} leaves every gene alone. "
                          "Lower the cutoff, or the clustering power.")
    n_modules = modules["module_label"].nunique()
    n_locations = X.shape[0]
    panels_note = _panels_note(used, names, genes, modules)
    if panels_note:
        notes.append(panels_note)
    shown, dpi = _locations_to_draw(X, genes, xy, min(n_modules, MAX_PANELS))
    del X

    ds.progress(0.52, "Smoothing with the positions shuffled, for the authors' lower bound")
    shuffled = _when_dense_enough(smoothie.compute_shuffled_correlation_percentiles, grid,
                                  used if len(used) > 1 else used[0], grid_based_or_not=grid,
                                  gaussian_sd=sd, min_spots_under_gaussian=MIN_SPOTS,
                                  num_processes=processes, seed=SEED)
    bound = None if shuffled is None else float(shuffled[2])    # p95, p99, p99.9

    ds.progress(0.72, "Mapping each module")
    download = _modules_csv(modules)
    png, drawn = _figure(shown, modules, dpi, len(download))

    if drawn < n_modules:
        notes.append(f"The figure maps the {drawn:,} largest modules of {n_modules:,}.")
    table, listed = _table(modules)
    if listed < n_modules:
        notes.append(f"The table lists the {listed:,} largest modules of {n_modules:,}; "
                     "the download holds them all.")
    if scale_note:
        notes.insert(0, scale_note)
    if kept < read:
        notes.insert(0, f"Quality control left out {read - kept:,} of {read:,} cells, with fewer "
                        f"than {MIN_COUNTS_CELL} transcripts or in a section left out.")
    where = (f"{sum(a.n_obs for a in used):,} cells"
             + (f", smoothed at {n_locations:,} locations" if grid else ""))
    ds.progress(1.0)
    return {
        "summary": _summary(modules, len(genes), where, width, cutoff, power, bound, notes),
        "table": table,
        "images": [image(png, "Each module on the tissue: the sum of its genes, each scaled from "
                              "0 to 1. Grey is low, red high, black highest, within each panel.")],
        "files": [download_file("smoothie-modules.csv", download, media_type="text/csv")],
    }


# ---- the way in ----------------------------------------------------------------------------------
def _spacing(ds):
    """The spacing between cells, in coordinates: the root of the area per cell, as the median
    reach of a cell's SPACING_K nearest neighbours gives it."""
    rng = np.random.default_rng(SEED)
    reach = []
    for part in ds.kdtrees():
        tree = part["tree"]
        if tree.n > SPACING_K:
            around = (tree.data if tree.n <= SPACING_SAMPLE
                      else tree.data[rng.choice(tree.n, SPACING_SAMPLE, replace=False)])
            reach.append(tree.query(around, k=SPACING_K + 1)[0][:, -1])     # self is the first
    if not reach:
        raise PluginError("too_few_cells",
                          f"No section here has {SPACING_K + 1} cells with coordinates, "
                          "which the smoothing needs at the very least.")
    return float(np.median(np.concatenate(reach))) * math.sqrt(math.pi / SPACING_K)


def _coordinate_units(ds, spacing):
    """(coordinate units per µm, a sentence when that is not what the dataset records)."""
    spatial = ds.meta.get("spatial") or {}
    units = spatial.get("units")
    if units in ("µm", "μm", "um", "micron", "microns"):
        return 1.0, None
    if units == "mm":
        return 1e-3, None
    # The rule the app has for the same question: an image layer that covers the cells.
    x, y = (spatial.get("ranges") or {}).get("x"), (spatial.get("ranges") or {}).get("y")
    for layer in (ds.meta.get("images") or {}).get("layers") or []:
        size, box = layer.get("microns_per_pixel"), layer.get("bounds")
        if (units == "px" and size and box and x and y
                and box[0] <= x[0] and x[1] <= box[2] and box[1] <= y[0] and y[1] <= box[3]):
            return 1.0 / float(size), None
    return spacing / CELL_SPACING_UM, (f"These coordinates have no recorded size, so the median "
                                       f"spacing between cells, {spacing:.3g} {units or 'units'}, "
                                       f"was taken as {CELL_SPACING_UM:g} µm.")


def _sections(ds):
    """One AnnData per section, through the authors' quality control and normalisation."""
    keep = np.isfinite(ds.xy).all(axis=1)
    if ds.section_filter is not None:
        keep &= ds.section == ds.section_filter
    if ds.subset is not None and ds.subset.size:
        chosen = np.zeros(ds.n_cells, dtype=bool)
        chosen[ds.subset] = True
        keep &= chosen
    rows = np.flatnonzero(keep)
    if rows.size < MIN_SPOTS:
        raise PluginError("too_few_cells",
                          f"{rows.size:,} cells have coordinates here; the smoothing needs at "
                          f"least {MIN_SPOTS}.")

    names, counts = _read_counts(ds, rows)
    xy, section = ds.xy[rows], ds.section[rows]
    entries, panel = _section_entries(ds), ds.gene_names
    sections, kept, notes = [], 0, []
    for s in np.unique(section):
        here = section == s
        entry = entries.get(int(s), {})
        label = str(entry.get("section_id") or s)
        # A section keeps the genes it measured, as a dataset does in the authors' multi-dataset
        # pipeline: concatenate_smoothed_matrices and their shuffle line the sections up afterwards.
        lacking = {str(panel[g]) for g in entry.get("unmeasured_genes") or ()}
        measured = [j for j, name in enumerate(names) if name not in lacking]
        adata = ad.AnnData(X=counts[here][:, measured] if lacking else counts[here])
        adata.var_names = [names[j] for j in measured]
        adata.uns["unmeasured"] = sorted(lacking)
        adata.obsm["spatial"] = np.array(xy[here], dtype=np.float32)
        sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELL)
        if adata.n_obs:
            sc.pp.filter_genes(adata, min_counts=MIN_COUNTS_GENE)
            sc.pp.filter_genes(adata, min_cells=MIN_CELLS_GENE)
        if adata.n_obs < MIN_SPOTS or not adata.n_vars:
            notes.append(f"Section {label} left out: {adata.n_obs:,} cells and {adata.n_vars:,} "
                         "genes pass quality control.")
            continue
        # On a straight line the package's chunks of the tissue have no width, and it divides by 0.
        if np.ptp(adata.obsm["spatial"], axis=0).min() == 0:
            notes.append(f"Section {label} left out: its cells lie on a straight line.")
            continue
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)
        sections.append((label, adata))
        kept += adata.n_obs
    if not sections:
        raise PluginError("too_few_counts",
                          f"Of these {rows.size:,} cells, too few hold {MIN_COUNTS_CELL} "
                          f"transcripts, or no gene reaches {MIN_COUNTS_GENE} counts, or the cells "
                          "lie on a line. Smoothie's quality control asks for both counts, so "
                          "select more cells.")
    return sections, rows.size, kept, notes


def _section_entries(ds):
    """Each code of ds.section -> its entry in ds.sections, matched as the SDK matches them: by
    `index` on a flat dataset, and by section_id on an atlas bound to a composition, whose codes are
    positions in that composition."""
    composition = ds.meta.get("_composition")
    if composition:
        by_id = {s.get("section_id"): s for s in ds.sections}
        return {i: by_id.get(c["section_id"], {}) for i, c in enumerate(composition)}
    return {int(s.get("index", i)): s for i, s in enumerate(ds.sections)}


def _read_counts(ds, rows):
    """(gene names, cells-by-genes counts as CSR), a gene at a time, as the dataset stores them.

    On the cells of a section that did not measure a gene, an SDK reads NaN or 0, by its version.
    Neither is a count, so neither is stored.
    """
    names = list(dict.fromkeys(str(g) for g in ds.gene_names))
    hits, values, ptr = [], [], [0]
    for j, name in enumerate(names):
        x = ds.gene(name, "raw")[rows]
        hit = np.flatnonzero((x != 0) & ~np.isnan(x))
        hits.append(hit)
        values.append(x[hit])
        ptr.append(ptr[-1] + hit.size)
        if j % 256 == 0:
            ds.progress(0.06 * j / len(names), "Reading the counts")
    return names, sparse.csc_matrix((np.concatenate(values), np.concatenate(hits), np.array(ptr)),
                                    shape=(rows.size, len(names))).tocsr()


# ---- where the package raises or crashes, a sentence -----------------------------------------------
def _when_dense_enough(step, grid, *args, **kwargs):
    """A step of the package, or None when its density check finds this tissue too sparse."""
    try:
        return step(*args, **kwargs)
    except ValueError as exc:
        if "min_spots_under_gaussian" not in str(exc):
            raise
    except (ZeroDivisionError, IndexError):     # a grid with no point near any cell
        if not grid:
            raise
    return None


def _refuse_too_sparse(sections, width, per_micron):
    """Every section failed the package's density check: more than 15% of its locations have fewer
    than MIN_SPOTS cells within three widths. Say which width passes it."""
    needed = 0.0
    for _, adata in sections:
        coords = adata.obsm["spatial"]
        reach = cKDTree(coords).query(coords, k=MIN_SPOTS)[0][:, -1]    # self is the first
        needed = max(needed, float(np.percentile(reach, 85)) / 3.0 / per_micron)
    raise PluginError(
        "too_sparse",
        f"At a smoothing width of {width:g} µm, more than 15% of the tissue has fewer than "
        f"{MIN_SPOTS} cells within {3 * width:g} µm, the reach of the smoothing. "
        f"Widen it to at least {math.ceil(max(needed, width + 1)):,} µm.")


def _refuse_no_link(pearson, cutoff):
    """make_spatial_network dies with an IndexError when no pair passes the cutoff."""
    diagonal = pearson.diagonal().copy()
    np.fill_diagonal(pearson, np.nan)
    measured = np.isfinite(pearson).any()   # at a very wide width there can be a single location
    best = float(np.nanmax(pearson)) if measured else -1.0
    np.fill_diagonal(pearson, diagonal)
    if best <= cutoff:
        why = (f"the highest correlation here is {best:.2f}. Lower the cutoff, or widen the smoothing"
               if measured else "at this width they do not vary across the tissue. Narrow the smoothing")
        raise PluginError("no_link", f"No two of the {pearson.shape[0]:,} genes correlate above "
                                     f"{cutoff:g}: {why}.")


# ---- the way out -----------------------------------------------------------------------------------
def _locations_to_draw(X, genes, xy, n_panels):
    """(an AnnData of the locations each panel draws, the panels' resolution).

    Past FULL_SIZE_MODULES the panels shrink and draw fewer locations, in step, so that the time and
    the bytes stay what a hundred modules take. A location is only thinned out past what a panel can
    show, and each gene's highest stays, so that thinning changes no gene's scale.
    """
    shrink = min(1.0, FULL_SIZE_MODULES / n_panels)
    dpi = int(PANEL_DPI * math.sqrt(shrink))
    dots = int(MAX_DOTS * shrink)
    if X.shape[0] > dots:
        rows = np.random.default_rng(SEED).choice(X.shape[0], dots, replace=False)
        rows = np.union1d(rows, X.argmax(axis=0))
        X, xy = X[rows].copy(), xy[rows]
    shown = ad.AnnData(X=X)
    shown.var_names = genes
    shown.obsm["spatial"] = np.array(xy, dtype=np.float32)
    return shown, dpi


def _panels_note(used, names, genes, modules):
    """What a run over sections of different gene panels has to say, or None.

    The authors' concatenate_smoothed_matrices and their shuffle fill a gene a section did not
    measure with zeros. Genes missing from one section share those zeros, so a module of nothing
    else can form, and the summary is where its reader learns that.
    """
    in_run = set(genes)
    lacking = {name: in_run.intersection(a.uns["unmeasured"]) for name, a in zip(names, used)}
    lacking = {name: missing for name, missing in lacking.items() if missing}
    if not lacking:
        return None
    filled = set().union(*lacking.values())
    if len(lacking) == 1:
        name, missing = next(iter(lacking.items()))
        who = f"Section {name} did not measure {len(missing):,} of these {len(genes):,} genes."
    else:
        who = (f"{len(lacking)} sections did not measure some of these {len(genes):,} genes, "
               f"{len(filled):,} in all.")
    share = modules["name"].isin(filled).groupby(modules["module_label"]).mean()
    mostly = int((share > 0.5).sum())
    verb = "is" if mostly == 1 else "are"
    found = f"{mostly:,} of the {share.size:,} modules {verb} mostly such genes. " if mostly else ""
    return (f"{who} As in the authors' multi-dataset pipeline, a section is smoothed on the genes "
            "it measured and the others count as zero there, in the correlations, the lower bound "
            "and the map, so genes missing from the same section correlate for that alone. "
            f"{found}Isolate a section to see its own modules.")


def _laid_out(parts):
    """(the sections' coordinates for the figure, whether they were moved).

    As they are, which is how the map lays them out, unless sections lie on top of each other, as
    those of an atlas do when a run brings no layout. Then on a grid, for the figure alone.
    """
    boxes = [(p.min(axis=0), p.max(axis=0)) for p in parts]
    overlap = any((lo_a < hi_b).all() and (lo_b < hi_a).all()
                  for (lo_a, hi_a), (lo_b, hi_b) in itertools.combinations(boxes, 2))
    if not overlap:
        return np.concatenate(parts), False
    cell = 1.1 * np.max([hi - lo for lo, hi in boxes], axis=0)
    columns = math.ceil(math.sqrt(len(parts)))
    return np.concatenate([p - lo + cell * (i % columns, i // columns)
                           for i, (p, (lo, _)) in enumerate(zip(parts, boxes))]), True


def _figure(shown, modules, dpi, spent):
    """(the authors' panel of each module, on one PNG sheet; how many modules it maps)."""
    drawn = np.sort(modules["module_label"].unique())[:MAX_PANELS]
    coords = shown.obsm["spatial"]
    nearest = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    span = float(np.ptp(coords, axis=0).max())
    spot = max(DOT_PX * span / (PANEL_IN * dpi), DOT_NEIGHBOURS * nearest)
    with tempfile.TemporaryDirectory(prefix="smoothie-") as folder, warnings.catch_warnings():
        warnings.simplefilter("ignore")     # plt.show() on Agg, and scanpy's advice on sc.pl.spatial
        smoothie.plot_modules(shown, modules[modules["module_label"].isin(drawn)],
                              output_folder=folder, plots_per_row=1, min_genes=MIN_GENES,
                              figsize=(PANEL_IN, PANEL_IN), spot_size=spot, dpi=dpi)
        panels = [plt.imread(os.path.join(folder, f"module_scores_row_{i + 1}.png"))[..., :3]
                  for i in range(drawn.size)]
    sheet = _sheet(panels)
    png = _png(sheet)
    limit = (OUTPUT_LIMIT - spent * 4 // 3 - 4096) * 3 // 4
    while len(png) > limit and min(sheet.shape[:2]) > 1:    # a panel's bytes depend on its tissue
        sheet = sheet[::2, ::2]
        png = _png(sheet)
    return png, drawn.size


def _sheet(panels):
    """The panels on one white sheet, about 4:3, in the order of their numbers."""
    h, w = max(p.shape[0] for p in panels), max(p.shape[1] for p in panels)
    columns = max(1, round(math.sqrt(len(panels) * (h / w) * (4 / 3))))
    n_rows = math.ceil(len(panels) / columns)
    sheet = np.ones((n_rows * h, columns * w, 3), dtype=np.float32)
    for i, panel in enumerate(panels):
        y = i // columns * h + (h - panel.shape[0]) // 2
        x = i % columns * w + (w - panel.shape[1]) // 2
        sheet[y:y + panel.shape[0], x:x + panel.shape[1]] = panel
    return sheet


def _png(sheet):
    buffer = io.BytesIO()
    plt.imsave(buffer, sheet, format="png")
    return buffer.getvalue()


def _modules_csv(modules):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["module", "gene", "degree", "weighted_degree", "rescaled_weighted_degree",
                     "clustering_coeff", "margin_score"])
    for m in modules.itertuples(index=False):
        writer.writerow([m.module_label, m.name, m.degree, f"{m.weighted_degree:.6g}",
                         f"{m.rescaled_weighted_degree:.6g}", f"{m.clustering_coeff:.6g}",
                         f"{m.margin_score:.6g}"])
    return buffer.getvalue()


def _table(modules):
    """One row per module; a table cell cannot hold a whole module, so its strongest genes."""
    rows = [[int(label), len(module), float(module["margin_score"].mean()),
             _members(module["name"].tolist())] for label, module in modules.groupby("module_label")]
    listed = rows[:MAX_ROWS]
    return {"title": "Gene modules, each with its most connected genes first",
            "columns": ["Module", "Genes", "Mean margin", "Members"], "rows": listed}, len(listed)


def _members(names):
    """As many names as fit a table cell, whole, with the number left over."""
    text, left = "", len(names)
    for name in names:
        tail = f", +{left - 1} more" if left > 1 else ""
        if len(text) + len(name) + 2 + len(tail) > MEMBERS_CHARS:
            return f"{text}, +{left} more"
        text = f"{text}, {name}" if text else name
        left -= 1
    return text


def _summary(modules, n_genes, where, width, cutoff, power, bound, notes):
    sizes = modules.groupby("module_label").size()
    found = f"**{len(sizes):,} modules** hold" if len(sizes) > 1 else "**1 module** holds"
    lines = [f"{found} {len(modules):,} of {n_genes:,} genes, over {where}: smoothing width "
             f"{width:g} µm, correlation above {cutoff:g}, clustering power {power:g}."]
    largest = [f"Module {label} ({n:,} genes: "
               + ", ".join(modules.loc[modules["module_label"] == label, "name"].head(3)) + ")"
               for label, n in sizes.head(3).items()]
    lines.append("Largest: " + "; ".join(largest) + ".")
    if bound is None:
        lines.append("The authors' lower bound for the cutoff could not be computed: once "
                     "shuffled, this tissue fails the package's density check.")
    elif cutoff < bound:
        lines.append(f"**This cutoff is below the authors' lower bound:** with the positions "
                     f"shuffled, the 99.9th percentile of correlation is {bound:.2f}. Raise the "
                     "cutoff above it.")
    else:
        lines.append(f"With the positions shuffled, the 99.9th percentile of correlation is "
                     f"{bound:.2f}, the authors' lower bound for the cutoff.")
    if notes:
        lines.append(" ".join(notes))
    return "\n\n".join(lines)
