"""BANKSY niches: tissue domains from each cell's expression and its neighbourhood's.

BANKSY (Singhal, Chou, Lee et al. 2024, Nat Genet 56:431-441) augments every cell's own expression
with what its spatial neighbours express, and clusters the joint space. With most of the weight on
the neighbourhood (lambda = 0.8) the clusters are tissue domains, or niches; with little
(lambda = 0.2) they are spatially informed cell types.

This file computes none of that. It imports the authors' Python package, `pybanksy`, from PyPI and
calls its functions in the order of their notebooks, with their values: normalize_total and
filter_hvg, initialize_banksy, generate_banksy_matrix, pca_umap, run_Leiden_partition and, when
asked, refine_once. Sections are the samples of their DLPFC_concatenate_multisample notebook: each
gets its own spatial graph and z-scores, and the matrices are joined before the PCA. Or, when asked,
the same functions run once over all the sections, set apart in space so that each keeps its own
neighbours, and the z-scores are then taken over all of them. What is here is the way in and the way
out: counts from the dataset into AnnData, sentences where the package would crash or quietly lose
a block, and the niches back on the map with a description of each.
"""
import gc
import gzip
import random
import warnings

import matplotlib

matplotlib.use("Agg")  # the package imports pyplot, and a tool opens no window

# A warning opens with the path of its file, and the registry refuses a gate report that holds a path.
warnings.formatwarning = lambda message, category, *_: f"{category.__name__}: {message}\n"
# The package's note at import that mclust needs rpy2. mclust is not offered here.
warnings.filterwarnings("ignore", message=r"\s*No rpy2 installed", category=UserWarning)

import anndata as ad
import banksy
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
from banksy.cluster_methods import run_Leiden_partition
from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy_utils.filter_utils import filter_hvg, normalize_total
from banksy_utils.refine_clusters import refine_once
from banksy_utils.umap_pca import pca_umap
from scipy import sparse
from threadpoolctl import threadpool_limits

from pointillviz_tool import PluginError, download_file

SEED = 1234                 # the authors' notebooks: np.random, random and partition_seed
COORD_KEYS = ("x", "y", "coord_xy")
HVG_FLAVOR = "seurat"       # filter_hvg's default, which their notebooks recommend
REFINE_NEIGHBOURS = 6       # refine_once's default num_neigh
#: Bytes the authors' code holds at its peak for a section, in units of the joint matrix: the blocks,
#: their concatenation, their z-scores and the matrix itself, all alive within generate_banksy_matrix.
#: Measured: 32 GB at 235,803 cells and 21 GB at 135,231, at 2,000 genes, whose joint matrices are
#: 11.3 and 6.5 GB.
PEAK_IN_SECTION = 3.0
#: The PCA centres a copy of the joint matrix, and joining sections builds a second one.
PEAK_JOINT = 2.0
BASE_BYTES = 2e9            # the counts, the libraries and the graph built after the PCA
#: BLAS and OpenMP threads, which change no result. The package's AGF takes one small product per cell in
#: a Python loop, and waking a pool of threads for each costs far more than it gives: a section of 2,224
#: cells took 10 minutes at the default of 124 threads on a busy server, and under 30 s at one.
THREADS = 1
#: The one step that gains from threads. Its labels were identical at 1 and 8 threads on 135,231 cells.
PCA_THREADS = 8
MAX_NICHES = 256            # the labels output's limit on categories
RESULT_BUDGET = 7_500_000   # views and files share an 8 MiB budget per result


def run(ds, params):
    with threadpool_limits(limits=THREADS):
        return _run(ds, params)


def _run(ds, params):
    lam = float(params["lambda_param"])
    k_geom = int(params["k_geom"])
    max_m = int(params["max_m"])
    decay = str(params["nbr_weight_decay"])
    pca_dim = int(params["pca_dims"])
    resolution = float(params["resolution"])
    max_labels = int(params["max_labels"])
    num_nn = int(params["num_nn"])
    n_top = int(params["n_top_genes"])
    refine_iterations = int(params["refine_iterations"])
    zscore_by = str(params["zscore_by"])

    if decay == "ranked" and max_m > 0:
        raise PluginError("ranked_needs_m0",
                          "pybanksy's ranked weights work only without gradients: its code fails "
                          "at an AGF order of 1. Set the order to 0, or choose other weights.")
    if max_labels == 1:
        raise PluginError("one_niche", "Asking for one niche asks for nothing: set Number of niches "
                                       "to 0 to go by the resolution, or to 2 or more.")

    # As the notebooks open: "the seeds affect the umap clusters and leiden partition".
    np.random.seed(SEED)
    random.seed(SEED)

    notes = []
    ds.progress(0.01, "Choosing the cells")
    rows, names_of = _cells(ds)
    genes = _genes_of_every_section(ds, ds.section[rows], notes)
    adata = _anndata(ds, rows, genes)

    before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=1)
    if adata.n_obs < before:
        notes.append(f"{before - adata.n_obs:,} cells with no counts were left out.")
    adata = _sections_large_enough(adata, k_geom, max_m, names_of, notes)
    sc.pp.filter_genes(adata, min_cells=1)
    if adata.n_vars < 2:
        raise PluginError("too_few_genes", "Fewer than two genes are expressed in these cells.")
    if adata.n_obs <= num_nn:
        raise PluginError("too_few_cells",
                          f"{adata.n_obs:,} cells can be clustered here, and Leiden's neighbour "
                          f"graph needs more than its {num_nn} neighbours. Select more cells, or "
                          "lower Expression neighbours.")

    # The authors' preprocessing, on the sparse counts, as their notebooks do it.
    ds.progress(0.20, "Normalising and choosing the variable genes")
    adata = normalize_total(adata)
    adata, _ = filter_hvg(adata, n_top_genes=n_top, flavor=HVG_FLAVOR)
    adata = adata.copy()
    gc.collect()
    G, n = adata.n_vars, adata.n_obs
    ds.log(f"{G:,} variable genes, {n:,} cells, {adata.obs['section'].nunique():,} sections")
    _enough_memory(adata, max_m, zscore_by)

    ds.progress(0.25, "Building the BANKSY matrix")
    groups = list(adata.obs.groupby("section", observed=True).indices.items())
    section_cells = [(where, adata.obsm[COORD_KEYS[2]][where]) for _, where in groups]
    if len(groups) > 1:
        notes.append(f"{len(groups)} sections, z-scored "
                     + ("across all of them." if zscore_by == "all" else "each on its own."))
    if zscore_by == "all" and len(groups) > 1:
        whole = adata.copy()
        whole.obsm[COORD_KEYS[2]] = _set_apart(adata.obsm[COORD_KEYS[2]], [w for _, w in groups])
        joined = _banksy_matrix(whole, k_geom, decay, max_m, lam, "these sections")
        del whole
        order = np.arange(n)
    else:
        matrices = []
        for i, (code, where) in enumerate(groups):
            ds.progress(0.25 + 0.30 * i / len(groups), f"Building the BANKSY matrix ({names_of[int(code)]})")
            matrices.append(_banksy_matrix(adata[where].copy(), k_geom, decay, max_m, lam,
                                           f"section {names_of[int(code)]}"))
            gc.collect()
        joined = matrices[0] if len(matrices) == 1 else ad.concat(matrices, join="outer")
        del matrices
        order = np.concatenate([where for _, where in groups])
    gc.collect()
    banksy_dict = {decay: {lam: {"adata": joined}}}

    ds.progress(0.58, "Principal components")
    pca_dim = min(pca_dim, joined.n_vars, joined.n_obs - 1)
    with threadpool_limits(limits=PCA_THREADS):
        pca_umap(banksy_dict, pca_dims=[pca_dim], plt_remaining_var=False, add_umap=False)
    joined.X = None         # only the components are clustered
    gc.collect()

    ds.progress(0.68, "Clustering with Leiden")
    results, _ = run_Leiden_partition(banksy_dict, None if max_labels else [resolution], num_nn=num_nn,
                                      num_iterations=-1, partition_seed=SEED, match_labels=False,
                                      max_labels=max_labels or None, verbose=False)
    found_at = float(results.iloc[0]["resolution"])
    in_order = np.asarray(results.iloc[0]["labels"].dense, dtype=np.int64)
    if len(np.unique(in_order)) > MAX_NICHES:
        raise PluginError("too_many_niches",
                          f"Leiden found {len(np.unique(in_order)):,} niches at resolution "
                          f"{found_at:g}; a map shows at most {MAX_NICHES}. Lower the resolution.")
    del banksy_dict, joined, results
    gc.collect()

    labels = np.empty(n, dtype=np.int64)
    labels[order] = in_order
    moved = 0
    if refine_iterations:
        ds.progress(0.90, "Refining the labels")
        raw = labels.copy()
        labels = _refine(labels, section_cells, refine_iterations)
        moved = int((labels != raw).sum())

    ds.progress(0.95, "Describing the niches")
    labels, found = _by_size(labels)
    kept = adata.obs["row"].to_numpy()
    niche = np.full(ds.n_cells, -1, dtype=np.int64)
    niche[kept] = labels
    sizes = np.bincount(labels, minlength=found)
    gene_names = adata.var_names.tolist()
    markers, z = _markers(adata.X, labels, found, sizes, gene_names)
    composition = _composition(ds, kept, labels, found)
    names = [f"N{c + 1}" for c in range(found)]
    share = sizes / n

    result = {
        "summary": _summary(n, G, found, names, share, markers, lam, k_geom, max_m, decay, pca_dim,
                            found_at, num_nn, max_labels, refine_iterations, moved, notes),
        # A cell with no niche (no coordinates, outside the selection, or in a section left out)
        # is NaN, which the map leaves unpainted.
        "labels": {"values": np.where(niche >= 0, niche, np.nan), "categories": names,
                   "label": "BANKSY niche"},
        "matrix": _matrix(names, composition, z, gene_names),
        "table": _table(names, sizes, share, markers, composition),
    }
    csv = "cell_index,niche\n" + "".join(f"{i},{names[c]}\n" for i, c in zip(kept, labels))
    packed = gzip.compress(csv.encode("ascii"), compresslevel=6, mtime=0)
    if 4 * len(packed) // 3 < RESULT_BUDGET:
        result["files"] = [download_file("banksy_niches.csv.gz", packed,
                                         media_type="application/gzip")]
    else:
        ds.log("The CSV of every cell's niche is larger than a result can carry, and was left out.")
    ds.progress(1.0)
    return result


# ---- the way in -----------------------------------------------------------------------------------
def _section_entries(ds):
    """Each code of ds.section -> its entry in ds.sections, matched as the SDK matches them: by
    `index` on a flat dataset, and by section_id on an atlas bound to a composition, whose codes are
    positions in that composition."""
    composition = ds.meta.get("_composition")
    if composition:
        by_id = {s.get("section_id"): s for s in ds.sections}
        return {i: by_id.get(c["section_id"], {}) for i, c in enumerate(composition)}
    return {int(s.get("index", i)): s for i, s in enumerate(ds.sections)}


def _cells(ds):
    """(the rows to cluster, each section code's name)."""
    keep = np.isfinite(ds.coords).all(axis=1)
    if ds.section_filter is not None:
        keep &= ds.section == ds.section_filter
    if ds.subset is not None and ds.subset.size:
        chosen = np.zeros(ds.n_cells, dtype=bool)
        chosen[ds.subset] = True
        keep &= chosen
    rows = np.flatnonzero(keep)
    if not rows.size:
        raise PluginError("too_few_cells", "None of these cells has coordinates.")
    entries = _section_entries(ds)
    return rows, {int(s): str(entries.get(int(s), {}).get("section_id") or s)
                  for s in np.unique(ds.section[rows])}


def _sections_large_enough(adata, k_geom, max_m, names_of, notes):
    """A section needs more cells than the authors' spatial graph has neighbours, and than
    refine_once has; one with fewer is left out, and said so."""
    floor = max(k_geom * (max_m + 1), REFINE_NEIGHBOURS) + 1
    sizes = adata.obs["section"].value_counts()
    small = sizes.index[sizes < floor]
    for s in sorted(small):
        notes.append(f"Section {names_of[int(s)]} left out: its {int(sizes[s]):,} cells are too few "
                     f"for the {floor - 1} neighbours of the spatial graph.")
    if len(small) == len(sizes):
        raise PluginError("too_few_cells",
                          f"No section here has the {floor:,} cells the spatial graph needs at "
                          f"{k_geom} neighbours. Select more cells, or lower Spatial neighbours.")
    return adata[~adata.obs["section"].isin(small)].copy() if len(small) else adata


def _genes_of_every_section(ds, codes, notes):
    """The panel's genes that every section in this run measured: pybanksy's matrices are joined gene
    by gene, and a gene a section never measured is not a zero there."""
    panel = ds.gene_names
    names = list(dict.fromkeys(str(g) for g in panel))
    entries = _section_entries(ds)
    lacking = set()
    for s in np.unique(codes):
        lacking.update(str(panel[g]) for g in entries.get(int(s), {}).get("unmeasured_genes") or ())
    if lacking:
        notes.append(f"{len(lacking):,} genes were left out because not every section measured them.")
    return [g for g in names if g not in lacking]


def _anndata(ds, rows, genes):
    """The counts of these cells as an AnnData, CSR in float64, with what the authors' code reads:
    obs x and y, obsm coord_xy. `row` and `section` are this tool's, to map the results back."""
    hits, values, ptr = [], [], [0]
    for j, name in enumerate(genes):
        x = ds.gene(name, "raw")[rows]
        hit = np.flatnonzero((x != 0) & ~np.isnan(x))
        hits.append(hit)
        values.append(x[hit])
        ptr.append(ptr[-1] + hit.size)
        if j % 256 == 0:
            ds.progress(0.02 + 0.16 * j / max(len(genes), 1), "Reading the counts")
    counts = sparse.csc_matrix((np.concatenate(values).astype(np.float64), np.concatenate(hits),
                                np.array(ptr)), shape=(rows.size, len(genes))).tocsr()
    xy = ds.coords[rows].astype(np.float64)
    obs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "row": rows, "section": ds.section[rows]},
                       index=rows.astype(str))
    adata = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    adata.obsm[COORD_KEYS[2]] = xy
    return adata


def _set_apart(xy, groups):
    """The coordinates with each section moved along x, past the last and further than any section is
    wide, so that the package's spatial graph, built over all the cells, links each cell only within its
    section: every section has more cells than the graph has neighbours (_sections_large_enough). The
    AGF's angles are differences of coordinates, which do not change."""
    out = xy.copy()
    gap = 2.0 * max(float(np.linalg.norm(np.ptp(xy[w], axis=0))) for w in groups) + 1.0
    offset = 0.0
    for w in groups:
        out[w, 0] += offset - xy[w, 0].min()
        offset += float(np.ptp(xy[w, 0])) + gap
    return out


def _banksy_matrix(part, k_geom, decay, max_m, lam, where):
    """The authors' spatial graph and BANKSY matrix of these cells, z-scored over them. Returning only
    the matrix lets the blocks, their concatenation and the weights go: nothing below reads them."""
    bdict = initialize_banksy(part, COORD_KEYS, k_geom, nbr_weight_decay=decay, max_m=max_m,
                              plt_edge_hist=False, plt_nbr_weights=False, plt_agf_angles=False,
                              plt_theta=False)
    _refuse_stacked_cells(bdict[decay]["weights"], decay, where)
    bdict, _ = generate_banksy_matrix(part, bdict, [lam], max_m, verbose=False)
    return bdict[decay][lam]["adata"]


def _enough_memory(adata, max_m, zscore_by):
    """The authors' code holds dense matrices, several at once: say so before the computer runs out.
    A dense joint matrix is 8 bytes x cells x genes x (2 + order)."""
    per_cell = 8.0 * adata.n_vars * (2 + max_m)
    needed = running = BASE_BYTES
    sizes = [adata.n_obs] if zscore_by == "all" else adata.obs["section"].value_counts().to_numpy()
    for size in sizes:
        needed = max(needed, running + PEAK_IN_SECTION * per_cell * size)
        running += per_cell * size
    needed = max(needed, BASE_BYTES + PEAK_JOINT * per_cell * adata.n_obs)
    free = psutil.virtual_memory().available
    if needed > free:
        raise PluginError(
            "not_enough_memory",
            f"pybanksy would need about {needed / 1e9:,.0f} GB for {adata.n_obs:,} cells and "
            f"{adata.n_vars:,} genes, and this computer has {free / 1e9:,.0f} GB free. Select fewer "
            "cells, isolate a section, lower Variable genes (it takes memory in proportion), or set "
            "the AGF order to 0.")


def _refuse_stacked_cells(weights, decay, where):
    """A cell standing on its neighbours gets weights of 0/0 or x/0. One such row makes its genes' means
    NaN, and the package's z-score then turns the whole neighbourhood block to zeros without a word."""
    for graph in weights.values():
        bad = ~np.isfinite(graph.data)
        if bad.any():
            cells = np.unique(np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))[bad]).size
            raise PluginError(
                "stacked_cells",
                f"In {where}, {cells:,} cells stand on the same spot as their neighbours, "
                f"which pybanksy's {decay} weights cannot handle: it would lose the neighbourhood "
                "without saying so. Choose the uniform weights, or select cells without these.")


# ---- the way out ----------------------------------------------------------------------------------
def _refine(labels, section_cells, iterations):
    """The authors' refine_once, section by section as in their notebook: a cell takes the label of
    most of its neighbours."""
    out = labels.copy()
    for where, xy in section_cells:
        part = ad.AnnData(obs=pd.DataFrame(index=where.astype(str)), obsm={COORD_KEYS[2]: xy})
        current = out[where]
        for _ in range(iterations):
            current, _, _ = refine_once(part, current, None, COORD_KEYS, REFINE_NEIGHBOURS)
        out[where] = current
    return out


def _by_size(labels):
    """Renumber niches by size, largest first. Returns (labels, count)."""
    ids, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    rank = np.empty(ids.size, dtype=np.int64)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(ids.size)
    return rank[inverse], int(ids.size)


def _markers(X, labels, found, sizes, genes):
    """Each niche's five highest genes by the z-score of its mean expression. This tool's description
    of the niches, on the normalised counts of the variable genes; it plays no part in finding them."""
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    var = np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.0))
    onehot = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(found, n))
    profile = (onehot @ X).toarray() / np.maximum(sizes, 1)[:, None]
    z = (profile - mean) / np.where(sd > 0, sd, np.inf)
    return [[genes[g] for g in np.argsort(-z[c], kind="stable")[:5]] for c in range(found)], z


def _composition(ds, rows, labels, found):
    """(levels, % of each niche's labelled cells) for the cell-type column, or None."""
    try:
        levels, codes = ds.cell_types()
    except PluginError:
        return None
    c = codes[rows]
    typed = c >= 0
    if not typed.any():
        return None
    counts = np.bincount(labels[typed] * len(levels) + c[typed],
                         minlength=found * len(levels)).reshape(found, len(levels))
    present = np.flatnonzero(counts.sum(axis=0) > 0)
    present = present[np.argsort(-counts[:, present].sum(axis=0), kind="stable")]
    pct = 100.0 * counts[:, present] / np.maximum(counts.sum(axis=1), 1)[:, None]
    return [levels[t] for t in present], pct


def _matrix(names, composition, z, genes):
    if composition is not None:
        levels, pct = composition
        keep = min(len(levels), 60, 10000 // len(names))
        return {"title": "Cell-type composition of each niche (% of the niche's typed cells)",
                "rows": names, "cols": levels[:keep], "values": pct[:, :keep].tolist(),
                "colormap": "viridis"}
    picked = []
    for c in range(len(names)):
        for g in np.argsort(-z[c], kind="stable")[:3]:
            if g not in picked:
                picked.append(int(g))
    picked = picked[:10000 // len(names)]
    return {"title": "Marker genes of each niche (mean z-score of the niche's own expression)",
            "rows": names, "cols": [genes[g] for g in picked],
            "values": z[:, picked].tolist(), "diverging": True}


def _table(names, sizes, share, markers, composition):
    columns = ["Niche", "Cells", "% of cells", "Marker genes"]
    rows = [[names[c], int(sizes[c]), round(100 * float(share[c]), 2), ", ".join(markers[c])]
            for c in range(len(names))]
    if composition is not None:
        levels, pct = composition
        columns.append("Main cell types")
        for c, row in enumerate(rows):
            top = np.argsort(-pct[c], kind="stable")[:3]
            row.append(", ".join(f"{levels[t]} {pct[c, t]:.0f}%" for t in top if pct[c, t] > 0))
    return {"title": "Niches", "columns": columns, "rows": rows, "sort_by": "Cells"}


def _summary(n, G, found, names, share, markers, lam, k_geom, max_m, decay, pca_dim, found_at, num_nn,
             max_labels, refine_iterations, moved, notes):
    lines = [
        f"**{found} niche{'' if found == 1 else 's'}** over {n:,} cells, from {G:,} variable genes, by "
        f"pybanksy {banksy.__version__}: lambda {lam:g}, {k_geom} spatial neighbours, AGF order {max_m}, "
        f"{decay} weights, {pca_dim} principal components, Leiden at resolution {found_at:.2f} on "
        f"{num_nn} expression neighbours.",
        "",
        f"- Largest: {names[0]}, {100 * share[0]:.1f}% of cells ({', '.join(markers[0][:3])}). "
        f"Smallest: {names[-1]}, {100 * share[-1]:.1f}% ({', '.join(markers[-1][:3])}).",
        "- Niche numbers are ordered by size and carry no other order.",
    ]
    if max_labels and found != max_labels:
        lines.append(f"- pybanksy's search for {max_labels} niches stopped at {found}, after its 25 "
                     "tries of the resolution. Set the resolution yourself to go further.")
    if refine_iterations:
        lines.append(f"- Refinement moved {moved:,} cells to their neighbourhood's niche.")
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines)
