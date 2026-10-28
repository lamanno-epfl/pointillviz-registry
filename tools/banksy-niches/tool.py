"""BANKSY niches: tissue domains from each cell's expression and its neighbourhood's.

BANKSY (Singhal, Chou, Lee et al. 2024, Nat Genet 56:431-441) augments every
cell's own expression with what its spatial neighbours express, and clusters
the joint space. With most of the weight on the neighbourhood (lambda = 0.8)
the clusters are tissue domains, or niches; with little (lambda = 0.2) they are
spatially informed cell types. This file reimplements the published algorithm
in NumPy and SciPy, and clusters with python-igraph and leidenalg as the
authors do. It takes no code from the authors' R or Python packages.

The features
------------
Three blocks per cell, over the same genes:

  own   x_i, library-size-normalised counts (CP10K; the scale cancels in the
        z-scores below)
  H0    sum_j w_ij x_j over the k nearest cells, with a Gaussian kernel scaled
        by the cell's own median neighbour distance m_i,
        w_ij ∝ exp(-d_ij² / m_i²), each row summing to 1
  H1    the azimuthal Gabor filter (AGF) over the 2k nearest cells,
        | sum_j w_ij e^{iφ_ij} (x_j − mean_N(i) x) |, φ_ij the direction from i
        to j in the xy plane. It is large where expression changes across the
        neighbourhood (a border, a gradient) and small inside a uniform domain

Every column is z-scored over the cells being clustered, and the blocks are
weighted by sqrt(1 − λ), sqrt(2λ/3) and sqrt(λ/3), or by sqrt(1 − λ) and sqrt(λ)
without the AGF, as in the authors' code.

From features to niches
-----------------------
The joint matrix is reduced to its top principal components, which are then
clustered in one of two ways:

  leiden   the authors' default, as pybanksy's run_Leiden_partition runs it.
           Every cell is linked to its 50 nearest cells in component space,
           with weight 1 / (1 + distance). A link is kept when the two cells
           share at least one of those neighbours, and either share five or
           more or are among the three pairs the cell shares the most with.
           Leiden (RBConfiguration quality, seed 1234, until no cell moves)
           partitions that graph; the resolution sets how many niches it finds
  kmeans   k-means++ seeding, fixed seeds, best of four starts, into the
           requested number of niches. The authors' R package offers it too
           (`clusterBanksy(algo = "kmeans")`)

The joint matrix is never built: three columns per gene for every cell would be
13 GB at 550,000 cells and 2,000 genes. It is streamed in row chunks twice,
once into its Gram matrix (which gives the means, the standard deviations and
the components) and once to project onto the components. Memory is the Gram
matrix plus the chunks being built.

What comes back
---------------
The niche map, one colour per niche, on the main view. The matrix gives each
niche's cell-type composition when the dataset has cell types, and its marker
genes otherwise; the table lists their sizes, marker genes and main cell types;
and every cell's niche can be downloaded as a CSV.
"""
import gzip
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import igraph
import leidenalg
import numpy as np
from scipy import sparse
from scipy.cluster.vq import vq
from scipy.linalg import blas
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from pointillviz_tool import PluginError, download_file

#: Bytes of joint matrix per streamed chunk; the row count follows from the width.
CHUNK_BYTES = 48_000_000
#: Threads building chunks, one per CPU up to 16. SciPy's sparse products
#: release the GIL: on the 135k-cell colon section eight threads build the
#: features four times faster than one. Past 16 the speed is flat while each
#: thread's chunk adds memory (42 s and 15.5 GB at 128 threads, 44 s and
#: 3.1 GB at 16).
WORKERS = min(os.cpu_count() or 1, 16)
#: Rows per block when the covariance matrix is finished in place.
ROW_BLOCK = 4096
#: k-means starts; the one with the smallest within-niche sum of squares wins.
KMEANS_STARTS = 4
KMEANS_MAX_ITER = 100
#: The authors' Leiden graph, as run_Leiden_partition builds it: neighbours in
#: component space, the shared-neighbour filter (rank and count) and the seed.
LEIDEN_NEIGHBOURS = 50
SNN_MAX_RANK = 3
SNN_MIN_SHARED = 5
LEIDEN_SEED = 1234
#: Rows per block when counting shared neighbours.
SHARED_BLOCK = 1024
#: smoothLabels' defaults in the R package.
SMOOTH_K = 15
SMOOTH_ROUNDS = 10
#: Above this many feature columns the components come from ARPACK, not a full eigh.
DENSE_EIGH_MAX = 2000
#: The most niches a map can show: the labels output's limit on categories.
MAX_NICHES = 256
#: Views, files and images share an 8 MiB budget per result; the CSV is dropped past this.
RESULT_BUDGET = 7_500_000


def run(ds, params):
    lam = float(params["lambda"])
    k = int(params["k_geom"])
    use_agf = bool(params["agf"])
    method = params["clustering"]
    resolution = float(params["resolution"])
    n_niches = int(params["n_niches"])
    n_top = int(params["n_genes"])
    n_pcs = int(params["n_pcs"])

    ds.progress(0.01, "Finding spatial neighbours")
    idx0, dist0 = ds.knn(k)
    rows = np.flatnonzero((idx0 >= 0).any(axis=1))
    if ds.subset is not None:
        rows = np.intersect1d(rows, ds.subset)
    if method == "kmeans":
        floor, need = max(50, 10 * n_niches), f"{n_niches} niches need"
    else:
        floor, need = LEIDEN_NEIGHBOURS + 1, "Leiden's neighbour graph needs"
    if rows.size < floor:
        raise PluginError("too_few_cells",
                          f"{rows.size:,} cells have spatial neighbours here; "
                          f"{need} at least {floor:,}.")

    genes, X = _expression(ds, rows, n_top)
    G, n = len(genes), rows.size
    ds.log(f"{G:,} genes, {n:,} cells")

    ds.progress(0.30, "Weighting neighbourhoods")
    w0, ok0 = _gaussian_rows(idx0, dist0)
    W0 = _sparse_rows(w0, idx0, ok0, ds.n_cells)
    agf = _agf_operators(ds, 2 * k) if use_agf else None
    weights = ([np.sqrt(1 - lam), np.sqrt(2 * lam / 3), np.sqrt(lam / 3)] if use_agf
               else [np.sqrt(1 - lam), np.sqrt(lam)])

    def features(chunk):
        own = X[chunk].toarray()
        blocks = [own, (W0[chunk] @ X).toarray()]
        if agf is not None:
            w_re, w_im, uniform, s_re, s_im = agf
            local = (uniform[chunk] @ X).toarray()
            re = (w_re[chunk] @ X).toarray() - local * s_re[chunk, None]
            im = (w_im[chunk] @ X).toarray() - local * s_im[chunk, None]
            blocks.append(np.hypot(re, im))
        return np.hstack(blocks)

    F = len(weights) * G
    step = int(np.clip(CHUNK_BYTES // (8 * F), 256, 4096))
    chunks = [rows[i:i + step] for i in range(0, n, step)]
    pool = ThreadPoolExecutor(WORKERS)
    # Pass 1: the Gram matrix of the features, shifted by the first chunk's
    # means so that the variances do not come from subtracting two large sums.
    gram = np.zeros((F, F), order="F")
    total = np.zeros(F)
    shift = None
    for i, B in enumerate(_in_order(pool, features, chunks)):
        if shift is None:
            shift = B.mean(axis=0)
        B -= shift
        total += B.sum(axis=0)
        gram = blas.dsyrk(1.0, B.T, beta=1.0, c=gram, trans=0, lower=0, overwrite_c=1)
        ds.progress(0.32 + 0.38 * (i + 1) / len(chunks), "Building the joint matrix")
    centre = total / n
    mean = shift + centre
    cov = _covariance(gram, centre, n)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    live = sd > 1e-10 * (1.0 + np.abs(mean))
    scale = np.repeat(weights, G) / np.where(live, sd, 1.0)
    scale[~live] = 0.0
    for a in range(0, F, ROW_BLOCK):
        cov[a:a + ROW_BLOCK] *= scale[a:a + ROW_BLOCK, None] * scale[None, :]

    ds.progress(0.72, "Principal components")
    n_pcs = min(n_pcs, F - 1, n - 1)
    if F <= DENSE_EIGH_MAX:
        vals, vecs = np.linalg.eigh(cov)
    else:
        vals, vecs = eigsh(cov, k=n_pcs, which="LA", v0=np.full(F, F ** -0.5))
    order = np.argsort(vals)[::-1][:n_pcs]
    vals, vecs = vals[order], vecs[:, order]
    vecs *= np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(n_pcs)])
    explained = float(vals.sum() / max(np.trace(cov), 1e-300))
    # Pass 2: project. Folding the z-scoring into the loadings makes each chunk
    # one matrix product: ((B - mean) * scale) @ V = B @ (scale * V) - mean @ (scale * V).
    loadings = vecs * scale[:, None]
    offset = mean @ loadings
    pcs = np.empty((n, n_pcs))
    for i, B in enumerate(_in_order(pool, features, chunks)):
        pcs[i * step:i * step + len(B)] = B @ loadings - offset
        ds.progress(0.74 + 0.14 * (i + 1) / len(chunks), "Projecting")

    if method == "kmeans":
        ds.progress(0.90, "Clustering with k-means")
        labels, _ = min((_kmeans(pcs, n_niches, seed) for seed in range(KMEANS_STARTS)),
                        key=lambda fit: fit[1])
        labels, found = _by_size(labels, n_niches)
    else:
        ds.progress(0.89, "Linking cells that share neighbours")
        graph, edge_weights = _snn_graph(pcs, pool)
        ds.log(f"Leiden graph: {graph.ecount():,} links between {n:,} cells")
        ds.progress(0.91, "Clustering with Leiden")
        part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                        weights=edge_weights, resolution_parameter=resolution,
                                        n_iterations=-1, seed=LEIDEN_SEED)
        labels = np.asarray(part.membership, dtype=np.int64)
        labels, found = _by_size(labels, int(labels.max()) + 1)
        if found > MAX_NICHES:
            raise PluginError("too_many_niches",
                              f"Leiden found {found:,} niches at resolution {resolution:g}; "
                              f"a map shows at most {MAX_NICHES}. Lower the resolution.")
    pool.shutdown()
    moved = 0
    if params["smooth"]:
        niche = np.full(ds.n_cells, -1, dtype=np.int64)
        niche[rows] = labels
        smoothed = _smooth(niche, ds.knn(SMOOTH_K)[0], found)[rows]
        moved = int((smoothed != labels).sum())
        labels, found = _by_size(smoothed, found)
    niche = np.full(ds.n_cells, -1, dtype=np.int64)
    niche[rows] = labels
    sizes = np.bincount(labels, minlength=found)

    ds.progress(0.95, "Describing the niches")
    onehot = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(found, n))
    profile = (onehot @ X[rows]).toarray() / np.maximum(sizes, 1)[:, None]
    z = (profile - mean[:G]) / np.where(live[:G], sd[:G], np.inf)
    markers = [[genes[g] for g in np.argsort(-z[c], kind="stable")[:5]] for c in range(found)]
    composition = _composition(ds, rows, labels, found)

    names = [f"N{c + 1}" for c in range(found)]
    share = sizes / n
    clusterer = (f"Leiden (resolution {resolution:g})" if method == "leiden"
                 else f"k-means ({n_niches} niches asked for)")
    result = {
        "summary": _summary(n, G, found, method, clusterer, lam, k, use_agf, n_pcs, explained,
                            names, share, markers, moved, params["smooth"],
                            n_niches if method == "kmeans" else None),
        # A cell with no niche (no spatial neighbours, or outside the selection)
        # is NaN, which the map leaves unpainted.
        "labels": {"values": np.where(niche >= 0, niche, np.nan), "categories": names,
                   "label": "BANKSY niche"},
        "matrix": _matrix(names, composition, z, genes),
        "table": _table(names, sizes, share, markers, composition),
    }
    csv = "cell_index,niche\n" + "".join(f"{i},{names[c]}\n" for i, c in zip(rows, labels))
    packed = gzip.compress(csv.encode("ascii"), compresslevel=6, mtime=0)
    if 4 * len(packed) // 3 < RESULT_BUDGET:
        result["files"] = [download_file("banksy_niches.csv.gz", packed,
                                         media_type="application/gzip")]
    else:
        ds.log("The CSV of every cell's niche is larger than a result can carry, "
               "and was left out.")
    return result


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------
def _expression(ds, rows, n_top):
    """CSR (n_cells, G) CP10K expression of the G most variable genes, and their names.

    One read per gene: its non-zeros are kept, and its mean and variance over
    the clustered cells feed the ranking, so the panel is read once whatever
    its size. The ranking is the dispersion criterion of Seurat v1 and scanpy's
    `flavor="seurat"`, which the authors' Python tutorials use: how far a gene's
    variance-to-mean ratio stands above that of genes of similar abundance.
    """
    names = ds.gene_names
    member = np.zeros(ds.n_cells, dtype=bool)
    member[rows] = True
    n = rows.size
    totals = ds.total_counts()[rows]
    # scanpy's normalize_total scales to the median library size. Only the
    # ranking sees this factor; the z-scores downstream cancel it.
    scale = float(np.median(totals[totals > 0])) / 1e4 if (totals > 0).any() else 1.0
    nz_idx, nz_val = [], []
    mean = np.zeros(len(names))
    var = np.zeros(len(names))
    for g, name in enumerate(names):
        v = ds.gene(name, "cp10k")
        nz = np.flatnonzero(v)
        nz_idx.append(nz.astype(np.int32))
        nz_val.append(v[nz])
        u = v[nz[member[nz]]].astype(np.float64) * scale
        mean[g] = u.sum() / n
        var[g] = max((u * u).sum() - n * mean[g] ** 2, 0.0) / max(n - 1, 1)
        if g % 256 == 0:
            ds.progress(0.02 + 0.26 * g / len(names), f"Reading genes ({g:,} of {len(names):,})")
    keep = _most_dispersed(mean, var, n_top)
    if keep.size < 2:
        raise PluginError("too_few_genes", "Fewer than two genes vary across these cells.")
    indptr = np.concatenate([[0], np.cumsum([nz_idx[g].size for g in keep])])
    X = sparse.csc_matrix((np.concatenate([nz_val[g] for g in keep]).astype(np.float64),
                           np.concatenate([nz_idx[g] for g in keep]), indptr),
                          shape=(ds.n_cells, keep.size)).tocsr()
    return [names[g] for g in keep], X


def _most_dispersed(mean, var, n_top):
    """Indices of the n_top genes with the highest bin-normalised log dispersion."""
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(mean > 0, var / mean, np.nan)
        disp[disp == 0] = np.nan
        disp = np.log(disp)
    ok = np.isfinite(disp)
    if not ok.any():
        return np.array([], dtype=np.int64)
    lmean = np.log1p(mean)
    lo, hi = lmean.min(), lmean.max()
    edges = np.linspace(lo, hi, 21)
    edges[0] -= 0.001 * (hi - lo)         # pandas.cut's first bin includes the minimum
    b = np.clip(np.searchsorted(edges, lmean, side="left") - 1, 0, 19)
    norm = np.full(mean.shape, np.nan)
    for i in range(20):
        m = ok & (b == i)
        if m.sum() == 1:
            norm[m] = 1.0                  # scanpy scores a gene alone in its bin as 1
        elif m.any():
            d = disp[m]
            sd = d.std(ddof=1)
            norm[m] = (d - d.mean()) / sd if sd > 0 else 0.0
    order = np.argsort(-np.where(ok, norm, -np.inf), kind="stable")
    return np.sort(order[:min(n_top, int(ok.sum()))])


# ---------------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------------
def _gaussian_rows(idx, dist):
    """Row-normalised exp(-d²/m²) weights, m the cell's median neighbour distance.

    `idx` and `dist` are ds.knn's, with -1 and inf where a slot could not be
    filled; those slots get no weight. A cell whose median distance is zero
    (stacked duplicates) takes the kernel's limit as m falls to zero: equal
    weight on the neighbours at distance zero.
    """
    ok = idx >= 0
    d = np.where(ok, dist, np.nan).astype(np.float64)
    have = ok.any(axis=1)
    med = np.zeros(len(idx))
    med[have] = np.nanmedian(d[have], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.exp(-(d / med[:, None]) ** 2)
    zero = have & (med == 0)
    w[zero] = d[zero] == 0
    w[~ok] = 0.0
    tot = w.sum(axis=1)
    w[tot > 0] /= tot[tot > 0, None]
    return w, ok


def _sparse_rows(values, idx, ok, n):
    r = np.repeat(np.arange(len(idx)), idx.shape[1])
    m = ok.ravel()
    return sparse.csr_matrix((values.ravel()[m], (r[m], idx.ravel()[m])), shape=(n, n))


def _agf_operators(ds, k):
    """Sparse operators for the AGF over k neighbours.

    Returns the real and imaginary parts of w_ij e^{iφ_ij}, the uniform
    neighbourhood mean that centres the AGF, and each row's sum of the complex
    weights, so that H1 = |W_re X − (U X) s_re + i (W_im X − (U X) s_im)|.
    """
    idx, dist = ds.knn(k)
    w, ok = _gaussian_rows(idx, dist)
    xy = ds.coords[:, :2].astype(np.float64)
    j = np.where(ok, idx, 0)
    phi = np.arctan2(xy[j, 1] - xy[:, 1:2], xy[j, 0] - xy[:, 0:1])
    re = np.where(ok, w * np.cos(phi), 0.0)
    im = np.where(ok, w * np.sin(phi), 0.0)
    uniform = np.where(ok, 1.0 / np.maximum(ok.sum(axis=1), 1)[:, None], 0.0)
    n = ds.n_cells
    return (_sparse_rows(re, idx, ok, n), _sparse_rows(im, idx, ok, n),
            _sparse_rows(uniform, idx, ok, n), re.sum(axis=1), im.sum(axis=1))


def _covariance(gram, centre, n):
    """In place: dsyrk's upper triangle -> the full population covariance.

    Row blocks rather than whole-matrix expressions, because at 5,000 genes the
    matrix is 1.8 GB and every `np.triu(...)` or `np.outer(...)` would be another.
    """
    F = len(centre)
    for a in range(0, F, ROW_BLOCK):
        b = min(a + ROW_BLOCK, F)
        gram[a:b, :a] = gram[:a, a:b].T
        diag = gram[a:b, a:b]
        gram[a:b, a:b] = np.triu(diag) + np.triu(diag, 1).T
    for a in range(0, F, ROW_BLOCK):
        b = min(a + ROW_BLOCK, F)
        gram[a:b] /= n
        gram[a:b] -= np.outer(centre[a:b], centre)
    return gram


def _in_order(pool, fn, items):
    """pool.map, but with at most WORKERS + 2 results built ahead of the consumer."""
    pending = deque()
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) > WORKERS + 1:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _snn_graph(pcs, pool):
    """The authors' shared-nearest-neighbour graph of the components, and its weights.

    pybanksy's LeidenPartition, as run_Leiden_partition sets it up: each cell's
    50 nearest cells by Euclidean distance (itself excluded), a link to each
    weighted 1 / (1 + d), kept when the two share at least one of those
    neighbours and either share five or more or are among the three the cell
    shares the most with (ties kept). The links are undirected, so a pair kept
    from both ends is linked twice, as it is there. Built with sparse products
    and handed to igraph as an edge list.
    """
    n = len(pcs)
    k = min(LEIDEN_NEIGHBOURS, n - 1)
    dist, idx = cKDTree(pcs).query(pcs, k=k + 1, workers=-1)
    # Drop the cell itself, as scikit-learn's kneighbors_graph does; when
    # duplicates crowd it out of the k + 1 nearest, the first of them goes.
    other = idx != np.arange(n)[:, None]
    other[other.all(axis=1), 0] = False
    idx, dist = idx[other].reshape(n, k), dist[other].reshape(n, k)

    shared = _shared_counts(idx, pool)
    linked = shared > 0
    rank = min(SNN_MAX_RANK, k)
    top = np.sort(shared, axis=1)[:, -rank]
    keep = linked & ((linked.sum(axis=1) <= rank)[:, None]
                     | (shared >= top[:, None]) | (shared >= SNN_MIN_SHARED))
    source = np.repeat(np.arange(n), k)[keep.ravel()]
    graph = igraph.Graph(n=n, edges=zip(source.tolist(), idx[keep].tolist()), directed=False)
    return graph, (1.0 / (1.0 + dist[keep])).tolist()


def _shared_counts(idx, pool):
    """(n, k): how many of their nearest neighbours each cell shares with each of them.

    Row blocks of the neighbour matrix times its transpose, each read at the
    cell's own neighbours only.
    """
    n, k = idx.shape
    nn = sparse.csr_matrix((np.ones(n * k, dtype=np.int32), idx.ravel(),
                            np.arange(0, n * k + 1, k)), shape=(n, n))
    nn_t = nn.T.tocsr()

    def block(a):
        b = min(a + SHARED_BLOCK, n)
        both = nn[a:b] @ nn_t
        both.sort_indices()
        # Row-major keys are sorted, so each neighbour's count is one binary search away.
        key = np.repeat(np.arange(b - a, dtype=np.int64) * n, np.diff(both.indptr)) + both.indices
        want = (np.arange(b - a, dtype=np.int64)[:, None] * n + idx[a:b]).ravel()
        at = np.minimum(np.searchsorted(key, want), key.size - 1)
        return np.where(key[at] == want, both.data[at], 0).reshape(b - a, k)

    return np.vstack(list(pool.map(block, range(0, n, SHARED_BLOCK))))


def _kmeans(x, k, seed):
    """Lloyd's k-means from k-means++ seeds. Returns (labels, sum of squares)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centres = np.empty((k, x.shape[1]))
    centres[0] = x[rng.integers(n)]
    d2 = ((x - centres[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        tot = d2.sum()
        centres[c] = x[rng.choice(n, p=d2 / tot) if tot > 0 else rng.integers(n)]
        d2 = np.minimum(d2, ((x - centres[c]) ** 2).sum(axis=1))
    labels = np.full(n, -1)
    for _ in range(KMEANS_MAX_ITER):
        new, dist = vq(x, centres)
        if np.array_equal(new, labels):
            break
        labels = new
        onehot = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
        size = np.asarray(onehot.sum(axis=1)).ravel()
        filled = size > 0
        centres[filled] = (onehot @ x)[filled] / size[filled, None]
    return labels.astype(np.int64), float((dist ** 2).sum())


def _by_size(labels, k):
    """Renumber niches by size, largest first, dropping empty ones. Returns (labels, count)."""
    sizes = np.bincount(labels, minlength=k)
    rank = np.empty(k, dtype=np.int64)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(k)
    return rank[labels], int((sizes > 0).sum())


def _smooth(niche, idx, n_labels):
    """BANKSY's smoothLabels: majority of the cell and its neighbours, if over half."""
    n = len(niche)
    voters = np.concatenate([np.arange(n)[:, None], idx], axis=1)
    r = np.repeat(np.arange(n), voters.shape[1])
    for _ in range(SMOOTH_ROUNDS):
        v = np.where(voters >= 0, niche[np.maximum(voters, 0)], -1).ravel()
        good = v >= 0
        counts = sparse.csr_matrix((np.ones(good.sum()), (r[good], v[good])),
                                   shape=(n, n_labels)).toarray()
        top = counts.argmax(axis=1)
        share = counts.max(axis=1) / np.maximum(good.reshape(n, -1).sum(axis=1), 1)
        move = (niche >= 0) & (share > 0.5) & (top != niche)
        if not move.any():
            break
        niche = niche.copy()
        niche[move] = top[move]
    return niche


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
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


def _summary(n, G, found, method, clusterer, lam, k, use_agf, n_pcs, explained, names,
             share, markers, moved, smooth, asked):
    lines = [
        f"**{found} niche{'' if found == 1 else 's'}** over {n:,} cells, from {G:,} variable genes "
        f"(λ = {lam:g}, k = {k}{', gradients over ' + str(2 * k) if use_agf else ''}), "
        f"by {clusterer} on {n_pcs} principal components, which hold "
        f"{100 * explained:.0f}% of the joint matrix's variance.",
        "",
        f"- Largest: {names[0]}, {100 * share[0]:.1f}% of cells ({', '.join(markers[0][:3])}). "
        f"Smallest: {names[-1]}, {100 * share[-1]:.1f}% ({', '.join(markers[-1][:3])}).",
        "- Niche numbers are ordered by size and carry no other order.",
    ]
    if method == "leiden":
        lines.append("- Leiden chose the number of niches. Raise the resolution for more, "
                     "smaller niches, or lower it for fewer.")
    else:
        lines.append("- k-means divides the cells into exactly the number of niches asked "
                     "for. If two niches share their marker genes and cell types, ask for fewer.")
        if found < asked:
            lines.append(f"- {asked - found} of the {asked} niches asked for ended up empty.")
    if smooth:
        lines.append(f"- Smoothing moved {moved:,} cells to their neighbourhood's niche.")
    return "\n".join(lines)
