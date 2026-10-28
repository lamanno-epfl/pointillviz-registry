# Mclust niches: niches from the mix of cell types around each cell, fitted
# by a Gaussian mixture in native R.
#
# Every cell is described by the share of each cell type among its k nearest
# cells. prcomp reduces those shares to at most 10 principal components, and
# mclust fits a Gaussian mixture with one shared covariance (model "EEE") and
# n_niches components to 2,000 of the cells, drawn at random from `seed`. The
# fitted mixture then gives every cell its most likely niche. Fitting on a
# sample keeps the run short on large sections: mclust starts from a
# hierarchical clustering whose cost grows with the square of the cells.
#
# What `ds` hands back is 1-based, with NA for "no value": the codes of
# ds$cell_types() index its levels, and the neighbours of ds$knn() index cells.

# Attached, not only called as mclust::Mclust: Mclust() looks its helpers up
# by name from the caller, so they must be on the search path.
suppressPackageStartupMessages(library(mclust))

#: Cells the mixture is fitted to.
FIT_CELLS <- 2000L
#: Principal components the mixture sees, at most.
MAX_DIMS <- 10L
#: Components whose spread is below this fraction of the first's are dropped:
#: shares sum to one, so the last direction carries nothing, and a direction
#: with no spread would make the shared covariance singular.
DIM_TOL <- 1e-3

run <- function(ds, params) {
  k <- params$k
  n_niches <- params$n_niches
  types <- ds$cell_types()          # list(levels = chr, codes = int), NA = unlabelled
  nn <- ds$knn(k)                   # list(idx = n x k int matrix, dist), NA = no neighbour
  n <- ds$n_cells
  n_types <- length(types$levels)

  # Count each cell's neighbours by type: one tabulate over (cell, type) pairs.
  # as.vector() reads the matrix column by column, so the cell index cycles.
  neighbour <- types$codes[as.vector(nn$idx)]
  cell <- rep(seq_len(n), times = ncol(nn$idx))
  counts <- matrix(tabulate((cell - 1L) * n_types + neighbour, nbins = n * n_types),
                   nrow = n, byrow = TRUE)
  labelled <- rowSums(counts)
  used <- labelled > 0
  if (!is.null(ds$subset)) used <- used & seq_len(n) %in% ds$subset
  ds$log(sprintf("%d of %d cells have a labelled neighbour among their %d nearest",
                 sum(used), n, k))
  if (sum(used) < 10L * n_niches) {
    pv_error("too_few_cells", sprintf(
      "%s cells have a labelled neighbour here; %d niches need at least %s.",
      format(sum(used), big.mark = ","), n_niches, format(10L * n_niches, big.mark = ",")))
  }
  mix <- counts[used, , drop = FALSE] / labelled[used]

  ds$progress(0.3, "Principal components of the mixes")
  varying <- apply(mix, 2L, stats::var) > 0
  if (sum(varying) < 2L) {
    pv_error("no_variation", "The cells' surroundings hold fewer than two cell types, so there is nothing to tell niches apart by.")
  }
  pcs <- stats::prcomp(mix[, varying, drop = FALSE], rank. = MAX_DIMS, tol = DIM_TOL)$x
  # One direction left (two cell types in all) needs mclust's one-dimensional
  # twin of "EEE": one variance shared by every niche.
  model <- if (ncol(pcs) == 1L) "E" else "EEE"

  ds$progress(0.5, sprintf("Fitting %d niches", n_niches))
  set.seed(params$seed)
  sample_rows <- if (nrow(pcs) > FIT_CELLS) sample.int(nrow(pcs), FIT_CELLS) else seq_len(nrow(pcs))
  fit <- Mclust(pcs[sample_rows, , drop = FALSE], G = n_niches, modelNames = model,
                verbose = FALSE)
  if (is.null(fit)) {
    pv_error("no_fit", sprintf(
      "mclust could not fit %d niches to these cells. Ask for fewer niches, or change the seed.",
      n_niches))
  }

  ds$progress(0.8, "Assigning every cell")
  niche <- predict(fit, newdata = pcs)$classification
  # Niche numbers by size, largest first; a niche no cell fell into is dropped.
  sizes <- tabulate(niche, nbins = n_niches)
  by_size <- order(sizes, decreasing = TRUE)
  niche <- match(niche, by_size)
  sizes <- sizes[by_size]
  found <- sum(sizes > 0)
  sizes <- sizes[seq_len(found)]
  names <- paste0("N", seq_len(found))

  # Each niche's own cells, by cell type.
  own <- types$codes[used]
  main <- vapply(seq_len(found), function(g) {
    count <- tabulate(own[niche == g], nbins = n_types)
    top <- head(order(count, decreasing = TRUE), 3L)
    top <- top[count[top] > 0]
    paste(sprintf("%s %.0f%%", types$levels[top], 100 * count[top] / sum(count)), collapse = ", ")
  }, character(1L))
  share <- 100 * sizes / sum(used)

  values <- rep(NA_character_, n)
  values[used] <- names[niche]
  lines <- c(
    sprintf(paste(
      "**%d niche%s** over %s cells, from the cell types of each cell's %d nearest cells.",
      "A Gaussian mixture (mclust, model %s) on %d principal components of those mixes was",
      "fitted to %s cells (seed %d) and then gave every cell its most likely niche."),
      found, if (found == 1L) "" else "s", format(sum(used), big.mark = ","), k, model,
      ncol(pcs), format(length(sample_rows), big.mark = ","), params$seed),
    "",
    sprintf("- Largest: %s, %.1f%% of cells (%s). Smallest: %s, %.1f%% (%s).",
            names[[1L]], share[[1L]], main[[1L]], names[[found]], share[[found]], main[[found]]),
    "- Niche numbers are ordered by size and carry no other order.")
  if (found < n_niches) {
    lines <- c(lines, sprintf("- %d of the %d niches asked for ended up empty.",
                              n_niches - found, n_niches))
  }

  list(
    summary = paste(lines, collapse = "\n"),
    # A factor with one entry per cell, in dataset order, and NA where a cell
    # has no niche. PointillViz takes the categories from its levels.
    labels = list(values = factor(values, levels = names), label = "Niche"),
    table = list(
      title = "Niches",
      rows = data.frame(Niche = names, Cells = sizes, `% of cells` = round(share, 2),
                        `Main cell types` = main, check.names = FALSE),
      sort_by = "Cells")
  )
}
