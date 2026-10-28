# {{NAME}}: the most common cell type among each cell's nearest neighbours,
# the starting point for a native R tool.
#
# `run(ds, params)` is the only thing PointillViz looks for. `ds` is the
# dataset the user is looking at; `params` holds the values of the parameters
# declared in plugin.json, already checked against their declared range.
# What `ds` hands back is 1-based, with NA for "no value": the codes of
# ds$cell_types() index its levels, and the neighbours of ds$knn() index cells.

run <- function(ds, params) {
  k <- params$k
  types <- ds$cell_types()          # list(levels = chr, codes = int), NA = unlabelled
  nn <- ds$knn(k)                   # list(idx = n x k int matrix, dist), NA = no neighbour

  # The cell type of every neighbour, then how often each type occurs per cell.
  neighbour <- matrix(types$codes[as.vector(nn$idx)], nrow = nrow(nn$idx))
  counts <- vapply(seq_along(types$levels),
                   function(t) rowSums(neighbour == t, na.rm = TRUE),
                   numeric(nrow(neighbour)))
  counts <- matrix(counts, nrow = nrow(neighbour))
  top <- max.col(counts, ties.method = "first")
  top[rowSums(counts) == 0] <- NA   # no position, or no labelled neighbour

  majority <- factor(types$levels[top], levels = types$levels)
  labelled <- sum(!is.na(majority))
  ds$log(sprintf("%d of %d cells have a labelled neighbour among their %d nearest",
                 labelled, ds$n_cells, k))

  # THE BRANCH TO COPY INTO YOUR OWN TOOL. A map with no labels and a real
  # result look the same once painted, so a tool that has nothing to say must
  # SAY so. pv_error() refuses in words the user reads: name what failed and
  # what to change. Any other error is reported as a crash with its call
  # stack, which is right for a bug and wrong for "nothing qualified".
  if (labelled == 0) {
    pv_error("no_neighbours", sprintf(
      "No cell has a labelled neighbour among its %d nearest. This needs cell positions and a cell-type annotation.",
      k))
  }

  share <- sort(table(majority), decreasing = TRUE)
  list(
    summary = sprintf(
      "**%s of %s cells** are labelled by the most common cell type among their %d nearest neighbours. The most frequent is **%s** (%.0f%% of them).",
      format(labelled, big.mark = ","), format(ds$n_cells, big.mark = ","), k,
      names(share)[[1]], 100 * share[[1]] / labelled),
    # A factor with one entry per cell, in dataset order, and NA where a cell
    # has no label. PointillViz takes the categories from its levels.
    labels = list(values = majority, label = "Neighbourhood majority")
  )
}
