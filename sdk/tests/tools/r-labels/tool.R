# A clean native R tool: every gate passes on it.
run <- function(ds, params) {
  types <- ds$cell_types()
  nn <- ds$knn(params$k)
  lab <- factor(types$levels[types$codes], levels = types$levels)
  with_neighbours <- sum(rowSums(!is.na(nn$idx)) > 0)
  ds$log(sprintf("%d cells have neighbours at k=%d", with_neighbours, params$k))
  counts <- as.vector(table(lab))
  list(summary = sprintf("%d labelled cells, k=%d.", sum(!is.na(lab)), params$k),
       labels = list(values = lab, label = "Cell type"),
       table = data.frame(type = types$levels, cells = counts))
}
