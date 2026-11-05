# Seurat niches: tissue niches from the cell types around each cell.
#
# Seurat's BuildNicheAssay counts the cell types among each cell's nearest cells,
# scales those counts and groups the cells with k-means. This file computes none
# of that. It loads the authors' package, puts the cell types and positions into
# a Seurat object as their Xenium vignette has them, and calls the function. What
# is here is the way in and the way out: cells with no type left out, as the
# vignette does, sections moved apart so that each keeps its own neighbours,
# sentences where the package would stop with an error, and the niches back on
# the map with a description of each.
#
# What `ds` hands back is 1-based, with NA for "no value".

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(ps)
})

SEED <- 42L           # the seed Seurat's own functions default to
STARTS <- 30L         # BuildNicheAssay's nstart, which it does not let a caller set
#: Bytes at the peak, measured: the libraries, and dense matrices of cells x groups
#: (the tallies, their sum, the assay's scaled copy and k-means' own), at 1.8 GB for
#: 135,231 cells x 73 groups, 1.4 GB for 235,803 x 19 and 29 GB for 708,889 x 327.
BASE_BYTES <- 1e9
DENSE_COPIES <- 15
MAX_SHOWN <- 60L      # cell types in the composition matrix
MAX_MATRIX <- 10000L  # the matrix output's limit on cells
RESULT_BUDGET <- 7500000  # views and files share an 8 MiB budget per result

many <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)
cells_were <- function(n) sprintf("%s cell%s", many(n), if (n == 1L) " was" else "s were")

run <- function(ds, params) {
  niches_k <- as.integer(params$niches_k)
  neighbors_k <- as.integer(params$neighbors_k)
  iter_max <- as.integer(params$iter_max)
  field <- params$field
  if (is.null(field) || !nzchar(field)) field <- NULL

  ds$progress(0.02, "Reading the cell types")
  groups <- ds$categorical(field)
  what <- group_words(ds, field)
  xy <- ds$xy()
  section <- ds$section()
  notes <- character()

  asked <- rep(TRUE, ds$n_cells)
  if (!is.null(ds$section_filter)) asked <- asked & section == ds$section_filter
  if (!is.null(ds$subset) && length(ds$subset)) asked <- asked & seq_len(ds$n_cells) %in% ds$subset
  placed <- is.finite(xy[, 1L]) & is.finite(xy[, 2L])
  typed <- !is.na(groups$codes) & nzchar(groups$levels[groups$codes])
  typed[is.na(typed)] <- FALSE
  if (!any(asked)) pv_error("too_few_cells", "There are no cells to run on in this section. Isolate another section, or select cells in this one.")
  keep <- asked & placed & typed
  if (any(asked & !placed)) notes <- c(notes, sprintf("%s left out: no coordinates.", cells_were(sum(asked & !placed))))
  if (any(asked & placed & !typed)) notes <- c(notes, sprintf("%s left out: no %s.", cells_were(sum(asked & placed & !typed)), what$one))
  if (!any(keep)) pv_error("no_cells", sprintf("None of these cells has both coordinates and %s. Choose another column.", what$a))

  # BuildNicheAssay finds neighbours over all its cells at once. A section needs
  # more cells than the neighbourhood holds, or its cells would reach into another.
  sizes <- table(section[keep])
  small <- as.integer(names(sizes)[sizes <= neighbors_k])
  names_of <- section_names(ds)
  for (s in small) {
    notes <- c(notes, sprintf("Section %s left out: its %s cells are too few for %d neighbours.",
                              names_of(s), many(sizes[[as.character(s)]]), neighbors_k))
  }
  keep <- keep & !(section %in% small)
  if (!any(keep)) {
    pv_error("too_few_cells", sprintf(
      "No section here has more than the %d cells a neighbourhood holds. Select more cells, or lower Spatial neighbours.",
      neighbors_k))
  }
  rows <- which(keep)
  labels <- groups$levels[groups$codes[rows]]
  present <- sort(unique(labels))
  if (length(present) < 2L) {
    pv_error("one_group", sprintf(
      "These cells all have one %s, so their surroundings cannot differ. Choose another column, or a wider selection.",
      what$one))
  }
  enough_memory(length(rows), length(present))

  ds$progress(0.08, "Building the Seurat object")
  ids <- paste0("cell", rows)
  placed_xy <- set_apart(xy[rows, , drop = FALSE], section[rows])
  # BuildNicheAssay reads cell types and positions and no expression, so the object
  # holds one empty row of counts: reading the genes would change nothing but the memory.
  empty <- sparseMatrix(i = integer(0), p = rep(0L, length(rows) + 1L), x = numeric(0),
                        dims = c(1L, length(rows)), dimnames = list("none", ids))
  object <- suppressWarnings(CreateSeuratObject(
    empty, meta.data = data.frame(group = labels, row.names = ids)))
  object[["fov"]] <- CreateFOV(data.frame(x = placed_xy[, 1L], y = placed_xy[, 2L], cell = ids),
                               type = "centroids", assay = "RNA")

  ds$progress(0.15, "Seurat is building the niches")
  set.seed(SEED)
  built <- quietly(ds, BuildNicheAssay(object, fov = "fov", group.by = "group",
                                       niches.k = niches_k, neighbors.k = neighbors_k,
                                       iter.max = iter_max), niches_k)
  object <- built$value
  ds$progress(0.9, "Describing the niches")

  raw <- as.integer(object$niches[ids])
  counted <- tabulate(raw, nbins = max(raw))
  by_size <- order(counted, decreasing = TRUE)
  niche <- match(raw, by_size)
  found <- sum(counted > 0L)
  sizes <- counted[by_size][seq_len(found)]
  niche_names <- paste0("N", seq_len(found))
  share <- 100 * sizes / length(rows)
  composition <- table(factor(niche_names[niche], levels = niche_names), factor(labels, levels = present))
  composition <- 100 * unclass(composition) / sizes
  overall <- order(tabulate(match(labels, present), nbins = length(present)), decreasing = TRUE)
  composition <- composition[, overall, drop = FALSE]
  main <- vapply(seq_len(found), function(n) {
    top <- head(order(composition[n, ], decreasing = TRUE), 3L)
    top <- top[composition[n, top] > 0]
    paste(sprintf("%s %.0f%%", colnames(composition)[top], composition[n, top]), collapse = ", ")
  }, character(1L))
  if (built$stopped > 0L) {
    notes <- c(notes, sprintf(
      "k-means stopped at its limit of %d iterations in %d of its %d starts. Raising K-means iterations lets them finish.",
      iter_max, built$stopped, STARTS))
  }
  if (built$transfer > 0L) {
    notes <- c(notes, sprintf(
      "k-means cut its transfer stage short in %d of its %d starts, as R's does on large data. It keeps the best start.",
      built$transfer, STARTS))
  }
  n_sections <- length(unique(section[rows]))
  if (n_sections > 1L) {
    notes <- c(notes, sprintf("%d sections share these niches; each cell's neighbours come from its own section.", n_sections))
  }

  values <- rep(NA_character_, ds$n_cells)
  values[rows] <- niche_names[niche]
  shown <- min(ncol(composition), MAX_SHOWN, MAX_MATRIX %/% found)
  download <- niche_file(rows, niche_names[niche])
  if (is.null(download)) notes <- c(notes, "The CSV of every cell's niche is larger than a result can carry, and was left out.")
  result <- list(
    summary = summary_text(rows, found, what, length(present), neighbors_k, iter_max, niche_names,
                           share, main, niches_k, notes),
    labels = list(values = factor(values, levels = niche_names), label = "Seurat niche"),
    matrix = list(
      title = sprintf("Composition of each niche (%% of the niche's cells, by %s)", what$one),
      rows = I(niche_names), cols = I(colnames(composition)[seq_len(shown)]),
      values = unname(composition[, seq_len(shown), drop = FALSE]), colormap = "viridis"),
    table = list(
      title = "Niches",
      rows = stats::setNames(data.frame(niche_names, sizes, round(share, 2), main),
                             c("Niche", "Cells", "% of cells", what$main)),
      sort_by = "Cells"))
  if (!is.null(download)) result$files <- list(download)
  ds$progress(1)
  result
}

# ---- the way in ----------------------------------------------------------------------
group_words <- function(ds, field) {
  fields <- ds$categorical_fields()
  name <- if (is.data.frame(fields)) fields$name else vapply(fields, function(f) f$name, "")
  label <- if (is.data.frame(fields)) fields$label else vapply(fields, function(f) f$label, "")
  at <- if (is.null(field)) 1L else match(field, c(name, label))
  if (is.na(at)) at <- 1L
  at <- (at - 1L) %% length(name) + 1L
  if (identical(name[[at]], "cell_type_idx")) {
    list(one = "cell type", a = "a cell type", many = "cell types", main = "Main cell types")
  } else {
    list(one = sprintf("level of %s", label[[at]]), a = sprintf("a level of %s", label[[at]]),
         many = sprintf("levels of %s", label[[at]]), main = "Main levels")
  }
}

section_names <- function(ds) {
  sections <- ds$meta$sections
  if (is.data.frame(sections)) {
    ids <- sections$section_id
    codes <- if (is.null(sections$index)) seq_along(ids) - 1L else sections$index
  } else {
    ids <- vapply(sections, function(s) as.character(s$section_id %||% ""), "")
    codes <- vapply(seq_along(sections), function(i) as.integer(sections[[i]]$index %||% (i - 1L)), 1L)
  }
  function(code) {
    at <- match(code - 1L, codes)
    if (is.na(at) || !nzchar(ids[[at]])) as.character(code) else ids[[at]]
  }
}

# Sections sit on one canvas and can overlap. On a grid wider than any section,
# a cell's nearest cells all lie in its own section, since each section has more
# cells than a neighbourhood holds. Moving a section changes no distance within it.
set_apart <- function(xy, section) {
  codes <- sort(unique(section))
  if (length(codes) < 2L) return(xy)
  widest <- max(vapply(codes, function(s) {
    sqrt(sum(apply(xy[section == s, , drop = FALSE], 2L, function(v) diff(range(v)))^2))
  }, 0))
  pitch <- 2 * widest + 1
  columns <- ceiling(sqrt(length(codes)))
  for (i in seq_along(codes)) {
    where <- section == codes[[i]]
    xy[where, 1L] <- xy[where, 1L] - min(xy[where, 1L]) + ((i - 1L) %% columns) * pitch
    xy[where, 2L] <- xy[where, 2L] - min(xy[where, 2L]) + ((i - 1L) %/% columns) * pitch
  }
  xy
}

enough_memory <- function(cells, groups) {
  needed <- BASE_BYTES + DENSE_COPIES * 8 * cells * groups
  free <- ps_system_memory()[["avail"]]
  if (needed > free) {
    pv_error("not_enough_memory", sprintf(
      "Seurat would need about %.0f GB for %s cells and %s groups, and this computer has %.0f GB free. Select fewer cells, isolate a section or choose a coarser column.",
      needed / 1e9, many(cells), many(groups), free / 1e9))
  }
}

# The function's progress notes stay out of the log. k-means warns once per start
# when it stops early, so those are counted for the summary, and its refusal of more
# niches than these cells can make is said in a sentence.
quietly <- function(ds, expr, niches_k) {
  stopped <- 0L
  transfer <- 0L
  value <- tryCatch(
    withCallingHandlers({
      out <- NULL
      utils::capture.output(out <- suppressMessages(expr))
      out
    }, warning = function(w) {
      text <- conditionMessage(w)
      if (grepl("did not converge", text, fixed = TRUE)) stopped <<- stopped + 1L
      else if (grepl("Quick-TRANSfer", text, fixed = TRUE)) transfer <<- transfer + 1L
      else if (!grepl("Feature names cannot have", text, fixed = TRUE)) ds$log(paste("Seurat:", text))
      invokeRestart("muffleWarning")
    }),
    error = function(e) {
      text <- conditionMessage(e)
      if (grepl("distinct data points|empty cluster", text)) {
        pv_error("too_many_niches", sprintf(
          "k-means cannot make %d niches from these cells: their surroundings differ too little. Lower Number of niches.",
          niches_k))
      }
      stop(e)
    })
  list(value = value, stopped = stopped, transfer = transfer)
}

# ---- the way out ----------------------------------------------------------------------
summary_text <- function(rows, found, what, groups, neighbors_k, iter_max, names, share, main, niches_k, notes) {
  lines <- c(
    sprintf("**%d niche%s** over %s cells, from the %s %s around each cell, by Seurat %s's BuildNicheAssay: %d spatial neighbours, k-means with %d starts of at most %d iterations.",
            found, if (found == 1L) "" else "s", many(length(rows)), many(groups), what$many,
            as.character(utils::packageVersion("Seurat")), neighbors_k, STARTS, iter_max),
    "",
    sprintf("- Largest: %s, %.1f%% of cells (%s). Smallest: %s, %.1f%% (%s).",
            names[[1L]], share[[1L]], main[[1L]], names[[found]], share[[found]], main[[found]]),
    "- Niche numbers are ordered by size and carry no other order.")
  if (found < niches_k) lines <- c(lines, sprintf("- %d of the %d niches asked for ended up empty.", niches_k - found, niches_k))
  paste(c(lines, if (length(notes)) paste0("- ", notes)), collapse = "\n")
}

niche_file <- function(rows, niches) {
  text <- paste0(c("cell_index,niche", paste0(rows - 1L, ",", niches), ""), collapse = "\n")
  bytes <- charToRaw(text)
  if (4 * length(bytes) / 3 >= RESULT_BUDGET) return(NULL)
  list(name = "seurat_niches.csv", media_type = "text/csv",
       base64 = gsub("[[:space:]]", "", jsonlite::base64_enc(bytes)))
}
