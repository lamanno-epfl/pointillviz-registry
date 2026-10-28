# The R half of a native R tool run, started by pointillviz_tool/r_bridge.py:
#
#     Rscript --vanilla runner.R <to python> <from python> <tool dir> <entry> [--probe]
#
# The tool is a file defining `run <- function(ds, params)`. This script
# sources it, builds `ds` (a list whose functions are requests to the Python
# process holding the dataset), calls run(ds, params), and sends the result
# back as JSON. With --probe it only sources the file and reports what `run`
# is, for the validator's entry gate.
#
# The channel to Python is two pipes of its own, named by the first two
# arguments (/dev/fd paths). This process's stdout leads to PointillViz's log
# and its stdin is empty, so nothing a tool prints, by any means (print, cat,
# system(), a child process, compiled code), lands in the protocol, and nothing
# it starts can read a reply. sink() also sends what R itself prints to stderr,
# so it reaches the log as it happens. Where a child cannot be handed extra
# descriptors (Windows) both arguments are "-", and the channel is this
# process's own stdout, kept as a connection before the sink, and stdin. Every
# line sent is plain ASCII (anything else travels as a \u escape), so no locale
# can mangle it. Replies arrive one JSON line per request; r_bridge.py
# documents the wire.
#
# This file is ASCII on purpose: Rscript parses it in the session's locale.

local({
  args <- commandArgs(trailingOnly = TRUE)
  tool_dir <- args[[3L]]
  entry <- args[[4L]]
  probe <- identical(args[5L], "--probe")

  # raw = TRUE: a pipe is not a file, and R would otherwise first read from
  # it to look for compression.
  channel <- if (identical(args[[1L]], "-")) stdout() else file(args[[1L]], "w", raw = TRUE)
  sink(stderr())
  options(warn = 1L)

  # A UTF-8 session where the machine has one. A tool file is UTF-8, and in a
  # C locale R can neither source it nor print what it computes from it.
  if (!isTRUE(l10n_info()[["UTF-8"]])) {
    for (candidate in c("C.UTF-8", "en_US.UTF-8", "UTF-8")) {
      if (nzchar(suppressWarnings(Sys.setlocale("LC_CTYPE", candidate)))) break
    }
  }

  ascii <- function(text) {
    hits <- gregexpr("[^\\x{01}-\\x{7f}]", text, perl = TRUE)
    if (hits[[1L]][1L] == -1L) return(text)
    regmatches(text, hits) <- list(vapply(regmatches(text, hits)[[1L]], function(ch) {
      code <- utf8ToInt(ch)
      if (is.na(code)) return("?")
      if (code < 65536L) return(sprintf("\\u%04x", code))
      code <- code - 65536L
      sprintf("\\u%04x\\u%04x", 55296L + code %/% 1024L, 56320L + code %% 1024L)
    }, ""))
    text
  }

  send_text <- function(text) {
    cat(ascii(text), "\n", sep = "", file = channel)
    flush(channel)
  }

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    why <- "The R package jsonlite is not installed. Add r-jsonlite to the tool's pixi.toml."
    send_text(if (probe) sprintf('{"probe":true,"ok":false,"phase":"import","error":"%s"}', why)
              else sprintf('{"t":"error","code":"crash","message":"%s"}', why))
    quit(save = "no", status = 1L)
  }

  to_json <- function(x) {
    jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA, na = "null", null = "null",
                     dataframe = "values")
  }
  send <- function(x) send_text(to_json(x))

  # ---- refusals ------------------------------------------------------------
  # pv_error(code, message) is how a tool refuses an input in words: the run
  # ends, and the user reads `message`. Anything else a tool signals is a
  # crash, reported with its call stack. Attached rather than defined in the
  # tool's environment, so rm(list = ls()) in a tool cannot remove it.
  pv_error <- function(code, message) {
    stop(structure(class = c("pv_error", "error", "condition"),
                   list(message = paste(message, collapse = " "),
                        call = sys.call(-1L), code = as.character(code))))
  }
  attach(list(pv_error = pv_error), name = "pointillviz", warn.conflicts = FALSE)

  # Evaluate `expr`, keeping the call stack of an error for its traceback.
  stack_at_error <- NULL
  caught <- function(expr) {
    stack_at_error <<- NULL
    tryCatch(
      list(ok = TRUE, value = withCallingHandlers(
        expr, error = function(e) stack_at_error <<- sys.calls())),
      error = function(e) list(ok = FALSE, error = e, calls = stack_at_error))
  }

  describe <- function(e) {
    call <- conditionCall(e)
    if (is.null(call)) return(conditionMessage(e))
    sprintf("Error in %s: %s", paste(deparse(call, nlines = 1L), collapse = ""),
            conditionMessage(e))
  }

  # The frames between this script and the error, one line each, with the
  # file and line of every call that has a source reference.
  traceback_text <- function(calls) {
    if (!length(calls)) return("")
    ours <- which(vapply(calls, function(cl) identical(cl[[1L]], quote(withCallingHandlers)),
                         logical(1L)))
    frames <- calls[-c(seq_len(if (length(ours)) min(ours) else 0L), length(calls))]
    frames <- Filter(function(cl) !identical(cl[[1L]], quote(.handleSimpleError)), frames)
    if (!length(frames)) return("")
    lines <- vapply(seq_along(frames), function(i) {
      cl <- frames[[i]]
      text <- paste(deparse(cl, nlines = 1L, width.cutoff = 80L), collapse = "")
      ref <- attr(cl, "srcref")
      file <- if (is.null(ref)) character(0) else utils::getSrcFilename(ref)
      if (length(file)) {
        text <- sprintf("%s at %s#%d", text, basename(file), utils::getSrcLocation(ref, "line"))
      }
      sprintf("%d: %s", i, text)
    }, "")
    paste(lines, collapse = "\n")
  }

  failed <- function(outcome) {
    if (outcome$ok) return(FALSE)
    e <- outcome$error
    if (inherits(e, "pv_error")) {
      send(list(t = "error", code = e$code, message = conditionMessage(e)))
    } else {
      send(list(t = "error", code = "crash", message = describe(e),
                traceback = traceback_text(outcome$calls)))
    }
    TRUE
  }

  # ---- the tool ------------------------------------------------------------
  tool <- new.env(parent = globalenv())
  started <- proc.time()[["elapsed"]]
  sourced <- caught(source(file.path(tool_dir, entry), local = tool, encoding = "UTF-8",
                           keep.source = TRUE))
  run <- if (sourced$ok) get0("run", envir = tool, inherits = FALSE) else NULL

  if (probe) {
    if (!sourced$ok) {
      send(list(probe = TRUE, ok = FALSE, phase = "import", error = describe(sourced$error),
                traceback = traceback_text(sourced$calls)))
    } else {
      formal <- if (is.function(run)) formals(if (is.primitive(run)) args(run) else run)
      send(list(probe = TRUE, ok = TRUE, has_run = !is.null(run), callable = is.function(run),
                run_type = class(run)[[1L]], formals = I(as.character(names(formal))),
                dots = "..." %in% names(formal),
                seconds = proc.time()[["elapsed"]] - started))
    }
    quit(save = "no", status = 0L)
  }
  if (failed(sourced)) quit(save = "no", status = 1L)
  if (!is.function(run)) {
    send(list(t = "error", code = "no_run",
              message = sprintf("%s does not define a run(ds, params) function.", entry)))
    quit(save = "no", status = 1L)
  }

  # ---- ds: every method is a request to the Python side ---------------------
  input <- if (identical(args[[2L]], "-")) file("stdin") else file(args[[2L]], raw = TRUE)
  open(input)

  read_array <- function(spec) {
    shape <- as.integer(unlist(spec[["shape"]]))
    con <- file(spec[["file"]], "rb")
    on.exit({
      close(con)
      unlink(spec[["file"]])
    })
    x <- readBin(con, what = if (identical(spec[["dtype"]], "i4")) "integer" else "double",
                 n = prod(shape), size = if (identical(spec[["dtype"]], "f8")) 8L else 4L,
                 endian = "little")
    if (length(shape) > 1L) dim(x) <- shape
    x
  }

  request <- function(method, args = NULL) {
    send(list(call = method,
              args = if (length(args)) args else structure(list(), names = character(0))))
    line <- readLines(input, n = 1L, encoding = "UTF-8")
    if (!length(line)) stop("PointillViz stopped answering.", call. = FALSE)
    reply <- jsonlite::fromJSON(line, simplifyVector = TRUE, simplifyDataFrame = FALSE,
                                simplifyMatrix = FALSE)
    if (!isTRUE(reply[["ok"]])) {
      # A bad argument is a bug at the R call that made it; anything else is
      # the dataset refusing, which reaches the user as a sentence.
      if (identical(reply[["code"]], "bad_call")) stop(reply[["message"]], call. = FALSE)
      pv_error(reply[["code"]], reply[["message"]])
    }
    if (!is.null(reply[["file"]])) return(read_array(reply))
    value <- reply[["value"]]
    for (key in names(reply[["arrays"]])) value[[key]] <- read_array(reply[["arrays"]][[key]])
    value
  }

  make_ds <- function(info) {
    list(
      n_cells = info[["n_cells"]],
      name = info[["name"]],
      title = info[["title"]],
      meta = info[["meta"]],
      device = info[["device"]],
      subset = info[["subset"]],
      section_filter = info[["section_filter"]],
      xy = function() request("xy"),
      coords = function() request("coords"),
      section = function() request("section"),
      gene_names = function() request("gene_names"),
      has_gene = function(name) request("has_gene", list(name = name)),
      gene = function(name, normalize = "cp10k") {
        request("gene", list(name = name, normalize = normalize))
      },
      genes = function(names, normalize = "cp10k") {
        out <- request("genes", list(names = I(as.character(names)), normalize = normalize))
        colnames(out) <- as.character(names)
        out
      },
      total_counts = function() request("total_counts"),
      categorical_fields = function() request("categorical_fields"),
      categorical = function(name = NULL) request("categorical", list(name = name)),
      cell_types = function() request("cell_types"),
      continuous_fields = function() request("continuous_fields"),
      continuous = function(name) request("continuous", list(name = name)),
      knn = function(k = 15L) request("knn", list(k = k)),
      log = function(message) send(list(t = "log", m = paste(message, collapse = " "))),
      progress = function(fraction, message = NULL) {
        send(list(t = "progress", f = fraction, m = message))
      }
    )
  }

  # ---- the result ------------------------------------------------------------
  as_labels <- function(lab) {
    if (!is.list(lab) || is.null(lab[["values"]])) return(lab)
    values <- lab[["values"]]
    if (is.numeric(values)) {
      pv_error("bad_result", paste(
        "labels$values must be a factor or a character vector in R, with NA where a",
        "cell has no label; the categories are taken from it, so there are no codes",
        "to write by hand."))
    }
    categories <- lab[["categories"]]
    if (is.null(categories)) {
      categories <- if (is.factor(values)) levels(values)
                    else sort(unique(as.character(values[!is.na(values)])), method = "radix")
    }
    categories <- as.character(unlist(categories))
    values <- as.character(values)
    codes <- match(values, categories) - 1L
    stray <- !is.na(values) & is.na(codes)
    if (any(stray)) {
      pv_error("bad_result", sprintf("labels$values holds \"%s\", which is not one of labels$categories.",
                                     values[stray][[1L]]))
    }
    colors <- lab[["colors"]]
    if (!is.null(colors)) {
      colors <- unlist(colors)
      if (!is.null(names(colors))) colors <- unname(colors[categories])
      lab[["colors"]] <- I(as.character(colors))
    }
    lab[["values"]] <- I(codes)
    lab[["categories"]] <- I(categories)
    lab
  }

  as_table <- function(tab) {
    if (is.data.frame(tab)) tab <- list(columns = names(tab), rows = tab)
    if (!is.list(tab)) return(tab)
    if (is.data.frame(tab[["rows"]]) && is.null(tab[["columns"]])) {
      tab[["columns"]] <- names(tab[["rows"]])
    }
    if (!is.null(tab[["columns"]])) tab[["columns"]] <- I(as.character(tab[["columns"]]))
    tab
  }

  # Vectors that must stay JSON arrays even at length one, which auto_unbox
  # would otherwise turn into a scalar.
  arrays <- function(obj, keys) {
    if (!is.list(obj)) return(obj)
    for (key in keys) {
      x <- obj[[key]]
      if (is.atomic(x) && !is.null(x) && is.null(dim(x)) && !inherits(x, "AsIs")) obj[[key]] <- I(x)
    }
    obj
  }

  check_finite <- function(x, path) {
    if (is.list(x)) {
      keys <- names(x)
      for (i in seq_along(x)) {
        key <- if (is.null(keys) || !nzchar(keys[[i]])) sprintf("[[%d]]", i)
               else paste0(if (nzchar(path)) "$" else "", keys[[i]])
        check_finite(x[[i]], paste0(path, key))
      }
    } else if (is.double(x) && any(is.infinite(x))) {
      pv_error("non_finite", sprintf("%s contains infinities. Use NA for missing values.", path))
    }
  }

  as_result <- function(res) {
    if (!is.list(res) || is.data.frame(res)) return(res)
    if (!is.null(res[["labels"]])) res[["labels"]] <- as_labels(res[["labels"]])
    if (!is.null(res[["table"]])) res[["table"]] <- as_table(res[["table"]])
    res[["cells"]] <- arrays(res[["cells"]], c("values", "domain"))
    res[["vectors"]] <- arrays(res[["vectors"]], c("x", "y", "u", "v", "value", "domain"))
    res[["matrix"]] <- arrays(res[["matrix"]], c("rows", "cols"))
    res[["radar"]] <- arrays(res[["radar"]], c("axes", "domain"))
    if (is.list(res[["radar"]]) && is.list(res[["radar"]][["profiles"]])) {
      res[["radar"]][["profiles"]] <- lapply(res[["radar"]][["profiles"]], arrays, "values")
    }
    res[["series"]] <- arrays(res[["series"]], "rules")
    if (is.list(res[["series"]]) && is.list(res[["series"]][["lines"]])) {
      res[["series"]][["lines"]] <- lapply(res[["series"]][["lines"]], arrays, c("x", "y"))
    }
    check_finite(res, "")
    res
  }

  # ---- the run -------------------------------------------------------------
  init <- caught(request("init"))
  if (failed(init)) quit(save = "no", status = 1L)
  ds <- make_ds(init$value)
  params <- init$value[["params"]]
  if (is.null(params)) params <- list()

  outcome <- caught(run(ds, params))
  if (failed(outcome)) quit(save = "no", status = 1L)
  encoded <- caught(to_json(as_result(outcome$value)))
  if (failed(encoded)) quit(save = "no", status = 1L)
  send_text(paste0('{"t":"result","result":', encoded$value, "}"))
})
