"""Native R tools: the R `ds` proxy, R's results, refusals, crashes, the channel, cancel.

These need Rscript with jsonlite on PATH, which the SDK's pixi environment has
(r-base, r-jsonlite). Every run goes through `launch.run` and the real
runner, as PointillViz runs an R tool.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from pointillviz_tool import Dataset, launch, r_bridge
from pointillviz_tool.fixture import CELL_TYPES, N_CELLS


def _r_ready() -> bool:
    rscript = shutil.which("Rscript")
    if not rscript:
        return False
    probe = subprocess.run([rscript, "--vanilla", "-e",
                            'quit(status = !requireNamespace("jsonlite", quietly = TRUE))'],
                           capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _r_ready(), reason="Rscript with jsonlite is not on PATH")


def _r_tool(tmp_path, code, name="rtool"):
    d = tmp_path / name
    d.mkdir()
    (d / "tool.R").write_text(code, encoding="utf-8")
    return d


def _manifest(outputs=("summary",), optional=()):
    return {"schema": 3, "id": "r-test", "name": "R Test", "entry": "tool.R",
            "outputs": list(outputs), "optional_outputs": list(optional)}


def _run(tool, dataset, params=None, outputs=("summary",), optional=(), **kw):
    return launch.run([sys.executable], str(tool), _manifest(outputs, optional), dataset,
                      params or {}, **kw)


# ===========================================================================
# The R ds is the Python one, 1-based
# ===========================================================================
TOUR = r'''
run <- function(ds, params) {
  xy <- ds$xy()
  types <- ds$cell_types()
  region <- ds$categorical("cat_region")
  nn <- ds$knn(5L)
  g <- ds$genes(c("Marker0", "Ubiquitous"), normalize = "raw")
  # The first placed cell of the filtered section: the only one with neighbours.
  first <- which(ds$section() == ds$section_filter & complete.cases(xy))[1]
  facts <- list(
    n_cells = ds$n_cells, name = ds$name, title = ds$title, device = ds$device,
    subset = ds$subset, section_filter = ds$section_filter,
    mode = ds$meta$spatial$mode,
    xy_dim = dim(xy), xy_complete = sum(complete.cases(xy)), x_sum = sum(xy[, 1], na.rm = TRUE),
    coords_cols = ncol(ds$coords()),
    sections = as.vector(table(ds$section())),
    n_genes = length(ds$gene_names()), first_genes = ds$gene_names()[1:3],
    has = c(ds$has_gene("Marker0"), ds$has_gene("Nope")),
    marker_raw = sum(ds$gene("Marker0", normalize = "raw")),
    marker_cp10k = sum(ds$gene("Marker0")),
    genes_dim = dim(g), genes_names = colnames(g), genes_sums = colSums(g),
    totals = sum(ds$total_counts()),
    fields = vapply(ds$categorical_fields(), function(f) f$name, ""),
    region_levels = region$levels, region_na = sum(is.na(region$codes)),
    type_levels = types$levels, type_na = sum(is.na(types$codes)),
    type_first = types$codes[1:20],
    type_names_first = types$levels[types$codes[1:20]],
    continuous_fields = vapply(ds$continuous_fields(), function(f) f$name, ""),
    area_sum = sum(ds$continuous("cont_cell_area")),
    knn_dim = dim(nn$idx), knn_na = sum(is.na(nn$idx)), knn_inf = sum(is.infinite(nn$dist)),
    first = first, knn_first = nn$idx[first, ], dist_first = nn$dist[first, 1],
    params = params
  )
  list(summary = as.character(jsonlite::toJSON(facts, auto_unbox = TRUE, digits = NA,
                                               na = "null")))
}
'''


def test_the_r_ds_is_the_python_one_counted_from_one(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, TOUR)
    reply = _run(tool, fixture_dir, {"k": 4, "genes": ["A"], "flag": True},
                 subset=[0, 5, 7], section_filter=1, device="cuda")
    assert reply["status"] == "ok", reply
    facts = json.loads(reply["result"]["summary"])
    ds = Dataset(fixture_dir, subset=[0, 5, 7], section_filter=1)
    xy = ds.xy
    labels, codes = ds.cell_types()
    idx, dist = ds.knn(5)
    raw = ds.genes(["Marker0", "Ubiquitous"], "raw")

    assert facts["n_cells"] == N_CELLS and facts["name"] == ds.name
    assert facts["title"] == ds.title and facts["device"] == "cuda"
    assert facts["subset"] == [1, 6, 8]                       # 1-based
    assert facts["section_filter"] == 2                        # 1-based
    assert facts["mode"] == "2d"
    assert facts["xy_dim"] == [N_CELLS, 2]
    assert facts["xy_complete"] == int(np.isfinite(xy).all(axis=1).sum())
    assert facts["x_sum"] == pytest.approx(float(np.nansum(xy[:, 0].astype(np.float64))))
    assert facts["coords_cols"] == 2
    assert facts["sections"] == [int((ds.section == s).sum()) for s in (0, 1)]
    assert facts["n_genes"] == len(ds.gene_names) and facts["first_genes"] == ds.gene_names[:3]
    assert facts["has"] == [True, False]
    assert facts["marker_raw"] == pytest.approx(float(ds.gene("Marker0", "raw").sum()))
    assert facts["marker_cp10k"] == pytest.approx(float(ds.gene("Marker0").astype(np.float64).sum()),
                                                  rel=1e-5)
    assert facts["genes_dim"] == [N_CELLS, 2]
    assert facts["genes_names"] == ["Marker0", "Ubiquitous"]
    assert facts["genes_sums"] == pytest.approx(raw.sum(axis=0).tolist())
    assert facts["totals"] == pytest.approx(float(ds.total_counts().sum()))
    assert facts["fields"] == [f["name"] for f in ds.categorical_fields()]
    region_levels, region_codes = ds.categorical("cat_region")
    assert facts["region_levels"] == region_levels
    assert facts["region_na"] == int((region_codes < 0).sum())
    assert facts["type_levels"] == labels == CELL_TYPES
    assert facts["type_na"] == int((codes < 0).sum())
    assert facts["type_first"] == [int(c) + 1 if c >= 0 else None for c in codes[:20]]
    assert facts["type_names_first"] == [labels[c] if c >= 0 else None for c in codes[:20]]
    assert facts["continuous_fields"] == [c["name"] for c in ds.continuous_fields()]
    assert facts["area_sum"] == pytest.approx(
        float(ds.continuous("cont_cell_area").astype(np.float64).sum()))
    assert facts["knn_dim"] == [N_CELLS, 5]
    assert facts["knn_na"] == int((idx < 0).sum())
    assert facts["knn_inf"] == int(np.isinf(dist).sum())
    first = int(np.flatnonzero((ds.section == 1) & np.isfinite(xy).all(axis=1))[0])
    assert facts["first"] == first + 1
    assert facts["knn_first"] == [int(i) + 1 for i in idx[first]]
    assert facts["dist_first"] == pytest.approx(float(dist[first, 0]), rel=1e-6)
    assert facts["params"] == {"k": 4, "genes": "A", "flag": True}


def test_ds_log_and_progress_reach_the_run(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  ds$log(c("two", "words")); ds$progress(0.5, "half")\n'
                             '  list(summary = "ok")\n}\n')
    reply = _run(tool, fixture_dir)
    assert reply["status"] == "ok", reply
    assert "two words" in reply["logs"]


# ===========================================================================
# What an R tool returns
# ===========================================================================
LABELS = r'''
run <- function(ds, params) {
  types <- ds$cell_types()
  chr <- types$levels[types$codes]
  mode <- params$mode
  labels <- switch(mode,
    factor = list(values = factor(chr, levels = rev(types$levels)), label = "Type"),
    character = list(values = chr),
    categories = list(values = chr, categories = c("T cell", types$levels[types$levels != "T cell"])),
    named_colors = list(values = factor(chr, levels = types$levels),
                        colors = setNames(rev(c("#e6194b", "#3cb44b", "#ffe119", "#4363d8",
                                                "#f58231", "#911eb4")), rev(types$levels))),
    numeric = list(values = types$codes - 1L, categories = types$levels),
    stray = list(values = chr, categories = types$levels[-1]),
    one = list(values = ifelse(is.na(chr), NA, "all"))
  )
  list(labels = labels)
}
'''


def _labels(tmp_path, fixture_dir, mode):
    tool = _r_tool(tmp_path, LABELS, name=f"labels-{mode}")
    return _run(tool, fixture_dir, {"mode": mode}, outputs=["labels"])


@pytest.fixture(scope="module")
def truth(fixture_info):
    labels, codes = Dataset(fixture_info["path"]).cell_types()
    return labels, [labels[c] if c >= 0 else None for c in codes]


def _decode(result):
    cats = result["labels"]["categories"]
    return [None if v is None else cats[v] for v in result["labels"]["values"]]


def test_labels_from_a_factor_keep_its_levels(tmp_path, fixture_dir, truth):
    reply = _labels(tmp_path, fixture_dir, "factor")
    assert reply["status"] == "ok", reply
    result = reply["result"]
    assert result["labels"]["categories"] == list(reversed(CELL_TYPES))
    assert result["labels"]["label"] == "Type"
    assert _decode(result) == truth[1]                 # codes are 0-based on the wire


def test_labels_from_characters_are_sorted_or_ordered_by_categories(tmp_path, fixture_dir, truth):
    result = _labels(tmp_path, fixture_dir, "character")["result"]
    present = sorted({t for t in truth[1] if t is not None})   # C order; Ghost has no cell
    assert result["labels"]["categories"] == present
    assert _decode(result) == truth[1]

    result = _labels(tmp_path, fixture_dir, "categories")["result"]
    assert result["labels"]["categories"][0] == "T cell"
    assert _decode(result) == truth[1]


def test_named_colours_follow_the_categories(tmp_path, fixture_dir):
    result = _labels(tmp_path, fixture_dir, "named_colors")["result"]
    assert result["labels"]["categories"] == CELL_TYPES
    assert result["labels"]["colors"] == ["#e6194b", "#3cb44b", "#ffe119", "#4363d8",
                                          "#f58231", "#911eb4"]


def test_numeric_codes_and_strays_are_refused_with_a_sentence(tmp_path, fixture_dir):
    reply = _labels(tmp_path, fixture_dir, "numeric")
    assert reply["code"] == "bad_result" and "factor or a character vector" in reply["message"]
    reply = _labels(tmp_path, fixture_dir, "stray")
    assert reply["code"] == "bad_result" and "Epithelial" in reply["message"]


SHAPES = r'''
run <- function(ds, params) {
  list(
    summary = "Épithélium — µm ≥ 1 \U0001F600",
    labels = list(values = rep("Épithélium", ds$n_cells)),
    table = data.frame(only = "x"),
    matrix = list(rows = "r", cols = "c", values = matrix(2.5, 1, 1)),
    series = list(lines = list(list(label = "one point", x = 1, y = 2)), rules = 0.5),
    radar = list(axes = c("a", "b", "c"), profiles = list(list(label = "p", values = c(0.1, 0.2, 0.3))))
  )
}
'''


def test_length_one_vectors_stay_arrays_and_text_survives_a_c_locale(tmp_path, fixture_dir,
                                                                     monkeypatch):
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    tool = _r_tool(tmp_path, SHAPES)
    reply = _run(tool, fixture_dir,
                 outputs=["summary", "labels", "table", "matrix", "series", "radar"])
    assert reply["status"] == "ok", reply
    result = reply["result"]
    assert result["summary"] == "Épithélium — µm ≥ 1 😀"
    assert result["labels"]["categories"] == ["Épithélium"]
    assert result["table"]["columns"] == ["only"] and result["table"]["rows"] == [["x"]]
    assert result["matrix"]["rows"] == ["r"] and result["matrix"]["values"] == [[2.5]]
    assert result["series"]["lines"][0]["x"] == [1.0] and result["series"]["rules"] == [0.5]
    assert result["radar"]["profiles"][0]["values"] == [0.1, 0.2, 0.3]


def test_a_data_frame_is_a_table(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  df <- data.frame(type = c("a", "b"), n = c(1.5, NA), ok = c(TRUE, FALSE))\n'
                             '  list(table = list(title = "T", rows = df))\n}\n')
    reply = _run(tool, fixture_dir, outputs=["table"])
    assert reply["status"] == "ok", reply
    table = reply["result"]["table"]
    assert table["title"] == "T" and table["columns"] == ["type", "n", "ok"]
    assert table["rows"] == [["a", 1.5, True], ["b", None, False]]


def test_an_infinity_is_refused_and_na_is_no_value(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  v <- rep(1, ds$n_cells); v[2] <- NA\n'
                             '  if (params$inf) v[3] <- Inf\n'
                             '  list(cells = list(values = v))\n}\n')
    reply = _run(tool, fixture_dir, {"inf": True}, outputs=["cells"])
    assert reply["code"] == "non_finite" and "cells$values" in reply["message"]
    reply = _run(tool, fixture_dir, {"inf": False}, outputs=["cells"])
    assert reply["status"] == "ok", reply
    assert reply["result"]["cells"]["values"][:3] == [1.0, None, 1.0]


# ===========================================================================
# Refusals, crashes, exits, and the channel
# ===========================================================================
def test_pv_error_is_a_refusal_not_a_crash(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  pv_error("too_few", "Only 3 cells; at least 30 are needed.")\n}\n')
    reply = _run(tool, fixture_dir)
    assert reply["code"] == "too_few"
    assert reply["message"] == "Only 3 cells; at least 30 are needed."
    assert "traceback" not in reply


def test_an_r_error_is_a_crash_with_rs_own_traceback(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'helper <- function(x) {\n'
                             '  stop("boom at ", x)\n'
                             '}\n'
                             'run <- function(ds, params) {\n'
                             '  helper(ds$n_cells)\n'
                             '}\n')
    reply = _run(tool, fixture_dir)
    assert reply["code"] == "crash"
    assert reply["message"] == f"Error in helper(ds$n_cells): boom at {N_CELLS}"
    assert "run(ds, params)" in reply["traceback"]
    assert "helper(ds$n_cells) at tool.R#5" in reply["traceback"]


def test_a_dataset_refusal_reaches_r_as_pv_error(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  if (params$catch) {\n'
                             '    got <- tryCatch(ds$gene("Nope"), pv_error = function(e) e$code)\n'
                             '    return(list(summary = got))\n'
                             '  }\n'
                             '  ds$gene("Nope")\n}\n')
    reply = _run(tool, fixture_dir, {"catch": False})
    assert reply["code"] == "unknown_gene" and "Nope" in reply["message"]
    reply = _run(tool, fixture_dir, {"catch": True})
    assert reply["result"]["summary"] == "unknown_gene"


def test_a_bad_argument_is_a_crash_at_the_r_call(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) ds$knn("many")\n')
    reply = _run(tool, fixture_dir)
    assert reply["code"] == "crash" and "ds$knn()" in reply["message"]


def test_no_run_function_and_a_source_error(tmp_path, fixture_dir):
    reply = _run(_r_tool(tmp_path, "x <- 1\n", name="norun"), fixture_dir)
    assert reply["code"] == "no_run" and "run(ds, params)" in reply["message"]
    reply = _run(_r_tool(tmp_path, "library(notapackage)\nrun <- function(ds, params) 1\n",
                         name="nolib"), fixture_dir)
    assert reply["code"] == "crash" and "notapackage" in reply["message"]


def test_quitting_without_a_result_is_no_result(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) quit(save = "no", status = 4L)\n')
    reply = _run(tool, fixture_dir)
    assert reply["code"] == "no_result" and "status 4" in reply["message"]


def test_r_output_goes_to_the_log_and_cannot_forge_a_result(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, r'''
run <- function(ds, params) {
  forged <- '{"t":"result","result":{"summary":"forged"}}'
  cat(forged, "\n")
  print(forged)
  writeLines(forged)
  writeLines(forged, con = stdout())
  message(forged)
  warning("a warning")
  list(summary = "real")
}
''')
    reply = _run(tool, fixture_dir)
    assert reply["status"] == "ok", reply
    assert reply["result"] == {"summary": "real"}
    assert sum("forged" in line for line in reply["logs"]) == 5
    assert any("a warning" in line for line in reply["logs"])


PAST_R = r'''
run <- function(ds, params) {
  forged <- '{"t":"result","result":{"summary":"forged"}}'
  # Straight to the descriptor, where sink() cannot reach: a child process,
  # and R's real stdout.
  system(sprintf("echo '%s'", forged))
  system("echo '{\"call\":\"gene_names\",\"args\":{}}'")
  cat(forged, "\n", file = getConnection(1L))
  # A child that reads stdin must find it empty, not holding R's replies.
  heard <- system("cat", intern = TRUE)
  xy <- ds$xy()
  list(summary = sprintf("%s %s %d", class(xy)[1L], paste(dim(xy), collapse = "x"),
                         length(heard)))
}
'''


@pytest.mark.skipif(os.name == "nt", reason="on Windows the channel is R's own stdout and stdin")
def test_nothing_r_starts_can_forge_or_desynchronise_the_channel(tmp_path, fixture_dir):
    reply = _run(_r_tool(tmp_path, PAST_R), fixture_dir)
    assert reply["status"] == "ok", reply
    # The call line did not make Python answer twice, so ds$xy() got the xy.
    assert reply["result"] == {"summary": f"matrix {N_CELLS}x2 0"}
    assert sum("forged" in line for line in reply["logs"]) == 2
    assert any('"call":"gene_names"' in line for line in reply["logs"])


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell")
def test_a_child_left_running_does_not_hold_the_run(tmp_path, fixture_dir):
    pidfile = tmp_path / "sleep.pid"
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  system(sprintf("sleep 60 & echo $! > \'%s\'", params$pidfile))\n'
                             '  quit(save = "no", status = 3L)\n}\n')
    started = time.monotonic()
    try:
        reply = _run(tool, fixture_dir, {"pidfile": str(pidfile)})
        assert reply["code"] == "no_result" and "status 3" in reply["message"]
        assert time.monotonic() - started < 20, "the run waited for the child R left behind"
    finally:
        try:
            os.kill(int(pidfile.read_text()), 9)
        except (OSError, ValueError):
            pass


def test_without_rscript_an_r_run_cannot_start(tmp_path, fixture_dir, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    tool = _r_tool(tmp_path, 'run <- function(ds, params) list(summary = "x")\n')
    reply = _run(tool, fixture_dir)
    assert reply["code"] == "no_subprocess" and "r-base" in reply["message"]


def test_a_sweep_runs_r_once_per_value(tmp_path, fixture_dir):
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  if (params$k == 4) pv_error("odd", "not 4")\n'
                             '  list(table = list(columns = "k", rows = list(list(params$k))))\n}\n')
    reply = _run(tool, fixture_dir, {}, outputs=["table"],
                 sweep={"param": "k", "values": [3, 4, 5]})
    assert reply["status"] == "ok", reply
    steps = reply["result"]["sweep"]["steps"]
    assert steps[0]["table"]["rows"] == [[3.0]] and steps[2]["table"]["rows"] == [[5.0]]
    assert steps[1] == {"error": {"code": "odd", "message": "not 4"}}


# ===========================================================================
# The probe, and cancel
# ===========================================================================
def test_the_probe_reports_what_run_is(tmp_path):
    good = r_bridge.probe(str(_r_tool(tmp_path, "run <- function(ds, params, ...) 1\n",
                                      name="good")), "tool.R")
    assert good["ok"] and good["has_run"] and good["callable"]
    assert good["formals"] == ["ds", "params", "..."] and good["dots"] is True

    one = r_bridge.probe(str(_r_tool(tmp_path, "run <- function(ds) 1\n", name="one")), "tool.R")
    assert one["formals"] == ["ds"] and one["dots"] is False

    value = r_bridge.probe(str(_r_tool(tmp_path, "run <- 3\n", name="value")), "tool.R")
    assert value["has_run"] and not value["callable"] and value["run_type"] == "numeric"

    broken = r_bridge.probe(str(_r_tool(tmp_path, "run <- function(ds, params) {\n",
                                        name="broken")), "tool.R")
    assert not broken["ok"] and broken["phase"] == "import"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(os.name == "nt", reason="checks POSIX process groups")
def test_cancel_stops_r_too(tmp_path, fixture_dir):
    pidfile = tmp_path / "r.pid"
    tool = _r_tool(tmp_path, 'run <- function(ds, params) {\n'
                             '  writeLines(as.character(Sys.getpid()), params$pidfile)\n'
                             '  Sys.sleep(600)\n}\n')
    out = {}
    worker = threading.Thread(target=lambda: out.update(reply=_run(
        tool, fixture_dir, {"pidfile": str(pidfile)}, run_id="cancel-r")))
    worker.start()
    deadline = time.monotonic() + 60
    while not (pidfile.exists() and pidfile.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(0.05)
    r_pid = int(pidfile.read_text())
    assert launch.cancel("cancel-r") is True
    worker.join(30)
    assert out["reply"]["code"] == "cancelled"
    deadline = time.monotonic() + 10
    while _alive(r_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(r_pid)
