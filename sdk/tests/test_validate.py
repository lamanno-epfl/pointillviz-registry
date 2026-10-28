"""The gates: each one passes a clean tool and fails, for one defect, the tool that has it.

Tools run in the current interpreter (`env="app"`, the default for a tool
without a pixi.toml) unless a test is about the environment gate, which uses
the real pixi CLI on small temporary workspaces and skips without pixi.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import pytest

from pointillviz_tool import bundle, envs, launch
from pointillviz_tool.dataset import PluginError
from pointillviz_tool.validate import GATE_ORDER, _scan_r, format_report, validate

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "tools")
GATES = [gid for gid, _required in GATE_ORDER]


def _r_ready() -> bool:
    rscript = shutil.which("Rscript")
    return bool(rscript) and subprocess.run(
        [rscript, "--vanilla", "-e", 'quit(status = !requireNamespace("jsonlite", quietly = TRUE))'],
        capture_output=True).returncode == 0


needs_r = pytest.mark.skipif(not _r_ready(), reason="Rscript with jsonlite is not on PATH")


def _pixi_or_skip():
    try:
        return envs.pixi_exe()
    except PluginError:
        pytest.skip("pixi is not installed")


def _copy(name, tmp_path):
    dest = tmp_path / name
    shutil.copytree(os.path.join(TOOLS, name), dest)
    return dest


def _tool(tmp_path, code=None, manifest=None, name="tool"):
    """A copy of the clean Python test tool, with its code or manifest changed."""
    d = tmp_path / name
    shutil.copytree(os.path.join(TOOLS, "py-cells"), d)
    if code is not None:
        (d / "tool.py").write_text(code)
    if manifest is not None:
        data = json.loads((d / "plugin.json").read_text())
        data.update(manifest)
        (d / "plugin.json").write_text(json.dumps(data, indent=1))
    return d


def gate(report, gate_id):
    return next(g for g in report["gates"] if g["id"] == gate_id)


def statuses(report):
    return {g["id"]: g["status"] for g in report["gates"]}


@pytest.fixture(scope="module")
def clean_report(tmp_path_factory):
    return validate(str(_copy("py-cells", tmp_path_factory.mktemp("clean"))))


# ===========================================================================
# The clean tool, and the report other modules code against
# ===========================================================================
def test_a_clean_tool_passes_every_gate(clean_report):
    r = clean_report
    assert r["ok"], format_report(r)
    assert statuses(r) == {**{g: "pass" for g in GATES}, "environment": "skip"}
    env = gate(r, "environment")
    assert env["required"] is False and "no pixi.toml" in env["message"]
    # Allowed here; the author learns the registry's rule before CI does.
    assert "registry accepts a tool only with its pixi.toml and pixi.lock" in env["message"]
    assert r["outputs_seen"] == ["summary", "cells"]
    assert r["risks"] == [] and "numpy" in r["imports"]
    assert (r["plugin_id"], r["version"], r["name"]) == ("py-cells", "1.0.0", "Python test tool")
    assert (r["language"], r["gpu"]) == ("python", None)


def test_the_report_has_the_documented_shape(clean_report):
    r = clean_report
    assert set(r) == {"ok", "plugin_id", "version", "name", "language", "gpu", "gates",
                      "risks", "imports", "outputs_seen", "seconds"}
    assert [g["id"] for g in r["gates"]] == ["bundle", "manifest", "environment", "entry",
                                             "imports", "fixture", "result", "edge",
                                             "no_mutation"]
    for g in r["gates"]:
        assert set(g) == {"id", "required", "status", "message", "detail", "seconds"}
        assert g["status"] in ("pass", "fail", "warn", "skip")
    assert [g["required"] for g in r["gates"]] == [True, True, False, True, False, True,
                                                   True, True, True]
    json.dumps(r, allow_nan=False)
    text = format_report(r)
    assert text.startswith("py-cells 1.0.0 — Python test tool (Python tool)")
    assert "PASSED" in text and "7/7 required gates pass" in text
    for gid in GATES:
        assert gid in text


# ===========================================================================
# bundle
# ===========================================================================
def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tool_files(name="py-cells"):
    # bundle.files, not listdir: a run of the tool in place leaves a __pycache__.
    return {rel: open(full, "rb").read() for rel, full in bundle.files(os.path.join(TOOLS, name))}


def test_a_well_formed_bundle_passes_end_to_end(tmp_path):
    path = tmp_path / "py-cells-1.0.0.pvtool"
    path.write_bytes(_zip(_tool_files()))
    r = validate(str(path))
    assert r["ok"], format_report(r)
    assert gate(r, "bundle")["detail"]["source"] == "the bundle"


def test_a_zip_slip_or_a_nested_bundle_is_refused(tmp_path):
    path = tmp_path / "evil.pvtool"
    path.write_bytes(_zip(dict(_tool_files(), **{"../../evil.py": b"x = 1\n"})))
    r = validate(str(path))
    assert not r["ok"] and gate(r, "bundle")["status"] == "fail"
    assert "escapes" in gate(r, "bundle")["message"]
    assert all(g["status"] == "skip" for g in r["gates"][1:])

    path.write_bytes(_zip({f"py-cells/{k}": v for k, v in _tool_files().items()}))
    r = validate(str(path))
    assert "zip the CONTENTS" in gate(r, "bundle")["message"]


def test_a_directory_is_judged_as_the_bundle_it_would_become(tmp_path):
    tool = _tool(tmp_path)
    # An environment built in the folder is never part of the tool.
    (tool / ".pixi" / "envs" / "default" / "lib").mkdir(parents=True)
    (tool / ".pixi" / "envs" / "default" / "lib" / "libbig.so").write_bytes(b"\0" * 1024)
    r = validate(str(tool))
    assert gate(r, "bundle")["status"] == "pass", gate(r, "bundle")
    assert gate(r, "bundle")["detail"]["source"] == "a zip of the directory"
    (tool / "payload.exe").write_bytes(b"MZ")
    r = validate(str(tool))
    assert gate(r, "bundle")["status"] == "fail" and "payload.exe" in gate(r, "bundle")["message"]


# ===========================================================================
# manifest
# ===========================================================================
def test_a_missing_manifest_fails_once_and_skips_every_later_gate(tmp_path):
    tool = _tool(tmp_path)
    (tool / "plugin.json").unlink()
    r = validate(str(tool))
    assert gate(r, "manifest")["status"] == "fail"
    assert all(gate(r, g)["status"] == "skip" for g in GATES[2:])
    assert format_report(r).startswith("(the manifest did not load")


def test_defaults_that_contradict_the_manifest_are_manifest_failures(tmp_path):
    tool = _tool(tmp_path, manifest={"params": [
        {"name": "k", "type": "int", "default": 99, "min": 3, "max": 50},
        {"name": "mode", "type": "select", "default": "fast", "options": ["slow", "exact"]}]})
    r = validate(str(tool))
    message = gate(r, "manifest")["message"]
    assert gate(r, "manifest")["status"] == "fail"
    assert "outside its own declared range" in message and "fast" in message


def test_schema_3_refuses_the_removed_keys(tmp_path):
    r = validate(str(_tool(tmp_path, manifest={"timeout_s": 60, "memory_mb": 2048})))
    assert gate(r, "manifest")["status"] == "fail"
    assert "timeout_s, memory_mb" in gate(r, "manifest")["message"]


def test_a_defaults_table_that_drifts_is_noted_not_failed(tmp_path):
    code = open(os.path.join(TOOLS, "py-cells", "tool.py")).read()
    code += "\nDEFAULTS = dict(k=20, mode='a')\n"
    tool = _tool(tmp_path, code=code, manifest={"params": [
        {"name": "k", "type": "int", "default": 15, "min": 3, "max": 50},
        {"name": "mode", "type": "select", "default": "a", "options": ["a", "b"]}]})
    r = validate(str(tool))
    assert r["ok"], format_report(r)
    assert "DEFAULTS['k'] is 20 but the manifest declares 15" in gate(r, "manifest")["message"]


# ===========================================================================
# environment (pixi)
# ===========================================================================
def _current_platform():
    import platform
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        return "linux-aarch64" if machine in ("aarch64", "arm64") else "linux-64"
    if sys.platform == "darwin":
        return "osx-arm64" if machine == "arm64" else "osx-64"
    return "win-64"


def test_without_a_pixi_toml_the_tool_runs_in_the_app_env_unless_own_is_asked(tmp_path):
    tool = _tool(tmp_path)
    r = validate(str(tool), env="own")
    assert gate(r, "environment")["status"] == "fail" and not gate(r, "environment")["required"]
    assert "no pixi.toml" in gate(r, "environment")["message"]
    assert gate(r, "entry")["status"] == "skip" and not r["ok"]


def test_the_environment_gate_lists_every_problem(tmp_path):
    _pixi_or_skip()
    other = "osx-arm64" if _current_platform() != "osx-arm64" else "linux-64"
    tool = _tool(tmp_path)
    (tool / "pixi.toml").write_text(f'[workspace]\nname = "t"\nchannels = ["conda-forge"]\n'
                                    f'platforms = ["{other}"]\n\n[dependencies]\n'
                                    'python = ">=3.10"\n')
    r = validate(str(tool), env="app")
    env = gate(r, "environment")
    assert env["status"] == "fail" and env["required"] is True and not r["ok"]
    assert "pixi.lock is missing" in env["message"]
    assert "does not declare numpy, scipy" in env["message"]
    assert f"({other}) do not include this machine's ({_current_platform()})" in env["message"]
    # --env app runs the tool where it is: the run gates still report.
    assert gate(r, "fixture")["status"] == "pass"


def test_an_unreadable_pixi_toml_is_named(tmp_path):
    _pixi_or_skip()
    tool = _tool(tmp_path)
    (tool / "pixi.toml").write_text("[workspace\nname = 'x'\n")
    r = validate(str(tool))
    assert gate(r, "environment")["message"] == "pixi cannot read pixi.toml."
    assert gate(r, "entry")["status"] == "skip"       # env own: nothing was built


def test_an_r_tool_must_declare_r(tmp_path):
    _pixi_or_skip()
    tool = _copy("r-labels", tmp_path)
    (tool / "pixi.toml").write_text(f'[workspace]\nname = "t"\nchannels = ["conda-forge"]\n'
                                    f'platforms = ["{_current_platform()}"]\n\n[dependencies]\n'
                                    'python = ">=3.10"\nnumpy = "*"\nscipy = "*"\n')
    env = gate(validate(str(tool), env="app"), "environment")
    assert "does not declare r-base, r-jsonlite" in env["message"]


# ===========================================================================
# entry
# ===========================================================================
def test_a_file_with_no_run_or_the_wrong_arity_fails_entry(tmp_path):
    r = validate(str(_tool(tmp_path, code="x = 1\n", name="norun")))
    assert gate(r, "entry")["status"] == "fail" and "defines no run" in gate(r, "entry")["message"]
    assert gate(r, "fixture")["status"] == "skip"

    r = validate(str(_tool(tmp_path, code="def run(ds):\n    return {}\n", name="arity")))
    assert "run(ds)" in gate(r, "entry")["message"]

    r = validate(str(_tool(tmp_path, code="run = 3\n", name="value")))
    assert "it is a int" in gate(r, "entry")["message"]

    r = validate(str(_tool(tmp_path, code="def run(ds, params, k=3):\n"
                                           "    return {'summary': 'x'}\n", name="extra")))
    assert gate(r, "entry")["status"] == "pass"


def test_an_unresolvable_import_fails_entry_with_the_exception(tmp_path):
    r = validate(str(_tool(tmp_path, code="import not_a_module_anywhere\n"
                                          "def run(ds, params):\n    return {}\n")))
    entry = gate(r, "entry")
    assert entry["status"] == "fail" and "not_a_module_anywhere" in entry["message"]
    assert entry["detail"]["notes"], "the traceback is what the author needs"


def test_an_import_that_exits_is_reported_not_lost(tmp_path):
    r = validate(str(_tool(tmp_path, code="import sys\nsys.exit(5)\n")))
    assert "SystemExit" in gate(r, "entry")["message"]
    r = validate(str(_tool(tmp_path, code="import os\nos._exit(7)\n", name="hard")))
    assert "died without answering (exit status 7)" in gate(r, "entry")["message"]


def test_helpers_named_like_standard_modules_do_not_break_the_probe(tmp_path):
    tool = _tool(tmp_path)
    for name in ("select.py", "inspect.py", "json.py"):
        (tool / name).write_text("raise ImportError('the tool replaced this module')\n")
    r = validate(str(tool))
    assert r["ok"], format_report(r)


def test_the_entry_file_is_never_imported_in_the_validator(tmp_path):
    marker = tmp_path / "imported"
    validate(str(_tool(tmp_path, code=f"open({str(marker)!r}, 'a').write(str(__import__('os').getpid()) + '\\n')\n"
                                      "def run(ds, params):\n    return {'summary': 'x'}\n")))
    pids = marker.read_text().split()
    assert pids and str(os.getpid()) not in pids


@needs_r
def test_r_entries_are_sourced_in_r(tmp_path):
    tool = _copy("r-labels", tmp_path)
    (tool / "tool.R").write_text("run <- function(ds) 1\n")
    entry = gate(validate(str(tool)), "entry")
    assert entry["status"] == "fail" and "function(ds), which takes 1 argument" in entry["message"]
    (tool / "tool.R").write_text("x <- 1\n")
    assert "defines no run" in gate(validate(str(tool)), "entry")["message"]
    (tool / "tool.R").write_text("library(notapackage)\n")
    entry = gate(validate(str(tool)), "entry")
    assert "sourcing tool.R raised" in entry["message"] and "notapackage" in entry["message"]


# ===========================================================================
# imports (advisory)
# ===========================================================================
def test_risks_are_disclosed_and_the_tool_still_passes(tmp_path):
    code = ("import socket\n"
            "from helper import touch\n"
            "def run(ds, params):\n"
            "    eval('1 + 1')\n"
            "    return {'summary': 'x', 'cells': {'values': [1.0] * ds.n_cells}}\n")
    tool = _tool(tmp_path, code=code)
    (tool / "helper.py").write_text("import os\n"
                                    "def touch():\n"
                                    "    os.environ['X'] = '1'\n"
                                    "    open('out.txt', 'w')\n")
    r = validate(str(tool))
    imports = gate(r, "imports")
    assert imports["status"] == "warn" and not imports["required"]
    assert r["ok"], format_report(r)
    assert r["risks"] == ["network", "filesystem_write", "dynamic_exec", "env"]
    assert "import socket at tool.py:1" in imports["message"]
    assert "helper.py:4" in imports["message"]


def test_the_r_scan_finds_risks_and_ignores_comments():
    modules, findings = _scan_r("tool.R", (
        'library(mclust)\n'
        'x <- data.table::fread("a")\n'
        '# system("rm -rf /") is only a comment\n'
        's <- "# not a comment"; download.file(u, "f")\n'
        'system2("ls")\n'
        'con <- file("out.txt", "w"); writeLines("a", con)\n'
        'my_url("x"); key <- Sys.getenv("HOME")\n'
        'eval(parse(text = "1"))\n'
        'requireNamespace("httr")\n'))
    assert modules == {"mclust", "data.table", "httr"}
    found = {(risk, where) for risk, where, _what in findings}
    assert ("network", "tool.R:4") in found and ("subprocess", "tool.R:5") in found
    assert ("filesystem_write", "tool.R:6") in found and ("env", "tool.R:7") in found
    assert ("dynamic_exec", "tool.R:8") in found and ("network", "tool.R") in found
    assert not any(where == "tool.R:3" for _r, where in found)
    assert sum(1 for risk, where in found if where == "tool.R:7") == 1     # my_url is not url


# ===========================================================================
# fixture, result
# ===========================================================================
def test_a_crash_on_the_defaults_fails_fixture_with_the_traceback(tmp_path):
    r = validate(str(_tool(tmp_path, code="def run(ds, params):\n    return 1 / 0\n")))
    fixture = gate(r, "fixture")
    assert fixture["status"] == "fail" and "ZeroDivisionError" in fixture["message"]
    assert any("1 / 0" in note for note in fixture["detail"]["notes"])
    assert gate(r, "result")["status"] == "skip" and gate(r, "edge")["status"] == "skip"


def test_a_refusal_of_its_own_defaults_fails_fixture(tmp_path):
    r = validate(str(_tool(tmp_path, code="from pointillviz_tool import PluginError\n"
                                          "def run(ds, params):\n"
                                          "    raise PluginError('nope', 'Never.')\n")))
    message = gate(r, "fixture")["message"]
    assert "refused its own defaults" in message and "600 cells" in message


def test_a_tool_that_exits_without_answering_fails_fixture(tmp_path):
    r = validate(str(_tool(tmp_path, code="import os\n"
                                          "def run(ds, params):\n    os._exit(0)\n")))
    assert "without returning a result" in gate(r, "fixture")["message"]


def test_parameters_with_no_default_are_filled_from_the_fixture(tmp_path):
    code = ("def run(ds, params):\n"
            "    ds.gene(params['g'])\n"
            "    ds.categorical(params['field'])\n"
            "    return {'summary': f\"{params['g']} {params['a']} {params['b']}\",\n"
            "            'cells': {'values': [1.0] * ds.n_cells}}\n")
    tool = _tool(tmp_path, code=code, manifest={"params": [
        {"name": "g", "type": "gene"},
        {"name": "field", "type": "categorical_field"},
        {"name": "a", "type": "category_value", "field_param": "field"},
        {"name": "b", "type": "category_value", "field_param": "field"}]})
    r = validate(str(tool))
    assert r["ok"], format_report(r)
    params = gate(r, "fixture")["detail"]["params"]
    assert params["g"] == "Ubiquitous"                  # not "Silent", which is expressed nowhere
    assert params["a"] != params["b"]


def test_a_rejected_return_value_is_the_result_gates_failure(tmp_path):
    r = validate(str(_tool(tmp_path, code="def run(ds, params):\n"
                                          "    return {'summary': 's', 'cells': {'values': [1.0]}}\n")))
    assert gate(r, "fixture")["status"] == "pass"
    result = gate(r, "result")
    assert result["status"] == "fail" and "bad_length" in result["message"]


def test_schema_3_returns_what_it_declares_and_schema_1_is_checked_here(tmp_path):
    code = "def run(ds, params):\n    return {'summary': 's'}\n"
    r = validate(str(_tool(tmp_path, code=code, name="three")))
    assert "declares cells in outputs" in gate(r, "result")["message"]

    one = _tool(tmp_path, code="def run(ds, params):\n"
                               "    return {'summary': 's', 'table': {'columns': ['a'], 'rows': []}}\n",
                manifest={"schema": 1, "outputs": ["summary"]}, name="one")
    result = gate(validate(str(one)), "result")
    assert result["status"] == "fail" and "table, which the manifest does not declare" in result["message"]


def test_a_blank_map_at_the_defaults_fails_result(tmp_path):
    r = validate(str(_tool(tmp_path, code="def run(ds, params):\n"
                                          "    return {'summary': 's', 'cells': {'values': [None] * ds.n_cells}}\n")))
    assert "cells (0 of 600 values are finite)" in gate(r, "result")["message"]


def test_blank_labels_fail_result_and_partial_labels_pass(tmp_path):
    code = ("def run(ds, params):\n"
            "    values = [None] * ds.n_cells\n"
            "    if params['k'] > 3:\n"
            "        values[0] = 0\n"
            "    return {'summary': 's', 'labels': {'values': values, 'categories': ['one']}}\n")
    tool = _tool(tmp_path, code=code, manifest={"outputs": ["summary", "labels"]})
    r = validate(str(tool))
    assert r["ok"], format_report(r)
    edge = gate(r, "edge")["message"]
    assert "painted nothing there — labels (0 of 600 cells have a label)" in edge


def test_a_declared_sweep_is_exercised(tmp_path):
    code = ("from pointillviz_tool import PluginError\n"
            "def run(ds, params):\n"
            "    if params['k'] == 50:\n"
            "        raise PluginError('too_big', 'k is too big here.')\n"
            "    return {'summary': 's', 'table': {'columns': ['k'], 'rows': [[params['k']]]}}\n")
    tool = _tool(tmp_path, code=code, manifest={
        "outputs": ["summary", "table"],
        "params": [{"name": "k", "type": "int", "default": 15, "min": 3, "max": 50, "sweep": True}]})
    r = validate(str(tool))
    assert r["ok"], format_report(r)
    fixture = gate(r, "fixture")
    assert "3-step sweep over k (3 to 50)" in fixture["message"]
    assert fixture["detail"]["sweep"]["refused"] == [50]


# ===========================================================================
# edge
# ===========================================================================
CORNERS = ("from pointillviz_tool import PluginError\n"
           "def run(ds, params):\n"
           "    k = params['k']\n"
           "    if k == {at} and {crash}:\n"
           "        return 1 / 0\n"
           "    if k == {at}:\n"
           "        raise PluginError('k_range', 'k is out of reach here.')\n"
           "    return {{'summary': 's', 'cells': {{'values': [1.0] * ds.n_cells}}}}\n")


@pytest.mark.parametrize("at,label", [(3, "minimum"), (50, "maximum")])
def test_a_crash_at_a_corner_fails_edge_and_a_refusal_passes(tmp_path, at, label):
    r = validate(str(_tool(tmp_path, code=CORNERS.format(at=at, crash=True), name="crash")))
    edge = gate(r, "edge")
    assert edge["status"] == "fail" and f"at the {label} corner (k={at}) the tool crashed" in edge["message"]
    r = validate(str(_tool(tmp_path, code=CORNERS.format(at=at, crash=False), name="refuse")))
    edge = gate(r, "edge")
    assert edge["status"] == "pass" and "refused in words" in edge["message"]


# ===========================================================================
# no_mutation
# ===========================================================================
def test_writing_into_the_dataset_fails_no_mutation(tmp_path):
    r = validate(str(_tool(tmp_path, code=(
        "import os\n"
        "def run(ds, params):\n"
        "    open(os.path.join(ds.dir, 'scribble.txt'), 'w').write('x')\n"
        "    return {'summary': 's', 'cells': {'values': [1.0] * ds.n_cells}}\n"))))
    gate_ = gate(r, "no_mutation")
    assert gate_["status"] == "fail" and "wrote scribble.txt" in gate_["message"]


def test_the_knn_cache_is_not_a_mutation(tmp_path):
    r = validate(str(_tool(tmp_path, code=(
        "import numpy as np\n"
        "def run(ds, params):\n"
        "    idx, _ = ds.knn(params['k'])\n"
        "    return {'summary': 's', 'cells': {'values': (idx >= 0).sum(axis=1).astype(float)}}\n"))))
    assert r["ok"], format_report(r)


# ===========================================================================
# Where the tool runs, and cancel
# ===========================================================================
def test_a_legacy_tool_gets_the_extra_paths(tmp_path):
    shim = tmp_path / "app" / "backend"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("")
    (shim / "plugin_api.py").write_text("from pointillviz_tool import PluginError\n")
    tool = _tool(tmp_path, code=(
        "from backend.plugin_api import PluginError\n"
        "def run(ds, params):\n"
        "    return {'summary': 's', 'cells': {'values': [1.0] * ds.n_cells}}\n"),
        manifest={"schema": 1, "timeout_s": 60, "memory_mb": 1024})
    assert gate(validate(str(tool)), "entry")["status"] == "fail"
    r = validate(str(tool), extra_paths=[str(tmp_path / "app")])
    assert r["ok"], format_report(r)


def test_cancelling_a_validation_ends_it(tmp_path):
    tool = _tool(tmp_path, code=("import time\n"
                                 "def run(ds, params):\n"
                                 "    time.sleep(600)\n"))
    out = {}
    worker = threading.Thread(target=lambda: out.update(
        report=validate(str(tool), run_id="validate-me")))
    worker.start()
    # cancel() says True once a run under that id is live: the fixture run.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not launch.cancel("validate-me"):
        time.sleep(0.1)
    worker.join(60)
    r = out["report"]
    assert not r["ok"]
    assert "cancelled" in gate(r, "fixture")["message"]
    assert all(gate(r, g)["message"] == "skipped: the validation was cancelled."
               for g in ("result", "edge", "no_mutation"))


def test_cancelling_during_a_slow_import_ends_the_validation(tmp_path):
    # With no clock, an import that blocks would hold the validation, and an
    # install waiting on it, for as long as it blocks.
    tool = _tool(tmp_path, code="import time\ntime.sleep(600)\n"
                                "def run(ds, params):\n    return {}\n")
    out = {}
    worker = threading.Thread(target=lambda: out.update(
        report=validate(str(tool), run_id="slow-import")))
    started = time.monotonic()
    worker.start()
    while time.monotonic() - started < 60 and not launch.cancel("slow-import"):
        time.sleep(0.1)
    worker.join(30)
    assert not worker.is_alive() and time.monotonic() - started < 60
    r = out["report"]
    assert gate(r, "entry")["status"] == "fail" and "cancelled" in gate(r, "entry")["message"]
    assert all(gate(r, g)["message"] == "skipped: the validation was cancelled."
               for g in ("imports", "fixture", "result", "edge", "no_mutation"))


GPU_ONLY = ("from pointillviz_tool import PluginError\n"
            "def run(ds, params):\n"
            "    if ds.device != 'cuda':\n"
            "        raise PluginError('no_gpu', 'This tool runs on a GPU only.')\n"
            "    return {'summary': 's', 'cells': {'values': [1.0] * ds.n_cells}}\n")


FALLBACK = ("def run(ds, params):\n"
            "    ds.log(f'computing on {ds.device}')\n"
            "    return {'summary': 's', 'cells': {'values': [1.0] * ds.n_cells}}\n")
RUN_GATES = ("fixture", "result", "edge", "no_mutation")


def test_a_gpu_tool_is_gated_on_the_cpu_like_any_other(tmp_path):
    # Every tool's gates run on the CPU, as in CI, "required" included: a GPU
    # method checks for CUDA and falls back, and that fallback is tested here.
    only = _tool(tmp_path, code=GPU_ONLY, manifest={"gpu": "required"}, name="gpu-only")
    r = validate(str(only))
    fixture = gate(r, "fixture")
    assert not r["ok"] and fixture["status"] == "fail" and fixture["required"]
    assert "This tool runs on a GPU only. (code: no_gpu)" in fixture["message"]
    assert 'declares "gpu": "required", and the gates run it on the CPU' in fixture["message"]

    fallback = _tool(tmp_path, code=FALLBACK, manifest={"gpu": "required"}, name="fallback")
    r = validate(str(fallback))
    assert r["ok"], format_report(r)
    assert all(gate(r, g)["status"] == "pass" and gate(r, g)["required"] for g in RUN_GATES)
    assert "computing on cpu" in gate(r, "fixture")["detail"]["logs"]

    # An author checking the GPU path still can: the tool sees ds.device == "cuda".
    r = validate(str(only), device="cuda")
    assert r["ok"], format_report(r)
    assert all(gate(r, g)["status"] == "pass" for g in RUN_GATES)


@needs_r
def test_a_clean_r_tool_passes_every_gate(tmp_path):
    r = validate(str(_copy("r-labels", tmp_path)))
    assert r["ok"], format_report(r)
    assert (r["language"], r["outputs_seen"]) == ("r", ["summary", "table", "labels"])
    assert "defines run <- function(ds, params)" in gate(r, "entry")["message"]
    assert "(R tool)" in format_report(r).splitlines()[0]
