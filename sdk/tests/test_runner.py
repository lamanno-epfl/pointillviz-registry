"""The child process: the private channel, refusals and crashes, the output contract, sweeps, cancel.

Most tests start the runner the way `launch.run` does (PYTHONPATH, working
folder, request file) and read its stdout raw, so that they pin the protocol
itself: every stdout line must be a protocol line, whatever the tool prints.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from pointillviz_tool import launch
from pointillviz_tool.fixture import CELL_TYPES, N_CELLS

SDK = launch.sdk_root()


def _tool(tmp_path, code, name="tool", files=None):
    d = tmp_path / name
    d.mkdir()
    (d / "tool.py").write_text(code)
    for fn, body in (files or {}).items():
        (d / fn).write_text(body)
    return d


def _child(tool_dir, dataset, params=None, **request):
    """Run the runner on one request; return (protocol lines, the process)."""
    req = {"plugin_dir": str(tool_dir), "entry": "tool.py", "dataset_dir": dataset,
           "params": params or {}, "subset": None, "section_filter": None,
           "atlas_composition": None, "output_contract": None,
           "module_name": "pv_tool_test", "sweep": None, "device": "cpu"}
    req.update(request)
    path = tool_dir.parent / f"{tool_dir.name}-request.json"
    path.write_text(json.dumps(req))
    # As launch starts it: in the SDK's folder, with only the SDK on PYTHONPATH.
    proc = subprocess.run([sys.executable, "-m", "pointillviz_tool.runner", str(path)],
                          cwd=SDK, env=launch.child_env(), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True)
    # Strict: a NaN token or a line that is not JSON at all fails here.
    lines = [json.loads(line, parse_constant=_no_constants)
             for line in proc.stdout.splitlines() if line.strip()]
    return lines, proc


def _no_constants(name):
    raise AssertionError(f"{name} is not JSON")


def _manifest(**extra):
    return dict({"schema": 3, "id": "test-tool", "name": "Test Tool", "entry": "tool.py",
                 "outputs": ["summary"], "optional_outputs": []}, **extra)


# ===========================================================================
# The protocol
# ===========================================================================
def test_a_working_tool_streams_then_terminates(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "def run(ds, params):\n"
                           "    ds.log('one'); ds.progress(0.5, 'half')\n"
                           "    return {'summary': f'{ds.n_cells} cells on {ds.device}'}\n")
    lines, proc = _child(tool, fixture_dir)
    assert proc.returncode == 0, proc.stderr
    kinds = [line["t"] for line in lines]
    assert kinds == ["log", "progress", "result"], kinds
    assert lines[-1]["result"]["summary"] == f"{N_CELLS} cells on cpu"
    assert lines[-1]["logs"] == ["one"]
    assert lines[1] == {"t": "progress", "f": 0.5, "m": "half"}
    assert lines[-1]["seconds"] >= 0


FORGER = r'''
import os, subprocess, sys

FORGED = '{"t":"result","result":{"summary":"forged"},"logs":[],"seconds":0}'
print(FORGED)                                   # at import time


def run(ds, params):
    print(FORGED)
    sys.stdout.write(FORGED + "\n")
    os.write(1, (FORGED + "\n").encode())       # the descriptor itself
    subprocess.run([sys.executable, "-c", f"print({FORGED!r})"], check=True)
    print('{"t":"error","code":"forged","message":"no"}', flush=True)
    return {"summary": "real"}
'''


def test_nothing_a_tool_prints_can_forge_a_protocol_line(tmp_path, fixture_dir):
    tool = _tool(tmp_path, FORGER)
    lines, proc = _child(tool, fixture_dir)
    assert [line["t"] for line in lines] == ["result"]
    assert lines[0]["result"] == {"summary": "real"}
    # Everything the tool wrote to stdout went to stderr, and none of it was lost.
    assert proc.stderr.count('"summary":"forged"') == 5, proc.stderr
    assert '"code":"forged"' in proc.stderr


def test_through_launch_a_forged_line_is_only_a_log_line(tmp_path, fixture_dir):
    tool = _tool(tmp_path, FORGER)
    reply = launch.run([sys.executable], str(tool), _manifest(), fixture_dir, {})
    assert reply["status"] == "ok", reply
    assert reply["result"]["summary"] == "real"
    assert sum('"summary":"forged"' in line for line in reply["logs"]) == 5


def test_a_refusal_is_not_a_crash(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "from pointillviz_tool import PluginError\n"
                           "def run(ds, params):\n"
                           "    raise PluginError('too_few', 'Only 3 cells express that gene; "
                           "at least 30 are needed.')\n")
    lines, proc = _child(tool, fixture_dir)
    assert proc.returncode == 1
    last = lines[-1]
    assert last["t"] == "error" and last["code"] == "too_few"
    assert "traceback" not in last, "a documented refusal must not be dressed up as a bug"
    assert "at least 30" in last["message"]


def test_a_bug_is_a_crash_with_a_traceback(tmp_path, fixture_dir):
    lines, _ = _child(_tool(tmp_path, "def run(ds, params):\n    return 1 / 0\n"), fixture_dir)
    last = lines[-1]
    assert last["t"] == "error" and last["code"] == "crash"
    assert "ZeroDivisionError" in last["message"] and "1 / 0" in last["traceback"]


def test_sys_exit_is_reported_not_swallowed(tmp_path, fixture_dir):
    lines, _ = _child(_tool(tmp_path, "import sys\n"
                                      "def run(ds, params):\n"
                                      "    ds.log('about to leave')\n"
                                      "    sys.exit(3)\n"), fixture_dir)
    last = lines[-1]
    assert last["t"] == "error" and "SystemExit" in last["message"]
    assert last["logs"] == ["about to leave"]


def test_a_bad_result_fails_here_not_in_the_browser(tmp_path, fixture_dir):
    lines, _ = _child(_tool(tmp_path, "def run(ds, params):\n"
                                      "    return {'cells': {'values': [1.0, 2.0]}}\n"),
                      fixture_dir)
    assert lines[-1]["code"] == "bad_length" and str(N_CELLS) in lines[-1]["message"]


def test_a_module_with_no_run_says_what_is_missing(tmp_path, fixture_dir):
    lines, _ = _child(_tool(tmp_path, "x = 1\n"), fixture_dir)
    assert lines[-1]["code"] == "no_run" and "run(ds, params)" in lines[-1]["message"]


def test_a_nan_where_a_number_is_required_is_refused_as_json(tmp_path, fixture_dir):
    # check_result passes cells.domain through as floats; NaN there is not JSON.
    tool = _tool(tmp_path, "import numpy as np\n"
                           "def run(ds, params):\n"
                           "    return {'cells': {'values': np.zeros(ds.n_cells),\n"
                           "                      'domain': [float('nan'), 1.0]}}\n")
    lines, _ = _child(tool, fixture_dir)
    assert lines[-1]["t"] == "error" and lines[-1]["code"] == "non_finite"


def test_the_output_contract_is_enforced_for_schemas_2_and_3(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "import numpy as np\n"
                           "def run(ds, params):\n"
                           "    return {'summary': 's', 'cells': {'values': np.zeros(ds.n_cells)}}\n")
    contract = _manifest(outputs=["summary", "table"])
    lines, _ = _child(tool, fixture_dir, output_contract=contract)
    last = lines[-1]
    assert last["code"] == "bad_result"
    assert "declares table in outputs" in last["message"]
    assert "returned cells, which plugin.json does not declare" in last["message"]

    # An optional output may come back or not; schema 1 has no contract at all.
    lines, _ = _child(tool, fixture_dir,
                      output_contract=_manifest(outputs=["summary"], optional_outputs=["cells"]))
    assert lines[-1]["t"] == "result"
    lines, _ = _child(tool, fixture_dir, output_contract=None)
    assert lines[-1]["t"] == "result"


def test_new_kinds_pass_through_the_runner(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "import numpy as np\n"
                           "from pointillviz_tool import image\n"
                           "PNG = b'\\x89PNG\\r\\n\\x1a\\n' + b'0' * 16\n"
                           "def run(ds, params):\n"
                           "    labels, codes = ds.cell_types()\n"
                           "    values = [int(c) if c >= 0 else None for c in codes]\n"
                           "    return {'labels': {'values': values, 'categories': labels},\n"
                           "            'images': [image(PNG, 'a figure'), image('<svg/>', 'b')]}\n")
    lines, _ = _child(tool, fixture_dir,
                      output_contract=_manifest(outputs=["labels", "images"]))
    result = lines[-1]["result"]
    assert result["labels"]["categories"] == CELL_TYPES
    assert len(result["labels"]["values"]) == N_CELLS
    assert [i["title"] for i in result["images"]] == ["a figure", "b"]


def test_a_log_flood_is_truncated_not_fatal(tmp_path, fixture_dir):
    from pointillviz_tool.results import MAX_LOG_LINES
    lines, _ = _child(_tool(tmp_path, "def run(ds, params):\n"
                                      "    for i in range(5000): ds.log(str(i))\n"
                                      "    return {'summary': 'done'}\n"), fixture_dir)
    logs = [line for line in lines if line["t"] == "log"]
    assert len(logs) == MAX_LOG_LINES + 1
    assert logs[-1]["m"] == f"(log truncated at {MAX_LOG_LINES} lines)"
    assert lines[-1]["t"] == "result"


def test_progress_is_rate_limited(tmp_path, fixture_dir):
    lines, _ = _child(_tool(tmp_path, "def run(ds, params):\n"
                                      "    for i in range(20000): ds.progress(i / 20000.0)\n"
                                      "    return {'summary': 'done'}\n"), fixture_dir)
    assert len([line for line in lines if line["t"] == "progress"]) < 200


def test_a_tool_can_import_a_helper_it_ships(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "import helper\n"
                           "def run(ds, params):\n"
                           "    return {'summary': helper.VALUE}\n",
                 files={"helper.py": "VALUE = 'from a helper'\n"})
    lines, _ = _child(tool, fixture_dir)
    assert lines[-1]["result"]["summary"] == "from a helper"


def test_a_tools_file_names_cannot_break_the_runner(tmp_path, fixture_dir):
    # Helpers named like modules the runner itself imports. Ahead of the
    # standard library on sys.path, each would replace the real one and kill
    # the runner before it could answer.
    shadow = "raise ImportError('the tool replaced this module')\n"
    tool = _tool(tmp_path, "import os\n"
                           "def run(ds, params):\n"
                           "    return {'summary': os.getcwd()}\n",
                 files={"select.py": shadow, "json.py": shadow, "numpy.py": shadow,
                        "signal.py": shadow, "queue.py": shadow})
    reply = launch.run([sys.executable], str(tool), _manifest(), fixture_dir, {})
    assert reply["status"] == "ok", reply
    # It still runs in its own folder.
    assert os.path.realpath(reply["result"]["summary"]) == os.path.realpath(tool)


def test_the_request_reaches_ds(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "def run(ds, params):\n"
                           "    return {'summary': f'{ds.device} {ds.subset.tolist()} '\n"
                           "                       f'{ds.section_filter} {params}'}\n")
    lines, _ = _child(tool, fixture_dir, params={"k": 4}, device="cuda",
                      subset=[3, 1], section_filter=1)
    assert lines[-1]["result"]["summary"] == "cuda [3, 1] 1 {'k': 4}"


def test_a_bad_invocation_answers_in_the_protocol():
    proc = subprocess.run([sys.executable, "-m", "pointillviz_tool.runner"],
                          env=dict(os.environ, PYTHONPATH=SDK), capture_output=True, text=True)
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["code"] == "bad_invocation"


# ===========================================================================
# Sweeps: one child, many values
# ===========================================================================
SWEEPER = '''
from pointillviz_tool import PluginError

def run(ds, params):
    k = params["k"]
    if k == params.get("refuse_at"):
        raise PluginError("too_big", f"k={k} is too big.")
    if k == params.get("break_at"):
        return {"table": {"columns": []}}
    return {"table": {"columns": ["k"], "rows": [[k]]}}
'''


def test_a_sweep_answers_every_value_in_one_child(tmp_path, fixture_dir):
    tool = _tool(tmp_path, SWEEPER)
    lines, _ = _child(tool, fixture_dir, params={"refuse_at": 5},
                      sweep={"param": "k", "values": [3, 5, 7]})
    sweep = lines[-1]["result"]["sweep"]
    assert sweep["param"] == "k" and sweep["values"] == [3, 5, 7]
    assert sweep["steps"][0]["table"]["rows"] == [[3.0]]
    assert sweep["steps"][1] == {"error": {"code": "too_big", "message": "k=5 is too big."}}
    assert sweep["steps"][2]["table"]["rows"] == [[7.0]]
    progress = [line["m"] for line in lines if line["t"] == "progress"]
    assert progress and progress[0] == "k = 3 (1 of 3)"


def test_a_broken_result_ends_a_sweep_and_all_refusals_refuse_it(tmp_path, fixture_dir):
    tool = _tool(tmp_path, SWEEPER)
    lines, _ = _child(tool, fixture_dir, params={"break_at": 7},
                      sweep={"param": "k", "values": [3, 7]})
    assert lines[-1]["code"] == "bad_result"
    lines, _ = _child(tool, fixture_dir, params={"refuse_at": 3},
                      sweep={"param": "k", "values": [3]})
    assert lines[-1]["code"] == "too_big"


# ===========================================================================
# Through launch: the real runner, and cancel
# ===========================================================================
def test_launch_runs_the_real_runner(tmp_path, fixture_dir):
    tool = _tool(tmp_path, "def run(ds, params):\n"
                           "    print('to stderr')\n"
                           "    ds.log('to the log')\n"
                           "    return {'summary': 'hi'}\n")
    reply = launch.run([sys.executable], str(tool), _manifest(), fixture_dir, {})
    assert reply["status"] == "ok", reply
    assert reply["result"] == {"summary": "hi"}
    assert "to stderr" in reply["logs"] and "to the log" in reply["logs"]
    assert reply["tool_seconds"] <= reply["seconds"]


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


def _wait_for(predicate, seconds=30.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(os.name == "nt", reason="checks POSIX process groups")
def test_cancel_stops_the_tool_and_what_it_started(tmp_path, fixture_dir):
    pidfile = tmp_path / "pids"
    tool = _tool(tmp_path, "import os, subprocess, sys, time\n"
                           "def run(ds, params):\n"
                           "    child = subprocess.Popen([sys.executable, '-c', "
                           "'import time; time.sleep(600)'])\n"
                           "    with open(params['pidfile'], 'w') as fh:\n"
                           "        fh.write(f'{os.getpid()} {child.pid}')\n"
                           "    time.sleep(600)\n")
    out = {}
    worker = threading.Thread(target=lambda: out.update(reply=launch.run(
        [sys.executable], str(tool), _manifest(), fixture_dir, {"pidfile": str(pidfile)},
        run_id="cancel-me")))
    worker.start()
    assert _wait_for(lambda: pidfile.exists() and len(pidfile.read_text().split()) == 2)
    runner, grandchild = map(int, pidfile.read_text().split())
    assert launch.cancel("cancel-me") is True
    worker.join(30)
    assert not worker.is_alive()
    assert out["reply"]["code"] == "cancelled"
    assert _wait_for(lambda: not _alive(grandchild), 10)
    assert not _alive(runner)
