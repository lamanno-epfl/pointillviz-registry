"""The parent side of a run: request, environment, protocol, logs, cancel.

These tests drive `launch.run` with a stand-in runner, so that they pin the
parent alone. The launcher is an argv prefix, so `[python, fake_runner.py]`
makes the child command `python fake_runner.py -m pointillviz_tool.runner
<request.json>`, and the stand-in reads the request from its last argument.
What it does is chosen by `params["mode"]`.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

from pointillviz_tool import launch

FAKE_RUNNER = r'''
import json, os, subprocess, sys, time

req_path = sys.argv[-1]
with open(req_path) as fh:
    req = json.load(fh)
params = req["params"] or {}
mode = params.get("mode", "ok")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def result(**extra):
    emit({"t": "result", "result": dict({"summary": "done"}, **extra),
          "logs": [], "seconds": 0.01})


if mode == "ok":
    emit({"t": "log", "m": "from ds.log"})
    emit({"t": "progress", "f": 0.5, "m": "half way"})
    print("from print", file=sys.stderr, flush=True)
    result(seen={
        "argv": sys.argv[1:], "cwd": os.getcwd(), "request": req,
        "request_path": req_path, "stdin": sys.stdin.read(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "POINTILLVIZ_DEVICE": os.environ.get("POINTILLVIZ_DEVICE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
elif mode == "refuse":
    emit({"t": "error", "code": "bad_input", "message": "No cells to work on.",
          "logs": [], "seconds": 0.02, "traceback": "Traceback: here"})
elif mode == "crash":
    print("Fatal Python error: Segmentation fault", file=sys.stderr, flush=True)
    os._exit(3)
elif mode == "flood":
    for i in range(500):
        print(f"line {i}", file=sys.stderr)
    sys.stderr.flush()
    result()
elif mode == "bar":
    sys.stderr.write("\r 10%|#   |\r 50%|##  |\r100%|####|\n")
    sys.stderr.flush()
    result()
elif mode == "hang":
    # A grandchild in the same group, then wait to be cancelled.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(params["pidfile"], "w") as fh:
        fh.write(f"{os.getpid()} {child.pid}")
    time.sleep(600)
elif mode == "linger":
    result()
    time.sleep(600)
elif mode == "orphan":
    # Leaves without answering, while a child it started keeps stdout open.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(params["pidfile"], "w") as fh:
        fh.write(str(child.pid))
    os._exit(0)
'''

MANIFEST = {"schema": 3, "id": "fake-tool", "name": "Fake Tool", "entry": "tool.py"}


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / "fake_runner.py"
    path.write_text(FAKE_RUNNER)
    return [sys.executable, "-u", str(path)]


@pytest.fixture
def tool(tmp_path):
    path = tmp_path / "tool"
    path.mkdir()
    (path / "tool.py").write_text("def run(ds, params):\n    return {'summary': 'hi'}\n")
    return path


def _run(fake, tool, fixture_dir, mode, **kw):
    params = dict({"mode": mode}, **kw.pop("params", {}))
    return launch.run(fake, str(tool), MANIFEST, fixture_dir, params, **kw)


def _alive(pid: int) -> bool:
    # A zombie is dead for this purpose: it runs nothing and holds nothing.
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


def _wait_for(predicate, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_sdk_root_holds_the_package():
    assert os.path.isfile(os.path.join(launch.sdk_root(), "pointillviz_tool", "launch.py"))


def test_a_pip_installed_sdk_reaches_the_tool_alone(tmp_path, tool, fixture_dir):
    # pip puts the SDK in site-packages beside the numpy and scipy it depends
    # on, built for the installer's Python. Only the package may reach the
    # tool's environment; a numpy that cannot be imported stands for theirs.
    site = tmp_path / "lib" / "site-packages"
    shutil.copytree(os.path.dirname(launch.__file__), site / "pointillviz_tool",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (site / "numpy").mkdir()
    (site / "numpy" / "__init__.py").write_text(
        "raise ImportError('a numpy built for another Python')\n")
    manifest = dict(MANIFEST, outputs=["summary"], optional_outputs=[], params=[])
    # The parent finds this environment's numpy first, and the SDK only in `site`.
    code = (f"import json, os, sys; sys.path.append({str(site)!r})\n"
            "from pointillviz_tool import launch\n"
            f"reply = launch.run([sys.executable], {str(tool)!r}, {manifest!r}, "
            f"{fixture_dir!r}, {{}})\n"
            "root = launch.sdk_root()\n"
            "print(json.dumps({'root': root, 'listing': os.listdir(root), 'reply': reply}))\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True)
    seen = json.loads(out.stdout)
    assert seen["reply"]["status"] == "ok", seen["reply"]
    assert seen["listing"] == ["pointillviz_tool"]
    assert not os.path.exists(seen["root"])            # removed when the parent exits


def test_request_environment_and_logs(fake, tool, fixture_dir, tmp_path):
    extra = tmp_path / "app-root"
    reply = _run(fake, tool, fixture_dir, "ok", subset=[3, 1], section_filter=1,
                 sweep={"param": "k", "values": [5, 10]}, extra_paths=[str(extra)])
    assert reply["status"] == "ok", reply
    seen = reply["result"]["seen"]

    assert seen["argv"] == ["-m", "pointillviz_tool.runner", seen["request_path"]]
    assert not os.path.exists(seen["request_path"])        # removed after the run
    # Not the tool's folder, which -m and PYTHONPATH would both put ahead of
    # the standard library; the runner moves there once its imports are done.
    assert os.path.realpath(seen["cwd"]) == os.path.realpath(launch.sdk_root())
    assert seen["PYTHONPATH"].split(os.pathsep) == [launch.sdk_root(), str(extra)]
    assert seen["stdin"] == ""                               # no stdin to wait on

    req = seen["request"]
    assert req["plugin_dir"] == str(tool)
    assert req["entry"] == "tool.py"
    assert req["dataset_dir"] == os.path.abspath(fixture_dir)
    assert req["subset"] == [3, 1] and req["section_filter"] == 1
    assert req["sweep"] == {"param": "k", "values": [5, 10]}
    assert req["device"] == "cpu"
    assert req["module_name"] == "pv_tool_fake_tool"
    assert req["output_contract"]["id"] == "fake-tool"     # schema 3 is held to its outputs
    assert "memory_mb" not in req and "timeout_s" not in req

    # ds.log lines and stderr lines both arrive as logs; progress does not.
    assert "from ds.log" in reply["logs"] and "from print" in reply["logs"]
    assert not any("half way" in line for line in reply["logs"])
    assert reply["tool_seconds"] == 0.01 and reply["seconds"] >= 0


def test_schema_one_has_no_output_contract(fake, tool, fixture_dir):
    reply = launch.run(fake, str(tool), dict(MANIFEST, schema=1), fixture_dir,
                       {"mode": "ok"})
    assert reply["result"]["seen"]["request"]["output_contract"] is None


def test_cpu_runs_hide_the_gpu(fake, tool, fixture_dir, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    cpu = _run(fake, tool, fixture_dir, "ok", device="cpu")["result"]["seen"]
    assert cpu["POINTILLVIZ_DEVICE"] == "cpu"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""

    gpu = _run(fake, tool, fixture_dir, "ok", device="cuda")["result"]["seen"]
    assert gpu["POINTILLVIZ_DEVICE"] == "cuda"
    assert gpu["CUDA_VISIBLE_DEVICES"] == "3"               # the server's own choice stands
    assert gpu["request"]["device"] == "cuda"

    with pytest.raises(ValueError):
        _run(fake, tool, fixture_dir, "ok", device="gpu")


def test_tool_refusal_is_passed_through(fake, tool, fixture_dir):
    reply = _run(fake, tool, fixture_dir, "refuse")
    assert reply["status"] == "error"
    assert reply["code"] == "bad_input"
    assert reply["message"] == "No cells to work on."
    assert reply["traceback"] == "Traceback: here"
    assert reply["tool_seconds"] == 0.02
    assert "logs" in reply


def test_exit_without_answer_attaches_stderr(fake, tool, fixture_dir):
    reply = _run(fake, tool, fixture_dir, "crash")
    assert reply["code"] == "no_result"
    assert "status 3" in reply["message"] and "Fake Tool" in reply["message"]
    assert "Segmentation fault" in reply["stderr"]
    assert "Fatal Python error: Segmentation fault" in reply["logs"]


def test_launcher_that_cannot_start(tool, fixture_dir, tmp_path):
    reply = launch.run([str(tmp_path / "no-such-python")], str(tool), MANIFEST,
                       fixture_dir, {})
    assert reply["code"] == "no_subprocess"
    assert reply["logs"] == []


def test_stderr_flood_is_capped(fake, tool, fixture_dir):
    from pointillviz_tool.results import MAX_LOG_LINES
    reply = _run(fake, tool, fixture_dir, "flood")
    assert reply["status"] == "ok"
    assert len(reply["logs"]) == MAX_LOG_LINES + 1
    assert reply["logs"][0] == "line 0"
    assert reply["logs"][-1] == f"(log truncated at {MAX_LOG_LINES} lines)"


def test_progress_bar_keeps_its_last_frame(fake, tool, fixture_dir):
    reply = _run(fake, tool, fixture_dir, "bar")
    assert reply["logs"] == ["100%|####|"]


def test_answer_then_linger_is_still_an_answer(fake, tool, fixture_dir):
    started = time.monotonic()
    reply = _run(fake, tool, fixture_dir, "linger")
    assert reply["status"] == "ok"
    assert time.monotonic() - started < 30


def test_exit_while_a_child_holds_stdout_does_not_hang(fake, tool, fixture_dir, tmp_path):
    pidfile = tmp_path / "orphan.pid"
    started = time.monotonic()
    reply = _run(fake, tool, fixture_dir, "orphan", params={"pidfile": str(pidfile)})
    try:
        assert reply["code"] == "no_result"
        assert time.monotonic() - started < 30
    finally:
        try:
            os.kill(int(pidfile.read_text()), 9)
        except (OSError, ValueError):
            pass


@pytest.mark.skipif(os.name == "nt", reason="checks POSIX process groups")
def test_cancel_kills_the_whole_group(fake, tool, fixture_dir, tmp_path):
    pidfile = tmp_path / "pids"
    out = {}

    def start():
        out["reply"] = _run(fake, tool, fixture_dir, "hang", run_id="run-1",
                            params={"pidfile": str(pidfile)})

    worker = threading.Thread(target=start)
    worker.start()
    assert _wait_for(lambda: pidfile.exists() and len(pidfile.read_text().split()) == 2)
    leader, grandchild = map(int, pidfile.read_text().split())

    assert launch.cancel("run-1") is True
    worker.join(30)
    assert not worker.is_alive()
    assert out["reply"]["status"] == "error"
    assert out["reply"]["code"] == "cancelled"
    assert out["reply"]["message"] == "Fake Tool was cancelled."
    assert _wait_for(lambda: not _alive(grandchild), 10), "the grandchild survived the cancel"
    assert not _alive(leader)
    # A finished run is forgotten, and an unknown id is not an error.
    assert launch.cancel("run-1") is False
    assert launch.cancel("never-started") is False


@pytest.mark.skipif(importlib.util.find_spec("pointillviz_tool.runner") is None,
                    reason="the runner is not in this checkout yet")
def test_real_runner_round_trip(tool, fixture_dir):
    manifest = dict(MANIFEST, outputs=["summary"], optional_outputs=[], params=[])
    reply = launch.run([sys.executable], str(tool), manifest, fixture_dir, {})
    assert reply["status"] == "ok", reply
    assert reply["result"]["summary"] == "hi"
