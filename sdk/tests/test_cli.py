"""`python -m pointillviz_tool`: new, validate, pack and run, with their exit statuses.

Each test runs the command as an author or CI would: a child process with the
SDK on PYTHONPATH. Tools run under this interpreter (`--env app`), so these
need no pixi.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

from pointillviz_tool import launch

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "tools")


def cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=launch.sdk_root())
    return subprocess.run([sys.executable, "-m", "pointillviz_tool", *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)


def _copy(tmp_path, name="py-cells"):
    dest = tmp_path / name
    shutil.copytree(os.path.join(TOOLS, name), dest)
    return dest


def test_no_command_prints_the_usage():
    proc = cli()
    assert proc.returncode == 2
    for command in ("new", "validate", "pack", "run"):
        assert command in proc.stderr


def test_new_writes_the_tool_and_says_what_to_do_next(tmp_path):
    proc = cli("new", "fresh-tool", "-o", tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "fresh-tool"
    assert proc.stdout.splitlines()[0] == str(out)
    assert "pixi lock" in proc.stdout and "pointillviz_tool validate ." in proc.stdout
    assert (out / "tool.py").is_file()

    proc = cli("new", "fresh-r-tool", "--r", "-o", tmp_path)
    assert proc.returncode == 0 and (tmp_path / "fresh-r-tool" / "tool.R").is_file()

    proc = cli("new", "fresh-tool", "-o", tmp_path)
    assert proc.returncode == 2 and "already exists" in proc.stderr


def test_validate_exits_0_only_when_the_report_is_ok(tmp_path):
    tool = _copy(tmp_path)
    proc = cli("validate", tool, "--json", "--env", "app")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] and report["plugin_id"] == "py-cells"

    proc = cli("validate", tool)
    assert proc.returncode == 0 and "PASSED" in proc.stdout

    (tool / "tool.py").write_text("def run(ds, params):\n    return 1 / 0\n")
    proc = cli("validate", tool)
    assert proc.returncode == 1
    assert "FAILED" in proc.stdout and "py-cells: validation failed" in proc.stderr

    proc = cli("validate", tmp_path / "nowhere")
    assert proc.returncode == 2 and "no such folder or file" in proc.stderr


def test_pack_writes_the_bundle_beside_the_folder(tmp_path):
    tool = _copy(tmp_path)
    (tool / "__pycache__").mkdir()
    (tool / "__pycache__" / "tool.cpython-311.pyc").write_bytes(b"\0")
    proc = cli("pack", tool)
    assert proc.returncode == 0, proc.stderr
    bundle = tmp_path / "py-cells-1.0.0.pvtool"
    assert proc.stdout.splitlines()[0] == str(bundle)
    with zipfile.ZipFile(bundle) as zf:
        assert sorted(zf.namelist()) == ["plugin.json", "tool.py"]

    proc = cli("pack", tool, "-o", tmp_path / "elsewhere.pvtool")
    assert proc.returncode == 0 and (tmp_path / "elsewhere.pvtool").is_file()
    assert cli("pack", tmp_path / "nowhere").returncode == 2


def test_run_prints_the_reply(tmp_path, fixture_dir):
    tool = _copy(tmp_path)
    proc = cli("run", tool, "--dataset", fixture_dir, "--params", '{"k": 7}')
    assert proc.returncode == 0, proc.stderr
    reply = json.loads(proc.stdout)
    assert reply["status"] == "ok" and reply["result"]["summary"].endswith("k=7.")
    assert reply["logs"] == ["6 cell types at k=7"]

    # Out of range is clamped, exactly as PointillViz coerces a form.
    reply = json.loads(cli("run", tool, "--dataset", fixture_dir, "--params", '{"k": 999}').stdout)
    assert reply["result"]["summary"].endswith("k=50.")


def test_run_refuses_what_it_cannot_run(tmp_path, fixture_dir):
    tool = _copy(tmp_path)
    proc = cli("run", tool, "--dataset", fixture_dir, "--params", "[1, 2]")
    assert proc.returncode == 2 and "JSON object" in proc.stderr
    proc = cli("run", tool, "--dataset", tmp_path)
    assert proc.returncode == 2 and "not a converted dataset" in proc.stderr
    proc = cli("run", tool, "--dataset", fixture_dir, "--env", "own")
    assert proc.returncode == 2 and "no pixi.toml" in proc.stderr

    (tool / "tool.py").write_text("from pointillviz_tool import PluginError\n"
                                  "def run(ds, params):\n"
                                  "    raise PluginError('nope', 'Not on this dataset.')\n")
    proc = cli("run", tool, "--dataset", fixture_dir)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["code"] == "nope"
