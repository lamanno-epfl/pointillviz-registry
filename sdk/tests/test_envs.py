"""pixi helpers: finding pixi, picking the launcher, lock checks, sharing PointillViz's environment.

The first half needs no pixi. The second half runs the real pixi CLI on small
temporary workspaces (it needs network access, or a warm package cache, to
solve), and the GPU tests at the end run only on a machine with an NVIDIA GPU.
"""

import importlib.util
import os
import platform
import shutil
import subprocess
import sys

import pytest

from pointillviz_tool import envs, launch, manifest
from pointillviz_tool.dataset import PluginError

REGISTRY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU_PROBE = os.path.join(REGISTRY, "examples", "gpu-probe")


def _current_platform() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        return "linux-aarch64" if machine in ("aarch64", "arm64") else "linux-64"
    if sys.platform == "darwin":
        return "osx-arm64" if machine == "arm64" else "osx-64"
    return "win-64"


def _workspace(path, name, dependencies="", extra=""):
    """A minimal pixi workspace for this machine's platform."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "pixi.toml"), "w") as fh:
        fh.write(f'[workspace]\nname = "{name}"\nchannels = ["conda-forge"]\n'
                 f'platforms = ["{_current_platform()}"]\n\n'
                 f"[dependencies]\n{dependencies}\n{extra}")
    return str(path)


def _pixi_or_skip() -> str:
    try:
        return envs.pixi_exe()
    except PluginError:
        pytest.skip("pixi is not installed")


def _gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    return bool(smi) and subprocess.run([smi, "-L"], capture_output=True).returncode == 0


# ===========================================================================
# No pixi needed
# ===========================================================================
def test_notices_nobody_can_act_on_stay_out_of_the_log():
    # pixi 0.83's own words: the deprecation a GPU tool's pixi.toml triggers,
    # and the unused-feature notes of a shared install's first steps.
    stderr = """\
 WARN Encountered 1 warning while parsing the manifest:
  \u26a0 the `[system-requirements]` table is deprecated in favor of virtual
  \u2502 packages on `platforms`
    \u256d\u2500[tools/gpu-tool/pixi.toml:19:1]
 19 \u2502 \u256d\u2500\u25b6 [feature.gpu.system-requirements]
  help: e.g. platforms = [{ name = "my-linux-64", platform = "linux-64", cuda
        = "12" }]

The default environment has been installed.
Added six *
warning: The feature tool-x is not used in any environment, so the added dependencies were not resolved or pinned
  Add it to an environment with `pixi workspace environment add <environment> --feature tool-x`
 WARN Encountered 1 warning while parsing the manifest:
  \u26a0 The feature 'tool-x' is defined but not used in any environment.
  help: Remove the feature from the manifest or add it to an environment

Added conda-forge (https://conda.anaconda.org/conda-forge/)
 WARN a warning of another kind
Updated environment default"""
    lines = []
    emit = envs._without_notices(lines.append)
    for line in stderr.split("\n"):
        emit(line)
    assert lines == ["The default environment has been installed.", "Added six *",
                     "Added conda-forge (https://conda.anaconda.org/conda-forge/)",
                     " WARN a warning of another kind", "Updated environment default"]


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX executables")
def test_pixi_exe_search_order(tmp_path, monkeypatch):
    configured = _executable(tmp_path / "configured" / "pixi")
    on_path = _executable(tmp_path / "bin" / "pixi")
    in_home = _executable(tmp_path / "home" / "bin" / "pixi")
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "home"))

    monkeypatch.setenv("POINTILLVIZ_PIXI", configured)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert envs.pixi_exe() == configured

    monkeypatch.delenv("POINTILLVIZ_PIXI")
    assert envs.pixi_exe() == on_path

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert envs.pixi_exe() == in_home

    # The installer's default place, ~/.pixi/bin, when PIXI_HOME is unset.
    monkeypatch.delenv("PIXI_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    user_pixi = _executable(tmp_path / "user" / ".pixi" / "bin" / "pixi")
    assert envs.pixi_exe() == user_pixi

    os.remove(user_pixi)
    with pytest.raises(PluginError) as missing:
        envs.pixi_exe()
    assert missing.value.code == "no_pixi"

    # A configured path that is wrong is reported, not silently passed over.
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("POINTILLVIZ_PIXI", str(tmp_path / "nowhere" / "pixi"))
    with pytest.raises(PluginError) as wrong:
        envs.pixi_exe()
    assert wrong.value.code == "no_pixi" and "POINTILLVIZ_PIXI" in wrong.value.message


def _launch_argv(pixi, tool, env):
    return [pixi, "run", "--quiet", "--frozen", "--manifest-path",
            os.path.join(tool, "pixi.toml"), "-e", env, "python"]


def test_cpu_and_app_launchers(tmp_path, monkeypatch):
    # Neither asks pixi anything, so a stand-in that cannot run will do.
    fake_pixi = str(tmp_path / "pixi")
    open(fake_pixi, "w").close()
    monkeypatch.setenv("POINTILLVIZ_PIXI", fake_pixi)

    bare = tmp_path / "bare"
    bare.mkdir()
    assert not envs.has_pixi(str(bare))
    assert envs.tool_environments(str(bare)) == []

    tool = _workspace(tmp_path / "tool", "tool", 'python = ">=3.10"')
    assert envs.has_pixi(tool)
    assert envs.launcher(tool, "own", "cpu") == _launch_argv(fake_pixi, tool, "default")
    assert envs.launcher(tool, "app", "cuda") == [sys.executable]
    with pytest.raises(ValueError):
        envs.launcher(tool, "shared", "cpu")


def test_shared_mode_without_tomllib_says_so(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)        # as on Python 3.10
    app = _workspace(tmp_path / "app", "app")
    with pytest.raises(PluginError) as exc:
        envs.remove_shared(os.path.join(app, "pixi.toml"), "my-tool")
    assert exc.value.code == "env_failed" and "Python 3.11" in exc.value.message


def test_app_manifest(tmp_path, monkeypatch):
    for var in ("PIXI_WORKSPACE_MANIFEST", "PIXI_PROJECT_MANIFEST"):
        monkeypatch.delenv(var, raising=False)
    assert envs.app_manifest() is None
    app = _workspace(tmp_path / "app", "app")
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", os.path.join(app, "pixi.toml"))
    assert envs.app_manifest() == os.path.join(app, "pixi.toml")
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(tmp_path / "gone" / "pixi.toml"))
    assert envs.app_manifest() is None
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(tmp_path / "app" / "pyproject.toml"))
    assert envs.app_manifest() is None


def test_requirements_become_pixi_add_specs():
    assert envs._conda_spec("numpy", "*") == "numpy *"
    assert envs._conda_spec("scipy", ">=1.10,<2") == "scipy >=1.10,<2"
    assert envs._conda_spec("pytorch", {"version": ">=2", "build": "cuda*"}) == "pytorch >=2 cuda*"
    assert envs._conda_spec("pytorch", {"build": "cuda*"}) == "pytorch * cuda*"
    assert envs._conda_spec("scanpy", {"version": "*", "channel": "bioconda"}) == "bioconda::scanpy *"
    assert envs._pypi_spec("tqdm", "*") == "tqdm"
    assert envs._pypi_spec("tqdm", ">=4") == "tqdm>=4"
    assert envs._pypi_spec("rich", {"version": ">=13", "extras": ["jupyter"]}) == "rich[jupyter]>=13"
    assert envs._pypi_spec("rich", {"extras": ["a", "b"]}) == "rich[a,b]"
    for bad in (lambda: envs._conda_spec("x", {"path": "../x"}),
                lambda: envs._pypi_spec("y", {"git": "https://example.org/y.git"})):
        with pytest.raises(PluginError) as exc:
            bad()
        assert exc.value.code == "env_failed" and "own environment" in exc.value.message


def test_changes_keep_only_existing_packages():
    def conda(name, version, build):
        return {"conda": f"https://conda.anaconda.org/conda-forge/linux-64/{name}-{version}-{build}.conda"}

    report = {"version": 1, "environment": {"default": {
        "linux-64": [
            {"name": "tqdm", "before": None, "after": conda("tqdm", "4.67.1", "pyhd8ed1ab_0")},
            {"name": "scipy", "before": conda("scipy", "1.17.1", "py311_1"),
             "after": conda("scipy", "1.14.1", "py311_2")},
            {"name": "old-thing", "before": conda("old-thing", "0.3", "h1_0"), "after": None},
            {"name": "libblas", "before": conda("libblas", "3.11.0", "11_openblas"),
             "after": conda("libblas", "3.11.0", "11_mkl")},
            {"name": "six", "before": {"pypi": "https://files/six-1.16.0-py2.py3-none-any.whl",
                                       "version": "1.16.0"},
             "after": {"pypi": "https://files/six-1.17.0-py2.py3-none-any.whl", "version": "1.17.0"}},
        ],
        "osx-arm64": [
            {"name": "scipy", "before": conda("scipy", "1.17.1", "py311_1"),
             "after": conda("scipy", "1.14.1", "py311_2")},
        ],
    }}}
    assert envs._changes(report) == [
        {"name": "scipy", "before": "1.17.1", "after": "1.14.1"},
        {"name": "old-thing", "before": "0.3", "after": None},
        {"name": "libblas", "before": "libblas-3.11.0-11_openblas.conda",
         "after": "libblas-3.11.0-11_mkl.conda"},
        {"name": "six", "before": "1.16.0", "after": "1.17.0"},
    ]
    assert envs._changes({"version": 1, "environment": {}}) == []


# ===========================================================================
# With pixi
# ===========================================================================
def test_a_gpu_run_uses_the_gpu_environment_when_there_is_one(tmp_path):
    pixi = _pixi_or_skip()
    cpu_only = _workspace(tmp_path / "cpu-only", "cpu-only", 'python = ">=3.10"')
    with_gpu = _workspace(tmp_path / "with-gpu", "with-gpu", 'python = ">=3.10"',
                          '[feature.gpu.dependencies]\npytorch = "*"\n\n'
                          '[feature.extra.dependencies]\nrich = "*"\n\n'
                          '[environments]\ngpu = ["gpu"]\nextra = ["extra"]\n')
    # pixi reads the manifest, not tomllib, so this also holds on Python 3.10.
    assert envs.tool_environments(cpu_only) == ["default"]
    assert envs.tool_environments(with_gpu) == ["default", "gpu", "extra"]
    assert envs.launcher(with_gpu, "own", "cuda") == _launch_argv(pixi, with_gpu, "gpu")
    assert envs.launcher(with_gpu, "own", "cpu") == _launch_argv(pixi, with_gpu, "default")
    assert envs.launcher(cpu_only, "own", "cuda") == _launch_argv(pixi, cpu_only, "default")

    # pixi lists nothing for a manifest it cannot read; building it says why.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pixi.toml").write_text("[workspace\nname = 'x'\n")
    assert envs.tool_environments(str(broken)) == ["default"]
    with pytest.raises(PluginError) as exc:
        envs.install_dedicated(str(broken))
    assert exc.value.code == "env_failed" and "pixi.toml" in exc.value.message


def test_lock_check(tmp_path):
    pixi = _pixi_or_skip()
    tool = _workspace(tmp_path / "tool", "tool")       # no packages: locks offline
    ok, message = envs.lock_check(tool)
    assert not ok and "pixi.lock is missing" in message

    subprocess.run([pixi, "lock", "--manifest-path", os.path.join(tool, "pixi.toml")],
                   check=True, capture_output=True)
    assert envs.lock_check(tool) == (True, "pixi.lock matches pixi.toml.")

    # A requirement the lock does not cover (one tiny noarch package).
    with open(os.path.join(tool, "pixi.toml"), "a") as fh:
        fh.write('\n[feature.more.dependencies]\ntzdata = "*"\n\n[environments]\nmore = ["more"]\n')
    ok, message = envs.lock_check(tool)
    assert not ok and message.startswith("pixi.lock does not match pixi.toml.")


def test_install_dedicated_reports_pixi_failure(tmp_path):
    _pixi_or_skip()
    tool = _workspace(tmp_path / "tool", "tool", 'python = ">=3.10"')   # never locked
    lines = []
    with pytest.raises(PluginError) as exc:
        envs.install_dedicated(tool, log=lines.append)
    assert exc.value.code == "env_failed"
    assert "lock file" in exc.value.message
    assert lines, "pixi's own output should reach the log"


APP_DEPS = 'python = "3.11.*"\nnumpy = "2.2.*"\nthreadpoolctl = "*"\n'


@pytest.fixture
def app(tmp_path):
    pixi = _pixi_or_skip()
    path = _workspace(tmp_path / "app", "app", APP_DEPS)
    subprocess.run([pixi, "lock", "--manifest-path", os.path.join(path, "pixi.toml")],
                   check=True, capture_output=True)
    return os.path.join(path, "pixi.toml")


def _files(manifest_path):
    folder = os.path.dirname(manifest_path)
    return tuple(open(os.path.join(folder, n)).read() for n in ("pixi.toml", "pixi.lock"))


def test_check_shared(app, tmp_path):
    before = _files(app)

    # Only additions, a conda package and a PyPI one: consistent, no warning.
    added = _workspace(tmp_path / "added", "added",
                       'python = ">=3.10"\nnumpy = "*"\ntqdm = "*"\n',
                       '[pypi-dependencies]\nsix = ">=1.16"\n')
    assert envs.check_shared(app, added, "added") == {
        "consistent": True, "changes": [], "error": None}

    # A package the app already has would move: inconsistent, and named.
    moved = _workspace(tmp_path / "moved", "moved", 'python = ">=3.10"\nthreadpoolctl = "<3.6"\n')
    report = envs.check_shared(app, moved, "moved")
    assert report["consistent"] is False and report["error"] is None
    assert [c["name"] for c in report["changes"]] == ["threadpoolctl"]
    assert report["changes"][0]["before"] != report["changes"][0]["after"]

    # The solve fails: inconsistent, with the solver's explanation.
    clash = _workspace(tmp_path / "clash", "clash", 'python = ">=3.10"\nnumpy = "<2"\n')
    report = envs.check_shared(app, clash, "clash")
    assert report["consistent"] is False and report["changes"] == []
    assert "numpy" in report["error"] and "solve" in report["error"]

    assert _files(app) == before, "the check must not touch the app's manifest or lock"


def test_apply_and_remove_shared(app, tmp_path):
    pixi = envs.pixi_exe()
    original = _files(app)
    subprocess.run([pixi, "install", "--frozen", "--manifest-path", app],
                   check=True, capture_output=True)

    def imports(module):
        return subprocess.run([pixi, "run", "--frozen", "--manifest-path", app, "python",
                               "-c", f"import {module}"], capture_output=True).returncode == 0

    tool = _workspace(tmp_path / "tool", "tool", 'python = ">=3.10"\ntqdm = "*"\n')
    lines = []
    envs.apply_shared(app, tool, "my-tool", log=lines.append)
    assert lines
    assert not any("not used in any environment" in line or "help:" in line
                   for line in lines), lines
    assert imports("tqdm")
    with_tool = _files(app)
    assert "tool-my-tool" in with_tool[0]

    # A tool that cannot be solved in: an error, and both files exactly as before.
    clash = _workspace(tmp_path / "clash", "clash", 'numpy = "<2"\n')
    with pytest.raises(PluginError) as exc:
        envs.apply_shared(app, clash, "clash")
    assert exc.value.code == "env_failed" and "numpy" in exc.value.message
    assert _files(app) == with_tool

    envs.remove_shared(app, "my-tool")
    assert _files(app)[0] == original[0]
    assert not imports("tqdm")
    envs.remove_shared(app, "my-tool")      # removing what is not there does nothing


# ===========================================================================
# GPU (this needs an NVIDIA GPU, and downloads CUDA PyTorch the first time)
# ===========================================================================
needs_gpu = pytest.mark.skipif(not _gpu(), reason="no NVIDIA GPU on this machine")


def _torch_report(device):
    """Python started by the launcher for `device`, as a run would start it."""
    env = dict(os.environ, POINTILLVIZ_DEVICE=device)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    code = "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
    out = subprocess.run(envs.launcher(GPU_PROBE, "own", device) + ["-c", code], env=env,
                         cwd=GPU_PROBE, capture_output=True, text=True, check=True)
    return out.stdout.split()[-2:]


@needs_gpu
def test_gpu_probe_launcher_reaches_cuda():
    _pixi_or_skip()
    assert envs.tool_environments(GPU_PROBE) == ["default", "gpu"]
    assert envs.lock_check(GPU_PROBE)[0]
    envs.install_dedicated(GPU_PROBE, "gpu")
    envs.install_dedicated(GPU_PROBE)

    available, cuda_build = _torch_report("cuda")
    assert available == "True" and cuda_build.startswith("12.")
    available, cuda_build = _torch_report("cpu")
    assert available == "False" and cuda_build == "None"     # the CPU build, in default


@needs_gpu
@pytest.mark.skipif(importlib.util.find_spec("pointillviz_tool.runner") is None,
                    reason="the runner is not in this checkout yet")
def test_gpu_probe_runs_on_both_devices(fixture_dir):
    _pixi_or_skip()
    m = manifest.load(GPU_PROBE)
    seen = {}
    for device in ("cuda", "cpu"):
        reply = launch.run(envs.launcher(GPU_PROBE, "own", device), GPU_PROBE, m,
                           fixture_dir, {"size": 512}, device=device)
        assert reply["status"] == "ok", reply
        # pixi's deprecation notice about [system-requirements] is not the tool's log.
        assert not any("system-requirements" in line for line in reply["logs"]), reply["logs"]
        seen[device] = dict(reply["result"]["table"]["rows"])
    assert seen["cuda"]["CUDA available"] == "yes"
    assert seen["cuda"]["pixi environment"] == "gpu-probe (gpu)"
    assert seen["cpu"]["CUDA available"] == "no"
    assert seen["cpu"]["pixi environment"] == "gpu-probe (default)"
