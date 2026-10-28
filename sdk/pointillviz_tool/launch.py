"""The parent side of a tool run: start the child, read its protocol, cancel it.

    reply = launch.run(envs.launcher(tool_dir, "own", "cpu"), tool_dir,
                       manifest, dataset_dir, params, run_id="r1")
    launch.cancel("r1")          # from another thread, while it runs

One driver for every caller. The host runs a user's click through it and the
validator runs its gates through it, so a gate certifies the execution path
production actually uses rather than a second copy of the same
fork-read-kill dance.

Running is a subprocess, never a thread
---------------------------------------
The child is `<launcher> -m pointillviz_tool.runner <request.json>`, where the
launcher starts Python in the tool's own pixi environment or is the current
interpreter (`envs.launcher`). It starts in a new process group and is
cancelled by GROUP: a tool that spawns children (a `multiprocessing` pool, an
R process, the python that `pixi run` starts) would otherwise leave them
running after a cancel, and — because they inherit the pipes — leave the
reader threads waiting on an EOF that can never arrive.

There is no wall clock and no memory cap. How long a tool may take and how
much it may use is its author's business; the person who started a run stops
it with `cancel(run_id)`.

What the child is given
-----------------------
  * `PYTHONPATH`: the folder holding this package, then `extra_paths` (the
    app root, for legacy tools that import `backend.*`). Set, not appended
    to: the tool's environment is what its pixi.toml declares, and a
    PYTHONPATH inherited from the server would put the server's packages,
    built for another interpreter, in front of it.
  * not the tool's folder, on PYTHONPATH or as the folder it starts in.
    Both come before the standard library on sys.path, so a helper the tool
    ships under the name of a module the runner needs (`select.py`) would
    replace that module and kill the runner before it could say why. The
    child starts in this package's own folder, which PYTHONPATH already
    holds, and the runner moves into the tool's folder, and puts it first on
    sys.path, once its own imports are done.
  * `POINTILLVIZ_DEVICE`: "cpu" or "cuda", the device the request also names
    and the tool sees as `ds.device`. A CPU run also sets
    `CUDA_VISIBLE_DEVICES=""`, so that a library which picks a GPU on its own
    cannot see one.
  * no stdin: with no clock running, a tool that waits for input would wait
    forever.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time

from .dataset import DEVICES
from .results import MAX_LOG_LINES

#: Grace between a tool's terminal line and the group kill that follows it,
#: and between a child's exit and giving up on the rest of its output.
_EXIT_GRACE_S = 2.0
#: Tail of the child's stderr kept for a crash report.
_STDERR_TAIL = 4000
#: How often a child that has not answered yet is checked for having left.
_POLL_S = 0.5

# run_id -> the live runs started under it. A list, so that a caller who
# reuses an id still cancels everything it started under that name.
_RUNS: dict = {}
_RUNS_LOCK = threading.Lock()
# The private copy `sdk_root` makes of a pip-installed SDK.
_staged = None
_STAGE_LOCK = threading.Lock()


def sdk_root() -> str:
    """A directory that contains the `pointillviz_tool` package and nothing else of note.

    Put first on the child's PYTHONPATH, so a tool environment needs no copy
    of the SDK: the runner and the tool's `from pointillviz_tool import ...`
    resolve to the very files the parent is running.

    In a checkout, or PointillViz's vendored copy, that is the folder above
    the package. In an SDK installed with pip it would be site-packages,
    whose numpy and scipy (installed beside the SDK as its dependencies)
    were built for the installer's Python: first on a tool environment's
    path, they shadow its own and every run dies importing numpy. So the
    package alone is copied, once per process, to a private temporary folder.
    """
    package = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(package)
    if os.path.basename(root).lower() not in ("site-packages", "dist-packages"):
        return root
    global _staged
    with _STAGE_LOCK:
        if _staged is None or not os.path.isfile(
                os.path.join(_staged, "pointillviz_tool", "runner.py")):
            _staged = tempfile.mkdtemp(prefix="pointillviz_tool_")
            shutil.copytree(package, os.path.join(_staged, "pointillviz_tool"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            atexit.register(shutil.rmtree, _staged, True)
    return _staged


def run(launcher, plugin_dir: str, manifest: dict, dataset_dir: str,
        params: dict, *, subset=None, section_filter=None,
        atlas_composition=None, sweep: dict = None, device: str = "cpu",
        run_id: str = None, extra_paths=(), module_name: str = None) -> dict:
    """One tool, one child process, the whole newline-JSON protocol.

    Always returns a dict: `{status:"ok", result, logs, seconds,
    tool_seconds}` or `{status:"error", code, message, logs, seconds?,
    stderr?, traceback?}`. The parent's codes are `no_subprocess` (the
    launcher could not be started), `no_result` (the child left without a
    terminal line) and `cancelled`; any other code is the child's own.

    `launcher` is the argv prefix that starts Python (`envs.launcher`).
    `params` are passed as given: coercing them against the manifest is the
    caller's job. `sweep` ({param, values}) runs every value in this one
    child. `run_id`, when given, is the name `cancel` stops this run by.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}.")
    plugin_dir = os.path.abspath(plugin_dir)
    req = {
        "plugin_dir": plugin_dir,
        "entry": manifest["entry"],
        "dataset_dir": os.path.abspath(dataset_dir),
        "params": params,
        "subset": None if subset is None else [int(i) for i in subset],
        "section_filter": None if section_filter is None else int(section_filter),
        "atlas_composition": atlas_composition,
        # Schemas 2 and 3 promise their declared outputs and nothing else;
        # schema 1 predates that promise.
        "output_contract": manifest if manifest.get("schema", 1) >= 2 else None,
        # One module name per tool: two tools whose entry files are both
        # `tool.py` must not collide in the child's sys.modules.
        "module_name": module_name or "pv_tool_" + manifest["id"].replace("-", "_"),
        "sweep": sweep,
        "device": device,
    }
    # A file rather than stdin: the request carries a lasso selection that can
    # be millions of indices, and writing that down a pipe the child is
    # streaming progress back on is a deadlock waiting to be measured.
    fd, req_path = tempfile.mkstemp(prefix="pv_run_", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(req, fh)

    started = time.monotonic()
    try:
        try:
            child = start([*launcher, "-m", "pointillviz_tool.runner", req_path],
                          child_env(device, extra_paths), run_id)
        except OSError as exc:
            return _fail("no_subprocess", f"Could not start the tool: {exc}",
                         logs=[])
        proc = child.proc
        try:
            readers = [threading.Thread(target=_pump_stdout, args=(child,), daemon=True),
                       threading.Thread(target=_pump_stderr, args=(child,), daemon=True)]
            for t in readers:
                t.start()
            # Wait for the terminal line (or EOF), with no deadline. The poll
            # covers a child that left while something it started still holds
            # its stdout open, which would otherwise keep EOF away for good.
            while not child.done.wait(_POLL_S):
                if proc.poll() is not None:
                    child.done.wait(_EXIT_GRACE_S)
                    break
            try:
                proc.wait(timeout=_EXIT_GRACE_S)
            except subprocess.TimeoutExpired:
                # It answered and then did not leave. The run is over either
                # way, and whatever is still alive is holding a pipe.
                _kill_group(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            # The readers finish at EOF, which comes once the group has gone.
            # One shared grace: something the tool left behind may hold the
            # pipes, and its output is not worth more than that wait.
            deadline = time.monotonic() + _EXIT_GRACE_S
            for t in readers:
                t.join(max(0.0, deadline - time.monotonic()))
        finally:
            child.release()
        seconds = round(time.monotonic() - started, 3)
        logs = child.snapshot()
        terminal = child.terminal

        if terminal.get("t") == "result":
            # `seconds` is the wall clock the user waited, interpreter (and
            # environment) start-up and all — the number the panel shows. The
            # child's own measurement is kept beside it because an author
            # tuning their tool wants the part they can actually make faster.
            return {"status": "ok",
                    "result": terminal.get("result") or {},
                    "logs": logs,
                    "seconds": seconds,
                    "tool_seconds": terminal.get("seconds")}
        if terminal.get("t") == "error":
            out = _fail(str(terminal.get("code") or "error"),
                        str(terminal.get("message") or "The tool reported an error."),
                        logs=logs, seconds=seconds,
                        tool_seconds=terminal.get("seconds"))
            if terminal.get("traceback"):
                out["traceback"] = terminal["traceback"]
            return out
        if child.cancelled:
            return _fail("cancelled", f"{manifest['name']} was cancelled.",
                         logs=logs, seconds=seconds)
        # No terminal line is a failure, never a success: a segfault, an
        # os._exit(), or an import that killed the interpreter all land here,
        # and the tool's own output plus any C-level message is in stderr —
        # which is why its tail is attached rather than discarded.
        return _fail("no_result",
                     f"{manifest['name']} exited (status {proc.returncode}) "
                     "without returning a result.",
                     logs=logs, seconds=seconds,
                     stderr=child.tail[-_STDERR_TAIL:])
    finally:
        try:
            os.unlink(req_path)
        except OSError:
            pass


def cancel(run_id: str) -> bool:
    """Stop the run started under `run_id`. True if one was running.

    Kills the process group rather than asking nicely: the person pressing
    Cancel wants the computer back, and a tool stuck in a C extension or a GPU
    kernel cannot be interrupted from inside anyway. The `run` call then
    returns `{code: "cancelled"}` to whoever started it — unless the tool had
    already answered, in which case its answer stands.
    """
    with _RUNS_LOCK:
        children = list(_RUNS.get(run_id, ()))
    for child in children:
        child.cancelled = True
        _kill_group(child.proc)
    return bool(children)


def child_env(device: str = "cpu", extra_paths=()) -> dict:
    """The environment of every process that runs a tool's code: what the
    module docstring lists under "What the child is given". The gates' import
    probe gets it too, so the import it certifies is the import a run makes."""
    env = dict(os.environ)
    # Absolute, because the child does not start where the caller is.
    env["PYTHONPATH"] = os.pathsep.join(
        [sdk_root(), *(os.path.abspath(p) for p in extra_paths)])
    env["POINTILLVIZ_DEVICE"] = device
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def start(argv: list, env: dict, run_id: str = None) -> "_Child":
    """Start one process that runs a tool's code, as every such process starts.

    Its own process group, so a cancel can take everything it started; this
    package's folder as its working directory (see the module docstring); no
    stdin, and stdout and stderr piped. Until its `release()`, `cancel(run_id)`
    stops it. Raises OSError when `argv` cannot be started.
    """
    if os.name == "nt":
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        group = {"start_new_session": True}
    proc = subprocess.Popen(argv, cwd=sdk_root(), env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **group)
    child = _Child(proc, run_id)
    _register(run_id, child)
    return child


class _Child:
    """One child process and what its two reader threads collect from it."""

    def __init__(self, proc, run_id=None):
        self.proc = proc
        self.run_id = run_id
        self.logs: list = []
        self.tail = ""
        self.terminal: dict = {}
        self.done = threading.Event()
        self.cancelled = False
        self._lock = threading.Lock()

    def release(self) -> None:
        """Forget it: `cancel` no longer reaches a process that has finished."""
        _unregister(self.run_id, self)

    def log(self, line: str) -> None:
        # The child bounds its own log(); this bounds everything else it can
        # write, which is unbounded by construction. Both readers append here,
        # so the cap is checked under a lock.
        with self._lock:
            if len(self.logs) < MAX_LOG_LINES:
                self.logs.append(line)
            elif len(self.logs) == MAX_LOG_LINES:
                self.logs.append(f"(log truncated at {MAX_LOG_LINES} lines)")

    def snapshot(self) -> list:
        with self._lock:
            return list(self.logs)


def _register(run_id, child) -> None:
    if run_id is None:
        return
    with _RUNS_LOCK:
        _RUNS.setdefault(run_id, []).append(child)


def _unregister(run_id, child) -> None:
    with _RUNS_LOCK:
        children = _RUNS.get(run_id)
        if children and child in children:
            children.remove(child)
            if not children:
                del _RUNS[run_id]


def _pump_stdout(child: _Child) -> None:
    """Parse the child's newline-delimited protocol until its terminal line."""
    try:
        for raw in child.proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                kind = msg.get("t") if isinstance(msg, dict) else None
            except ValueError:
                msg, kind = None, None
            if kind == "log":
                child.log(str(msg.get("m", "")))
            elif kind == "progress":
                # Parsed and dropped: a run is one request and one reply, so
                # there is no consumer yet. Parsing it anyway keeps a tool that
                # reports progress from being mistaken for one printing rubbish.
                continue
            elif kind in ("result", "error"):
                child.terminal.update(msg)
                break
            else:
                # The runner writes the protocol to a private descriptor, so a
                # tool's print() cannot land here. Anything that still does is
                # kept as a log line rather than failing the run over it.
                child.log(line)
    except (ValueError, OSError):
        pass                        # the stream was closed under us; the
                                    # outcome is decided by the caller either way
    finally:
        # Set on EOF as well as on a terminal line, so a child that dies
        # without answering is reported at once.
        child.done.set()
        _close(child.proc.stdout)


def _pump_stderr(child: _Child) -> None:
    """Turn the child's stderr into log lines, and keep its tail for a crash report.

    On a thread, and not merely read at the end, because the pipe buffer is
    64 kB: a tool that writes more than that and is never read from blocks in
    write(), which is indistinguishable from a hang.
    """
    try:
        for raw in child.proc.stderr:
            text = raw.decode("utf-8", "replace")
            child.tail = (child.tail + text)[-_STDERR_TAIL:]
            # A progress bar redraws itself with \r and ends its line once, at
            # the end: keep what a terminal would be showing, not every frame.
            line = text.rstrip("\r\n").rsplit("\r", 1)[-1].rstrip()
            if line.strip():
                child.log(line)
    except (ValueError, OSError):
        pass
    finally:
        _close(child.proc.stderr)


def _close(stream) -> None:
    # Each reader closes its own stream when it is done with it. Closing one
    # from another thread would block until that thread's read returns, which
    # is never while something the tool started still holds the pipe open.
    try:
        stream.close()
    except (ValueError, OSError):
        pass


def _kill_group(proc) -> None:
    """Kill the child's whole process group (POSIX) or process tree (Windows).

    The group, not the process: the child was started as a group leader, so
    everything it spawned is in that group. Killing only the child would leave
    a tool's worker pool running with the server's privileges.

    Only while the child is unreaped. A reaped pid can be recycled, and a
    killpg against a recycled group id would kill something unrelated; an
    un-waited zombie still owns its pid, so the id cannot be reused here.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # Windows has no group to signal; taskkill walks the tree instead.
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _fail(code: str, message: str, **extra) -> dict:
    out = {"status": "error", "code": code, "message": message}
    out.update({k: v for k, v in extra.items() if v not in (None, "")})
    return out
