"""Runs ONE community tool, in its own process, and reports on a private channel.

    python -m pointillviz_tool.runner <request.json>

`launch.run` starts this, inside the tool's own pixi environment or
PointillViz's, and reads what it writes.

Why a subprocess at all
-----------------------
The alternative is importing the tool into the server. That would mean a tool's
`import` runs inside the process holding every cached graph and memmap; a tool
that spins forever holds a server thread forever; a tool that allocates without
bound takes the server's memory; a segfault in a C extension the tool imported
takes the whole viewer down. None of those are hypothetical for community code,
and all of them cost exactly one run here instead. It also lets the person who
started a run stop it: `launch.cancel` kills this process's group, which a
thread cannot be. There is no clock and no memory cap: how long a tool takes
and how much it uses is its author's business.

Protocol
--------
The request file: {plugin_dir, entry, dataset_dir, params, subset,
section_filter, atlas_composition, output_contract, module_name, sweep, device}.

Output: one JSON object per line, as things happen ({"t":"log","m":...} and
{"t":"progress","f":0.4,"m":...}), then exactly one {"t":"result", result,
logs, seconds} or {"t":"error", code, message, logs, seconds, traceback?} as
the LAST line. Streaming rather than one blob at the end, so a long tool can
say what it is doing.

The protocol travels on a PRIVATE channel: a duplicate of the stdout
descriptor the parent gave, made before anything else runs, after which
descriptor 1 itself is pointed at stderr. So everything a tool writes to
stdout, whether `print()`, a C library's printf, a child process or R's
`cat`, lands on stderr, which the parent shows as log lines and keeps a tail
of for crash reports. Nothing a tool prints can be taken for, or forge, a
protocol line.

A SWEEP ({param, values}) runs the tool once per value, in this one process
and against one Dataset (so a neighbour graph built at the first step serves
the rest), and returns {"sweep": {param, values, steps}}: each step is either
that value's result, validated exactly like a single run's, or {"error":
{code, message}} when the tool refused that value. A radius that finds no
pairs is an answer about that radius, not a failed sweep.

Native R tools (an entry ending in .R) are run by `r_bridge`, which starts R
beside this process and serves its `ds` calls from the same Dataset; what
comes back is checked exactly like a Python tool's return value.

Nothing here trusts the tool: the result goes through `check_result`, and for
schemas 2 and 3 through the manifest's output contract, before it is written,
so a non-conforming tool fails in this process with a clear message rather
than reaching the browser as a malformed payload.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback

from . import r_bridge
from .dataset import Dataset, PluginError
from .results import MAX_LOG_LINES, RESULT_CODES, check_result, output_mismatch

#: The protocol channel, opened by `main` before anything else runs.
_channel = None


def _protocol_channel():
    """Keep a private copy of stdout for the protocol, and send stdout itself to stderr.

    Descriptor-level, not `sys.stdout = sys.stderr`: a C extension, a child
    process and R all write to descriptor 1 without asking Python, and each
    would otherwise be able to break, or forge, a protocol line. The copy is
    not inheritable (os.dup), so no child process holds it either.
    """
    sys.stdout.flush()
    fd = os.dup(1)
    os.dup2(2, 1)
    try:
        # Line by line, so a tool's print() reaches the log in step with
        # its ds.log() lines rather than in one block at exit.
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    return open(fd, "w", encoding="utf-8", newline="\n")


def _send(line: str) -> None:
    _channel.write(line + "\n")
    _channel.flush()


def _emit(obj) -> None:
    _send(json.dumps(obj, allow_nan=False))


def main(argv) -> int:
    global _channel
    _channel = _protocol_channel()
    if len(argv) != 2:
        _emit({"t": "error", "code": "bad_invocation",
               "message": "usage: python -m pointillviz_tool.runner <request.json>"})
        return 2

    started = time.time()
    logs: list = []

    def log(message: str) -> None:
        # Bounded: a tool logging inside a per-cell loop would otherwise send
        # millions of lines through the pipe and into the panel.
        if len(logs) < MAX_LOG_LINES:
            logs.append(str(message))
            _emit({"t": "log", "m": str(message)})
        elif len(logs) == MAX_LOG_LINES:
            logs.append("...")
            _emit({"t": "log", "m": f"(log truncated at {MAX_LOG_LINES} lines)"})

    last = [0.0]

    def progress(fraction: float, message=None) -> None:
        # Rate-limited to ~10/s. A tool calling progress() every iteration is
        # the normal case, and every one of those is a write and a flush.
        now = time.time()
        if now - last[0] < 0.1 and fraction < 1.0:
            return
        last[0] = now
        _emit({"t": "progress", "f": max(0.0, min(1.0, float(fraction))),
               "m": None if message is None else str(message)})

    def fail(code: str, message: str, trace: str = None) -> int:
        out = {"t": "error", "code": code, "message": message, "logs": logs,
               "seconds": round(time.time() - started, 3)}
        if trace:
            out["traceback"] = trace
        _emit(out)
        return 1

    try:
        with open(argv[1], "r", encoding="utf-8") as fh:
            req = json.load(fh)
        plugin_dir = os.path.abspath(req["plugin_dir"])
        dataset_dir = os.path.abspath(req["dataset_dir"])
        # The tool's folder: its working directory, and first on sys.path, so
        # it may `import` a helper module it ships beside its entry file. Only
        # now, after this module's own imports: a helper named like one of
        # them (select.py) must not replace it, which is why launch puts the
        # folder neither on PYTHONPATH nor where the interpreter starts.
        os.chdir(plugin_dir)
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        run = _load_run(plugin_dir, req["entry"], req.get("module_name") or "pv_tool")

        ds = Dataset(dataset_dir, subset=req.get("subset"),
                     section_filter=req.get("section_filter"),
                     composition=req.get("atlas_composition"),
                     device=req.get("device") or "cpu",
                     log=log, progress=progress)
        contract = req.get("output_contract")

        def one(params):
            payload = check_result(run(ds, params), ds.n_cells)
            if contract:
                missing, extra = output_mismatch(contract, payload)
                if missing or extra:
                    raise PluginError("bad_result", _contract_sentence(missing, extra))
            return payload

        sweep = req.get("sweep")
        if sweep:
            payload = _run_sweep(one, dict(req.get("params") or {}), sweep, progress)
        else:
            payload = one(req.get("params") or {})
        try:
            line = json.dumps({"t": "result", "result": payload, "logs": logs,
                               "seconds": round(time.time() - started, 3)},
                              allow_nan=False)
        except ValueError:
            # check_result turns NaN into null wherever it reads numbers, but
            # a few fields pass through as given (a domain, a series rule).
            # A NaN there is not JSON, and would reach the browser as a token
            # its parser rejects.
            raise PluginError(
                "non_finite",
                "The result holds NaN or an infinity in a field that must be a "
                "number, such as a domain or a series rule. Use null there, or "
                "leave the field out.")
    except PluginError as exc:
        return fail(exc.code, exc.message)
    except r_bridge.RError as exc:
        # R's own error and call stack, which say where in tool.R it failed;
        # the Python frames of the bridge would only say that R did.
        return fail(exc.code, str(exc), exc.traceback or None)
    except BaseException as exc:                       # noqa: BLE001 — see below
        # Deliberately BaseException. A tool that calls sys.exit() mid-run
        # raises SystemExit, which would otherwise leave the parent reading a
        # stream that ends with no terminal line and reporting "produced no
        # result" for something with a perfectly good explanation.
        return fail("crash", f"{type(exc).__name__}: {exc}", traceback.format_exc()[-4000:])
    _send(line)
    return 0


def _contract_sentence(missing: list, extra: list) -> str:
    parts = []
    if missing:
        parts.append(f"plugin.json declares {', '.join(missing)} in outputs, but run() "
                     f"did not return {'it' if len(missing) == 1 else 'them'}")
    if extra:
        parts.append(f"run() returned {', '.join(extra)}, which plugin.json does not "
                     "declare in outputs or optional_outputs")
    return "; and ".join(parts) + "."


def _load_run(plugin_dir: str, entry: str, module_name: str):
    """The tool's `run(ds, params)`: the entry module's own, or R's through the bridge."""
    if entry.lower().endswith(".r"):
        return lambda ds, params: r_bridge.run(ds, params, plugin_dir, entry)
    module = _load_entry(plugin_dir, entry, module_name)
    run = getattr(module, "run", None)
    if not callable(run):
        raise PluginError("no_run", f"{entry} does not define a run(ds, params) function.")
    return run


def _run_sweep(one, params: dict, sweep: dict, progress) -> dict:
    """The tool at every value of the swept parameter, in this process.

    A PluginError at one value is recorded as that step's answer and the sweep
    goes on; a malformed return value (the tool breaking its own contract) or a
    crash stops everything, as it would a single run. At least one step must
    succeed, or the sweep as a whole reports the first refusal.
    """
    name, values = str(sweep["param"]), list(sweep["values"])
    steps, refused = [], None
    for i, value in enumerate(values):
        progress(i / max(1, len(values)), f"{name} = {value} ({i + 1} of {len(values)})")
        try:
            steps.append(one(dict(params, **{name: value})))
        except PluginError as exc:
            # A return value the contract rejects is the tool breaking its own
            # promise, at any value: that ends the sweep like a single run.
            if exc.code in RESULT_CODES:
                raise
            refused = refused or exc
            steps.append({"error": {"code": exc.code, "message": exc.message}})
    if refused is not None and all("error" in s for s in steps):
        raise refused
    return {"sweep": {"param": name, "values": values, "steps": steps}}


def _load_entry(plugin_dir: str, entry: str, module_name: str):
    """Import the tool's entry file under a private module name.

    A private name, not the file's own stem: two installed tools both called
    `tool.py` would otherwise collide in sys.modules. One process runs one
    tool, so the name only has to be unique against the standard library.
    """
    path = os.path.join(plugin_dir, entry)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginError("no_entry", f"Could not load {entry}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(main(sys.argv))
