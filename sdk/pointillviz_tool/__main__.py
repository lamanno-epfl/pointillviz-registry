"""The SDK's command line, for tool authors and for CI.

    python -m pointillviz_tool new <id> [--r] [-o DIR]
    python -m pointillviz_tool validate <dir|bundle> [--json] [--env own|app] [--device cpu|cuda]
    python -m pointillviz_tool pack <dir> [-o OUT]
    python -m pointillviz_tool run <dir> --dataset <dir> [--params JSON] [--env own|app] [--device cpu|cuda]

`--env own` (the default for a tool with a pixi.toml) builds the tool's own
pixi environment and runs it there; `--env app` runs it under this Python,
which is how CI validates a tool from inside its environment.

Exit status: 0 when the command did what was asked (for `validate`, every
required gate passed; for `run`, the tool returned a result), 1 when it did
not, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

from . import bundle, envs, launch, manifest, scaffold, validate as gates
from .dataset import PluginError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pointillviz_tool",
                                 description="Write, check, pack and run PointillViz tools.")
    sub = ap.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("new", help="write a new tool that already passes the gates")
    p.add_argument("id", help="the tool id: lowercase letters, digits and hyphens")
    p.add_argument("--r", action="store_true", help="a native R tool (tool.R) instead of Python")
    p.add_argument("-o", "--out", default=".", help="the folder to write it in (default: here)")

    p = sub.add_parser("validate", help="run every gate on a tool folder or a .pvtool")
    p.add_argument("path", help="a tool folder, or a .pvtool file")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    _where(p)

    p = sub.add_parser("pack", help="zip a tool folder into <id>-<version>.pvtool")
    p.add_argument("dir", help="the tool folder (the one holding plugin.json)")
    p.add_argument("-o", "--out", default=None,
                   help="the bundle's path (default: <id>-<version>.pvtool beside the folder)")

    p = sub.add_parser("run", help="run a tool on a converted dataset and print the reply")
    p.add_argument("dir", help="the tool folder")
    p.add_argument("--dataset", required=True, help="a converted dataset folder (with meta.json)")
    p.add_argument("--params", default=None, help='parameters as a JSON object, e.g. \'{"k": 10}\'')
    _where(p)

    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help(sys.stderr)
        return 2
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


def _where(p) -> None:
    p.add_argument("--env", choices=("own", "app"), default=None,
                   help="own: the tool's pixi environment (default with a pixi.toml); "
                        "app: this Python")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="the device the tool runs on (default: cpu)")


def _new(args) -> int:
    try:
        out = scaffold.scaffold(args.id, args.out, language="r" if args.r else "python")
    except (manifest.ManifestError, OSError) as exc:
        return _error(exc, 2)
    print(out)
    print(f"Next:\n  cd {out}\n  pixi lock\n  python -m pointillviz_tool validate .")
    return 0


def _validate(args) -> int:
    if not os.path.exists(args.path):
        return _error(f"{args.path}: no such folder or file", 2)
    report = _until_interrupted(lambda run_id: gates.validate(
        args.path, env=args.env, device=args.device, run_id=run_id, log=_progress))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(gates.format_report(report))
        if not report["ok"]:
            # The one line a CI log tail will show.
            sys.stderr.write(f"{os.path.basename(os.path.abspath(args.path))}: "
                             "validation failed\n")
    return 0 if report["ok"] else 1


def _pack(args) -> int:
    if not os.path.isdir(args.dir):
        return _error(f"{args.dir} is not a folder", 2)
    try:
        out = bundle.pack(args.dir, args.out)
    except (bundle.BundleError, manifest.ManifestError) as exc:
        return _error(exc, 1)
    print(f"{out}\n{os.path.getsize(out):,} bytes")
    return 0


def _run(args) -> int:
    if not os.path.isfile(os.path.join(args.dataset, "meta.json")):
        return _error(f"{args.dataset} is not a converted dataset (it has no meta.json)", 2)
    try:
        m = manifest.load(args.dir)
        supplied = json.loads(args.params) if args.params else {}
        if not isinstance(supplied, dict):
            raise manifest.ManifestError("--params must be a JSON object.")
        params = manifest.coerce_params(m, supplied)
    except (manifest.ManifestError, ValueError) as exc:
        return _error(exc, 2)
    env = args.env or ("own" if envs.has_pixi(args.dir) else "app")
    if env == "own" and not envs.has_pixi(args.dir):
        return _error(f"{args.dir} has no pixi.toml, so it has no environment of its own; "
                      "use --env app", 2)
    try:
        launcher = envs.launcher(args.dir, env, args.device)
        if env == "own":
            envs.install_dedicated(args.dir, launcher[launcher.index("-e") + 1], _progress)
    except PluginError as exc:
        return _error(exc.message, 1)
    reply = _until_interrupted(lambda run_id: launch.run(
        launcher, args.dir, m, args.dataset, params, device=args.device, run_id=run_id))
    print(json.dumps(reply, indent=2))
    return 0 if reply["status"] == "ok" else 1


_COMMANDS = {"new": _new, "validate": _validate, "pack": _pack, "run": _run}


def _until_interrupted(work):
    """`work(run_id)` on a thread, so that Ctrl-C cancels the tool it is running.

    A tool runs in a process group of its own, which a terminal's Ctrl-C does
    not reach; without this, interrupting the command would leave the tool
    running.
    """
    run_id = f"cli-{os.getpid()}"
    box = {}

    def target():
        try:
            box["value"] = work(run_id)
        except BaseException as exc:                    # noqa: BLE001 — re-raised below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(0.2)
    except KeyboardInterrupt:
        launch.cancel(run_id)
        raise
    if "error" in box:
        raise box["error"]
    return box["value"]


def _progress(line: str) -> None:
    """pixi's progress while an environment is built: stderr, so --json stays clean."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _error(what, status: int) -> int:
    sys.stderr.write(f"error: {what}\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
