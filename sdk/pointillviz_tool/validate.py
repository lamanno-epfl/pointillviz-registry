"""The gates a community tool must pass before PointillViz will run it.

    report = validate("tools/my-tool")                 # a directory, or a .pvtool
    print(format_report(report))

The command line is `python -m pointillviz_tool validate <dir|bundle>`.

`validate` returns one report: one entry per gate, plus the disclosure surface
(what the tool imports and what it touches) that the install dialog shows a
human before they say yes. The gates, in order: bundle, manifest,
environment, entry, imports (advisory), fixture, result, edge, no_mutation.

Why a report and not an exception
---------------------------------
The first user of this is the tool AUTHOR, iterating. Stopping at the first
failure would mean nine edit-run cycles to find nine problems, so every gate
that CAN run does run and every one of them reports. The exceptions are the
gates with a hard prerequisite: nothing can be checked without a manifest, and
`result` has nothing to check when the tool never returned. Those `skip`,
which is not `pass`, so `ok` is still False. One case skips a gate that is
then not required: a tool without a pixi.toml runs in PointillViz's own
environment, so `environment` has nothing to check. A tool that declares a
GPU, even `"gpu": "required"`, is still run on the validation's device: the
CPU by default, and always in CI. A GPU method checks for CUDA and falls back
to the CPU, and that fallback is the path CI can test.

Why a fixture and not a real dataset
------------------------------------
`pointillviz_tool.fixture` builds a real converted dataset (same `cells.bin`
layout, same gene-major `gene_counts.bin`) that is deliberately awkward: 8
cells with NaN coordinates, a cell type with zero members, a gene expressed
nowhere, two sections. A tool validated against a clean rectangle passes here
and fails on the first slide someone opens.

Where the tool runs
-------------------
In the environment it will run in. With `env="own"` (the default for a tool
with a pixi.toml) the tool's own pixi environment is built first, exactly as
its pixi.lock pins it; with `env="app"` the tool runs under the interpreter
running this code, which is how CI validates a tool from inside its
environment and how PointillViz validates a tool installed into its own.
Every run goes through `launch.run`, the driver PointillViz itself uses: a
gate that launched tools its own way would be certifying a program nobody
runs. Gate runs use the CPU unless `device` says otherwise.

Even the import is a child process. Importing a community tool executes its
module body, and doing that here would run it with the validator's
privileges, where `sys.exit()` at import time takes the report with it. The
probe goes through `runner._load_entry` (or, for R, through runner.R in probe
mode), so the import that is validated is the import a run performs.

There are no time or memory limits, here or in a run: how long a tool takes
and how much it uses is its author's responsibility. Nor is there a
determinism gate: reproducibility is the author's too.

Why `imports` never fails
-------------------------
This is not a sandbox: a tool is arbitrary code run with the privileges of
whoever started PointillViz. A gate that FAILED a tool for importing `socket`
would advertise a protection that does not exist; the same tool could reach
the network through a dependency, or write a file through `numpy.save`, and
pass. So the gate discloses what, and at which file and line, in a sentence
the install dialog can show a person who then decides. Over-reporting is the
safe direction and under-reporting is not, which is why a computed `open()`
mode counts as a write.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import zipfile

from . import bundle, envs, launch
from .dataset import DEVICES, PluginError
from .manifest import ManifestError, coerce_params, language, load as load_manifest, sweep_values
from .results import OUTPUT_KINDS, RESULT_CODES, output_mismatch

#: Gate order and required-ness. `environment` is required only for a tool
#: with a pixi.toml (decided per tool in `_validate_dir`).
GATE_ORDER = (
    ("bundle", True),
    ("manifest", True),
    ("environment", True),
    ("entry", True),
    ("imports", False),
    ("fixture", True),
    ("result", True),
    ("edge", True),
    ("no_mutation", True),
)

#: Reported in this order, so two reports on the same tool read the same way.
RISK_KINDS = ("network", "subprocess", "filesystem_write", "dynamic_exec", "env")

#: What every tool environment must hold: the SDK runs inside it.
_BASE_PACKAGES = ("python", "numpy", "scipy")
_R_PACKAGES = ("r-base", "r-jsonlite")


class _Skip(Exception):
    """A gate declining to run because its input never arrived."""


# ===========================================================================
# Public entry point
# ===========================================================================
def validate(path: str, *, env: str = None, device: str = "cpu", extra_paths=(),
             run_id: str = None, log=None) -> dict:
    """Run every gate against a tool directory or a `.pvtool` bundle; return the report.

    `env` is "own" (build and run in the tool's own pixi environment; the
    default when it has a pixi.toml) or "app" (run under this interpreter;
    the default without one). `device` is the device the runs use ("cpu"
    unless an author is checking a GPU path, whatever the tool's `gpu`
    says). `extra_paths` go on the runs' PYTHONPATH after the SDK: the app
    root, for a legacy tool that imports `backend.*`. `run_id` names every run this validation starts, so
    `launch.cancel(run_id)` stops it; a cancelled run ends the validation.
    `log` receives pixi's progress while an environment is built.
    """
    if env not in (None, "own", "app"):
        raise ValueError(f"env must be 'own' or 'app', not {env!r}.")
    if device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda', not {device!r}.")
    started = time.perf_counter()
    options = {"env": env, "device": device, "extra_paths": tuple(extra_paths),
               "run_id": run_id, "log": log}
    if os.path.isdir(path):
        return _validate_dir(os.path.abspath(path), None, options, started)
    with open(path, "rb") as fh:
        data = fh.read()
    tmp = tempfile.mkdtemp(prefix="pv_validate_bundle_")
    try:
        dest = os.path.join(tmp, "tool")
        try:
            bundle.extract(data, dest)
        except bundle.BundleError as exc:
            # Nothing was written that can be validated, so the remaining gates
            # have no subject. They are still LISTED, because a report with a
            # missing gate reads as a gate that was quietly not run.
            return _refused_report(str(exc), len(data), started)
        # The environment is built inside the extracted copy (a pixi
        # environment cannot be moved), and goes with it.
        return _validate_dir(dest, data, options, started)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _validate_dir(plugin_dir: str, bundle_bytes, options: dict, started: float) -> dict:
    ctx = _Context(plugin_dir, bundle_bytes, **options)
    gates = []
    try:
        for gid, required in GATE_ORDER:
            if gid == "environment":
                required = ctx.has_pixi
            if ctx.cancelled:
                gate = _skipped(gid, required, "skipped: the validation was cancelled.")
            else:
                gate = _run_gate(ctx, gid, required)
            ctx.status[gid] = gate["status"]
            gates.append(gate)
    finally:
        ctx.cleanup()
    return _assemble(ctx, gates, started)


# ===========================================================================
# Driver
# ===========================================================================
class _Context:
    """What the gates share: the tool, its manifest, its launcher, the fixture, the runs."""

    def __init__(self, plugin_dir: str, bundle_bytes=None, *, env=None, device="cpu",
                 extra_paths=(), run_id=None, log=None):
        self.plugin_dir = plugin_dir
        self.bundle_bytes = bundle_bytes
        self.has_pixi = envs.has_pixi(plugin_dir)
        self.env = env or ("own" if self.has_pixi else "app")
        self.device = device
        self.extra_paths = extra_paths
        self.run_id = run_id
        self.log = log
        self.manifest = None
        self.status: dict = {}
        self.params: dict = {}
        self.first = None                    # the fixture run's outcome
        self.ran_any = False
        self.cancelled = False
        self.built = None                    # (environment, seconds) once built
        self.outputs_seen: list = []
        self.imports: list = []
        self.risks: list = []
        self.fixture_info = None
        self._launcher = None
        self._tmp = None
        self._dataset_dir = None
        self._pre_hashes = None

    # -- prerequisites ---------------------------------------------------
    def require_manifest(self) -> dict:
        # Not merely "did it parse": the manifest GATE has to have passed. A
        # manifest that loads but declares a select default that is not one of
        # its own options is a manifest nothing later can be judged against,
        # and spending a fixture build and three subprocesses on it would bury
        # the one-line fix under five more failures.
        if self.manifest is None or self.status.get("manifest", "pass") != "pass":
            raise _Skip("skipped: the manifest gate did not pass, and every later "
                        "gate is defined in terms of the manifest.")
        return self.manifest

    def require(self, gate_id: str, why: str) -> None:
        if self.status.get(gate_id) != "pass":
            raise _Skip(f"skipped: {why}")

    # -- where the tool runs -----------------------------------------------
    def launcher(self) -> list:
        """The argv prefix that starts Python for this tool's runs.

        In its own environment, that environment is built first: the one the
        launcher names (`gpu` for a GPU run of a tool that declares it,
        `default` otherwise), so the first run does not spend its time
        downloading packages. Raises PluginError when it cannot be built.
        """
        if self._launcher is None:
            if self.env == "own" and not self.has_pixi:
                raise PluginError("env_failed", "the tool has no pixi.toml, so it has no "
                                                "environment of its own.")
            launcher = envs.launcher(self.plugin_dir, self.env, self.device)
            if self.env == "own":
                environment = launcher[launcher.index("-e") + 1]
                began = time.perf_counter()
                envs.install_dedicated(self.plugin_dir, environment, self.log)
                self.built = (environment, time.perf_counter() - began)
            self._launcher = launcher
        return self._launcher

    # -- the fixture -----------------------------------------------------
    def dataset(self) -> str:
        """Build the fixture once, and snapshot it before anything runs."""
        if self._dataset_dir is None:
            from .fixture import write_fixture
            self._tmp = self._tmp or tempfile.mkdtemp(prefix="pv_validate_")
            self.fixture_info = write_fixture(os.path.join(self._tmp, "fixture"))
            self._dataset_dir = self.fixture_info["path"]
            self._pre_hashes = _hash_tree(self._dataset_dir)
        return self._dataset_dir

    @property
    def pre_hashes(self):
        return self._pre_hashes

    def run(self, params: dict, sweep: dict = None) -> dict:
        """One gate run, through the driver PointillViz itself uses."""
        self.ran_any = True
        reply = launch.run(self.launcher(), self.plugin_dir, self.manifest, self.dataset(),
                           params, sweep=sweep, device=self.device, run_id=self.run_id,
                           extra_paths=self.extra_paths)
        if reply.get("code") == "cancelled":
            self.cancelled = True
        return reply

    def cleanup(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


def _run_gate(ctx: _Context, gate_id: str, required: bool) -> dict:
    started = time.perf_counter()
    try:
        status, message, detail = _GATES[gate_id](ctx)
    except _Skip as exc:
        status, message, detail = "skip", str(exc), None
    except Exception as exc:                                    # noqa: BLE001
        # A gate that throws is a bug in THIS module, not in the tool. Say so
        # in those words: a tool author who reads "fixture: failed" and starts
        # rewriting their tool has been sent the wrong way.
        status = "fail"
        message = (f"the {gate_id} gate could not run: {type(exc).__name__}: {exc} "
                   "— this is a validator bug, not a fault in the tool.")
        detail = {"notes": traceback.format_exc().splitlines()[-8:]}
    return {"id": gate_id, "required": required, "status": status,
            "message": message, "detail": detail,
            "seconds": round(time.perf_counter() - started, 3)}


def _assemble(ctx: _Context, gates: list, started: float) -> dict:
    m = ctx.manifest or {}
    return {
        "ok": all(g["status"] == "pass" for g in gates if g["required"]),
        "plugin_id": m.get("id"),
        "version": m.get("version"),
        "name": m.get("name"),
        "language": language(m) if m else None,
        "gpu": m.get("gpu"),
        "gates": gates,
        "risks": list(ctx.risks),
        "imports": list(ctx.imports),
        "outputs_seen": list(ctx.outputs_seen),
        "seconds": round(time.perf_counter() - started, 3),
    }


def _skipped(gate_id: str, required: bool, message: str) -> dict:
    return {"id": gate_id, "required": required, "status": "skip",
            "message": message, "detail": None, "seconds": 0.0}


def _refused_report(message: str, n_bytes: int, started: float) -> dict:
    gates = []
    for gid, required in GATE_ORDER:
        if gid == "bundle":
            gates.append({"id": gid, "required": required, "status": "fail",
                          "message": message, "detail": {"bytes": n_bytes},
                          "seconds": 0.0})
        else:
            gates.append(_skipped(gid, required, "skipped: the bundle could not be opened."))
    return {"ok": False, "plugin_id": None, "version": None, "name": None,
            "language": None, "gpu": None, "gates": gates, "risks": [], "imports": [],
            "outputs_seen": [], "seconds": round(time.perf_counter() - started, 3)}


# ===========================================================================
# Gate: bundle
# ===========================================================================
def _gate_bundle(ctx: _Context):
    data = ctx.bundle_bytes
    origin = "the bundle"
    if data is None:
        data = _repack(ctx.plugin_dir)
        origin = "a zip of the directory"
    try:
        info = bundle.inspect(data)
    except bundle.BundleError as exc:
        return "fail", str(exc), {"bytes": len(data), "source": origin}
    names = info["names"]
    return "pass", (
        f"{len(names)} file{'' if len(names) == 1 else 's'}, {_size(len(data))} packed "
        f"and {_size(info['total_uncompressed'])} unpacked, plugin.json at the root."
    ), {"source": origin, "bytes": len(data),
        "uncompressed": info["total_uncompressed"],
        "notes": [f"  {n}" for n in sorted(names)[:12]]}


def _repack(plugin_dir: str) -> bytes:
    """Zip a tool directory in memory, as `bundle.pack` would.

    `pack()` is not called because it loads the manifest first (it needs the
    id and version to NAME the file), and this gate deliberately runs BEFORE
    the manifest gate: an archive carrying `../../evil.py` should be reported
    as that, not as "plugin.json is not valid JSON". The files are
    `bundle.files`, the one definition of what is part of a tool, so
    validating a directory and validating its `.pvtool` cannot disagree, and
    an environment built in the folder (`.pixi/`) is never part of it.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, full in bundle.files(plugin_dir):
            zf.write(full, rel)
    return buf.getvalue()


# ===========================================================================
# Gate: manifest
# ===========================================================================
def _gate_manifest(ctx: _Context):
    try:
        ctx.manifest = load_manifest(ctx.plugin_dir)
    except ManifestError as exc:
        return "fail", str(exc), None
    m = ctx.manifest
    problems = _default_problems(m)
    if problems:
        return "fail", ("the declared defaults are not self-consistent: "
                        + "; ".join(problems) + "."), {"problems": problems}
    params = ", ".join(f"{p['name']} ({p['type']})" for p in m["params"]) or "none"
    gpu = f"; GPU {m['gpu']}" if m["gpu"] else ""
    head = (f"{m['id']} {m['version']} by {m['author']}; {_LANGUAGE[language(m)]} entry "
            f"{m['entry']}, outputs {', '.join(m['outputs'])}{gpu}.")
    detail = {"entry": m["entry"], "language": language(m), "gpu": m["gpu"],
              "outputs": m["outputs"], "requires": m["requires"],
              "notes": [f"  parameters: {params}"]}
    drift = _defaults_drift(m, ctx.plugin_dir) if language(m) == "python" else []
    if drift:
        # A note on a PASS, not a failure: what the host passes is the
        # manifest's value either way, so this costs the author's own direct
        # calls rather than the user's run.
        return "pass", (
            f"{head} Note: {'; '.join(drift)} — the host passes the manifest's value, so "
            "calling run() directly and running through the viewer would compute "
            "different numbers."
        ), dict(detail, defaults_drift=drift)
    return "pass", head, detail


_LANGUAGE = {"python": "Python", "r": "R"}


def _default_problems(manifest: dict) -> list:
    """Defects in the DECLARED defaults — no dataset, no subprocess, no run.

    `manifest.validate` checks that a select HAS options and that a number HAS
    bounds, but not that the default the same manifest declares is one of
    those options or inside those bounds; `coerce_params` is where that is
    known, and it runs later. So the manifest gate dry-runs the real coercion
    over the declared defaults and reports what it says.

    Here rather than in the fixture gate: a select default that is not among
    its own options is knowable with no dataset at all, and reporting it
    against `fixture` sends an author reading their entry file to look for a
    bug that is four lines up in their JSON.

    Only parameters that actually DECLARE a default are probed. A `gene` or
    `categorical_field` parameter is meant to have none — the viewer fills it
    from the open dataset — and "Gene is required" is coerce_params being
    right, not a defect to gate on.
    """
    declared = [p for p in manifest["params"] if p.get("default") not in (None, "")]
    problems = []
    for p in declared:
        name, want = p["name"], p["default"]
        # One parameter at a time: coerce_params raises on the FIRST bad value
        # and returns nothing for the rest, and a report that names one of two
        # broken defaults costs the author a second round trip to find the
        # other one.
        try:
            got = coerce_params(dict(manifest, params=[p]), {})[name]
        except ManifestError as exc:
            problems.append(str(exc).rstrip("."))
            continue
        except (ValueError, TypeError) as exc:
            problems.append(f"{name}: the default {want!r} cannot be coerced to "
                            f"{p['type']} ({exc})")
            continue
        # Clamped, not refused: coerce_params treats an out-of-range number as
        # a stale client and clamps, which is the right recovery at request
        # time and the wrong silence in a manifest -- the form would open at a
        # value the manifest does not name.
        if p["type"] in ("int", "float") and float(got) != float(want):
            problems.append(
                f"{name}: the default {want!r} is outside its own declared range "
                f"{p['min']:g}..{p['max']:g}, so the control would open at {got!r}")
        if p["type"] == "genes" and isinstance(want, list) and len(got) < len(want):
            problems.append(
                f"{name}: the default names {len(want)} genes but max_items is "
                f"{p['max_items']}, so {len(want) - len(got)} of them would be dropped")
    return problems


def _dict_literal(node):
    """The dict a module-level assignment builds, or None — two spellings only.

    `{"k": 15}` and `dict(k=15)` are the same table written two ways, and an
    author refactoring one into the other has no reason to think twice, so
    the second cannot be a blind spot. Deliberately just these two:
    `{**BASE, "k": 15}`, a dict assembled in a loop, a factory call — each is
    another idiom and the list has no end, so a detector that chased them
    would grow false positives faster than coverage. The rest fall to
    disclosure-by-absence, which is honest: this reports what it can see.
    """
    try:
        value = ast.literal_eval(node)
    except Exception:                        # noqa: BLE001 — not a literal table
        value = None
    if isinstance(value, dict):
        return value
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "dict" and not node.args):
        return None
    # No positional args above, and no ** unpacking here: under either, the
    # "every key is a declared parameter" guard cannot be evaluated, and a
    # guard that cannot be evaluated is not a guard.
    if not all(kw.arg for kw in node.keywords):
        return None
    out = {}
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except Exception:                    # noqa: BLE001 — one non-literal
            return None                      # value skips the table wholesale,
    return out                               # exactly as the brace spelling does


def _defaults_drift(manifest: dict, plugin_dir: str) -> list:
    """Module-level dicts in the entry file that contradict the manifest.

    A tool that keeps its own defaults dict — so `run()` stays callable
    straight from a notebook — has two sources for one fact, and nothing else
    in the system ever compares them. Measured on a reference tool: a manifest
    `k` of 15 against a module `k` of 20 was invisible to every required gate
    and to twenty-one of that tool's own tests, because each of them reads the
    default from one side or the other and never from both. The user always
    gets the manifest's value, so the two halves compute different numbers
    from a request that looks identical.

    The dict is found by its KEYS being the manifest's own parameter names,
    never by being called DEFAULTS, which would be a naming convention no
    contract requires. Every key must be a declared parameter and there must
    be at least two, so a lone `{"k": 5}` meaning something unrelated cannot
    trip it. Parameters the manifest deliberately leaves default-free are
    skipped: the viewer fills them from the open dataset, so a module dict
    naming a gene for direct calls is not drift.

    Reported, never failed. The manifest is authoritative for what the host
    passes, so drift costs the author's own direct calls rather than the
    user's run.
    """
    declared = {p["name"]: p.get("default") for p in manifest["params"]}
    if len(declared) < 2:
        return []
    path = os.path.join(plugin_dir, manifest["entry"])
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return []                    # the entry gate owns unreadable entry files
    found = []
    for node in tree.body:           # module level only; a dict built inside a
        if not isinstance(node, ast.Assign):   # function is not a second source
            continue
        value = _dict_literal(node.value)
        if value is None or len(value) < 2:
            continue
        if not all(k in declared for k in value):
            continue
        name = (node.targets[0].id if len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) else "a module-level dict")
        for key in sorted(value):
            want = declared[key]
            if want is None or want == "":
                continue             # the viewer fills these; see the docstring
            if value[key] != want:
                found.append(f"{name}[{key!r}] is {value[key]!r} but the manifest "
                             f"declares {want!r}")
    return found


# ===========================================================================
# Gate: environment
# ===========================================================================
def _gate_environment(ctx: _Context):
    """The tool's pixi environment is in order. Checked, not installed.

    pixi itself reads pixi.toml (`pixi info --json`), so this holds on any
    Python the validator runs under, and agrees with what pixi will build.
    """
    m = ctx.require_manifest()
    if not ctx.has_pixi:
        if ctx.env == "own":
            return "fail", (
                "the tool has no pixi.toml, so it has no environment of its own to run "
                "in. Add one that declares python, numpy and scipy, then run `pixi lock`; "
                "or validate it in the current environment with --env app."), None
        # Allowed, for a tool that stays on one machine; the registry is
        # where a schema-3 tool must bring its environment, and an author
        # should learn that here rather than from a failed CI step.
        registry = (" The registry accepts a tool only with its pixi.toml and pixi.lock."
                    if m["schema"] >= 3 else "")
        raise _Skip("skipped: the tool has no pixi.toml, so it runs in PointillViz's own "
                    "environment." + registry)
    try:
        envs.pixi_exe()
    except PluginError as exc:
        return "fail", exc.message, None

    manifest_path = os.path.join(ctx.plugin_dir, "pixi.toml")
    info = envs._info(ctx.plugin_dir)
    environments = info.get("environments_info") or []
    default = next((e for e in environments if e.get("name") == "default"), None)
    if default is None:
        # `pixi info` says nothing about a manifest it cannot read; the lock
        # check runs the parser, and its error says where the file is wrong.
        _s, _o, err = envs._pixi(["lock", "--check", "--manifest-path", manifest_path],
                                 check=False)
        return "fail", "pixi cannot read pixi.toml.", {
            "notes": [f"  {line}" for line in envs._error_text(err).splitlines()[:12]]}

    problems, notes = [], []
    ok, message = envs.lock_check(ctx.plugin_dir)
    if not ok:
        first, _, rest = message.partition("\n")
        problems.append(first.rstrip("."))
        notes += [f"  {line}" for line in rest.splitlines()[:12]]
    needed = list(_BASE_PACKAGES) + (list(_R_PACKAGES) if language(m) == "r" else [])
    declared = {str(d).lower() for d in (default.get("dependencies") or [])}
    declared |= {str(d).lower() for d in (default.get("pypi_dependencies") or [])}
    missing = [name for name in needed if name not in declared]
    if missing:
        problems.append(
            f"pixi.toml does not declare {', '.join(missing)} under [dependencies]; "
            f"{'a native R tool needs' if language(m) == 'r' else 'every tool needs'} "
            f"{', '.join(needed)}, because the SDK runs inside the tool's environment")
    platforms = sorted({p.get("subdir") or p.get("name") for p in default.get("platforms") or []})
    here = info.get("platform")
    if here not in platforms:
        problems.append(f"pixi.toml's platforms ({', '.join(platforms) or 'none'}) do not "
                        f"include this machine's ({here}), so the environment cannot be "
                        "built here")
    names = [e.get("name") for e in environments]
    detail = {"environments": names, "platforms": platforms, "declared": sorted(declared),
              "notes": notes}
    if problems:
        return "fail", "; ".join(problems) + ".", detail
    return "pass", (
        f"pixi.lock matches pixi.toml, which declares {', '.join(needed)} for "
        f"{', '.join(platforms)}; environments: {', '.join(names)}."
    ), detail


# ===========================================================================
# Gate: entry
# ===========================================================================
def _gate_entry(ctx: _Context):
    m = ctx.require_manifest()
    entry = m["entry"]
    if ctx.env == "own":
        ctx.require("environment", "the tool's environment is not in order, so it was "
                                   "not built.")
    try:
        launcher = ctx.launcher()
    except PluginError as exc:
        return "fail", f"the tool's environment could not be built: {exc.message}", None
    built = ""
    if ctx.built:
        built = f"Built the {ctx.built[0]} environment in {ctx.built[1]:.1f} s; "
    probe = _probe_entry(ctx, launcher, entry)

    if probe.get("phase") == "cancelled":
        ctx.cancelled = True
        return "fail", f"{built}loading {entry} was cancelled before it finished.", None
    if probe.get("phase") == "probe":
        return "fail", (f"{built}the import probe for {entry} died without answering "
                        f"(exit status {probe.get('returncode')})."), \
               {"notes": (probe.get("stderr") or "").splitlines()[-10:]}
    if probe.get("phase") == "rscript":
        return "fail", f"{built}{entry} cannot be loaded: {probe['error']}", None
    if not probe.get("ok"):
        where = ("declare what it needs in pixi.toml and run `pixi lock`" if ctx.has_pixi
                 else "this tool has no pixi.toml, so that is PointillViz's own environment")
        verb = "sourcing" if language(m) == "r" else "importing"
        return "fail", (
            f"{built}{verb} {entry} raised {probe.get('error')}. Everything it loads must "
            f"resolve in the environment the tool runs in: {where}."
        ), {"notes": (probe.get("traceback") or "").splitlines()[-12:]}

    loaded = f"{'sourced' if language(m) == 'r' else 'imported'} in {probe.get('seconds', 0):.2f} s"
    if not probe.get("has_run"):
        return "fail", (f"{built}{entry} loads cleanly but defines no run(ds, params) "
                        "function; that is the one symbol a tool must define."), None
    if not probe.get("callable"):
        return "fail", (f"{built}{entry} defines `run` but it is a {probe.get('run_type')}, "
                        "not a function."), None
    if language(m) == "r":
        formals = probe.get("formals") or []
        signature = f"function({', '.join(formals)})"
        if len(formals) < 2 and not probe.get("dots"):
            return "fail", (
                f"{built}run must be callable as run(ds, params); the function found is "
                f"{signature}, which takes {len(formals)} argument"
                f"{'' if len(formals) == 1 else 's'}."), probe
        return "pass", f"{built}{entry} {loaded} and defines run <- {signature}.", probe
    if probe.get("signature") is None:
        # A C callable has no readable signature. It cannot be faulted for a
        # signature nobody can see, and the fixture gate will call it anyway.
        return "pass", (f"{built}{entry} {loaded} and defines a callable `run` whose "
                        "signature could not be read."), probe
    if not probe.get("accepts_two"):
        return "fail", (
            f"{built}run must be callable as run(ds, params); the signature found is "
            f"run{probe['signature']}, which takes {probe['n_positional']} positional "
            f"parameter{'' if probe['n_positional'] == 1 else 's'} "
            f"({probe['n_required']} required)."
        ), probe
    return "pass", f"{built}{entry} {loaded} and defines run{probe['signature']}.", probe


def _probe_entry(ctx: _Context, launcher: list, entry: str) -> dict:
    """Load the entry file in a CHILD, in the tool's environment, and say what `run` is.

    Never in this process: a tool's module body is arbitrary code, and an
    `import` that exits or crashes would take the validator with it. The
    child re-enters this module with --probe-entry, so the probe is ordinary
    readable code. It is started as a run is (`launch.start`), with a run's
    environment and under the validation's `run_id`: there is no clock, so an
    import that blocks would otherwise hold the validation, and an install
    waiting on it, for as long as it blocks, with nothing able to stop it.
    """
    try:
        child = launch.start(
            [*launcher, "-m", "pointillviz_tool.validate", "--probe-entry",
             ctx.plugin_dir, entry],
            launch.child_env(ctx.device, ctx.extra_paths), ctx.run_id)
    except OSError as exc:
        return {"ok": False, "phase": "probe", "returncode": None,
                "stderr": f"Could not start {launcher[0]}: {exc}"}
    proc = child.proc
    try:
        # stderr on a thread: a chatty import must not fill its pipe and
        # block. Its end is not waited for, because something the tool started
        # at import time may hold it open; stdout is the probe's private
        # channel, which no such process holds.
        err: list = []
        reader = threading.Thread(target=lambda: err.extend(proc.stderr), daemon=True)
        reader.start()
        out = proc.stdout.read()
        proc.wait()
        reader.join(2.0)
    finally:
        child.release()
    if child.cancelled:
        return {"ok": False, "phase": "cancelled"}
    stderr = b"".join(list(err)).decode("utf-8", "replace")
    for line in reversed(out.decode("utf-8", "replace").splitlines()):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("probe"):
            # An R probe that died answers through the Python one; what R
            # printed on its way out is in this stderr.
            obj.setdefault("stderr", stderr)
            return obj
    return {"ok": False, "phase": "probe", "returncode": proc.returncode, "stderr": stderr}


def _probe_main(plugin_dir: str, entry: str) -> int:
    """The child side of `_probe_entry`: one JSON line on the private channel."""
    from . import r_bridge, runner
    channel = runner._protocol_channel()

    def answer(obj) -> int:
        channel.write(json.dumps(obj) + "\n")
        channel.flush()
        return 0

    plugin_dir = os.path.abspath(plugin_dir)
    # Match the runner: the tool's folder is where it runs, and first on
    # sys.path once this module's own imports are done, so a tool may import
    # its own modules at module scope.
    os.chdir(plugin_dir)
    if entry.lower().endswith(".r"):
        return answer(r_bridge.probe(plugin_dir, entry))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    out = {"probe": True, "ok": False}
    began = time.perf_counter()
    try:
        module = runner._load_entry(plugin_dir, entry, "pv_tool_probe")
    except BaseException as exc:                                # noqa: BLE001
        # BaseException for the same reason the runner uses it: a tool whose
        # module body calls sys.exit() must be reported, not silently missing.
        out.update(phase="import", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-4000:])
        return answer(out)
    run = getattr(module, "run", None)
    out.update(ok=True, seconds=round(time.perf_counter() - began, 3),
               has_run=run is not None, callable=callable(run), run_type=type(run).__name__)
    if callable(run):
        try:
            sig = inspect.signature(run)
        except (TypeError, ValueError):
            out["signature"] = None
        else:
            kinds = inspect.Parameter
            pos = [p for p in sig.parameters.values()
                   if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)]
            star = any(p.kind == kinds.VAR_POSITIONAL for p in sig.parameters.values())
            required = [p for p in pos if p.default is kinds.empty]
            out["signature"] = str(sig)
            out["n_positional"] = len(pos)
            out["n_required"] = len(required)
            # Callable AS run(ds, params) is the contract — not "declares
            # exactly two names". run(ds, params, k=3) honours it; so does
            # run(*args). run(ds) does not, and neither does run(ds, p, extra).
            out["accepts_two"] = len(required) <= 2 and (star or len(pos) >= 2)
    return answer(out)


# ===========================================================================
# Gate: imports  (a disclosure surface, never a rejection)
# ===========================================================================
#: Importing any of these is how a tool reaches the network. Matched on the
#: TOP-LEVEL module, so `from urllib.request import urlopen` is caught too.
_NETWORK_MODULES = frozenset({
    "socket", "urllib", "urllib3", "http", "httplib", "requests", "httpx",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc",
    "aiohttp", "websockets", "paramiko",
})
_SUBPROCESS_MODULES = frozenset({"subprocess", "multiprocessing", "pty", "popen2"})
_DYNAMIC_MODULES = frozenset({"importlib", "imp", "runpy"})
_DYNAMIC_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
#: Fully-qualified calls, after alias resolution (`np.save` -> `numpy.save`).
_WRITE_CALLS = frozenset({
    "os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir",
    "os.removedirs", "os.mkdir", "os.makedirs", "os.truncate", "os.link",
    "os.symlink", "os.chmod", "os.utime",
    "shutil.rmtree", "shutil.move", "shutil.copy", "shutil.copy2",
    "shutil.copyfile", "shutil.copytree", "shutil.make_archive",
    "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt",
})
#: Method names that write wherever they are called from — a pathlib.Path, a
#: DataFrame, an ndarray. Matched on the last component alone because the
#: receiver's type is not knowable from the AST, and for a DISCLOSURE surface a
#: false positive costs a sentence in a dialog while a false negative costs the
#: whole point of the gate.
_WRITE_METHODS = frozenset({"write_text", "write_bytes", "tofile", "to_csv",
                            "to_parquet", "to_hdf", "unlink", "touch"})
_ENV_METHODS = frozenset({"update", "setdefault", "pop", "clear", "popitem"})

#: The same five risks in R, found by pattern rather than by parsing: calls
#: (a name not preceded by a letter, digit, `.` or `_`, then an opening
#: parenthesis) and packages. Comments are stripped first.
_R_CALLS = (
    ("network", r"download\.file|url|socketConnection|socketAccept|make\.socket"),
    ("subprocess", r"system2?|pipe|shell"),
    ("filesystem_write", r"writeLines|writeBin|writeChar|saveRDS|save|save\.image|"
                         r"write\.csv2?|write\.table|write|sink|file\.create|file\.remove|"
                         r"file\.rename|file\.copy|unlink|dir\.create"),
    ("dynamic_exec", r"eval\s*\(\s*parse"),
    ("env", r"Sys\.getenv|Sys\.setenv|Sys\.unsetenv"),
)
_R_CALL_RES = [(risk, re.compile(rf"(?<![\w.])({names})\s*\(")) for risk, names in _R_CALLS]
#: file(..., "w") and the other writing modes.
_R_FILE_WRITE = re.compile(r"(?<![\w.])file\s*\([^)]*[\"'][wa]b?\+?[\"']")
_R_LIBRARY = re.compile(r"(?<![\w.])(?:library|require|requireNamespace|loadNamespace)"
                        r"\s*\(\s*[\"']?([A-Za-z][A-Za-z0-9.]*)")
_R_NAMESPACE = re.compile(r"(?<![\w.])([A-Za-z][A-Za-z0-9.]*):::?(?=[A-Za-z.`])")
_R_NETWORK_PACKAGES = frozenset({"curl", "httr", "httr2", "RCurl", "websocket"})
_R_SUBPROCESS_PACKAGES = frozenset({"processx", "callr", "parallel"})


def _gate_imports(ctx: _Context):
    m = ctx.require_manifest()
    files = [(rel, full) for rel, full in bundle.files(ctx.plugin_dir)
             if rel.endswith(".py") or rel.lower().endswith(".r")]
    files.sort(key=lambda rf: (rf[0] != m["entry"], rf[0]))
    modules, findings, unparsed = set(), [], []
    for rel, full in files:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if rel.lower().endswith(".r"):
            found_modules, found = _scan_r(rel, text)
            modules |= found_modules
            findings += found
            continue
        try:
            tree = ast.parse(text, rel)
        except SyntaxError as exc:
            # Not this gate's verdict to give: the entry gate already failed on
            # a file that will not compile, and a helper that will not compile
            # fails at import. Say what could not be read and move on.
            unparsed.append(f"{rel}:{exc.lineno}: {exc.msg}")
            continue
        scan = _Scan(rel)
        scan.visit(tree)
        modules |= scan.modules
        findings += scan.findings

    # The union over every source file in the bundle, not just the entry
    # file: a helper's imports are equally the tool's imports, and a reader
    # deciding whether to install is asking about the bundle.
    ctx.imports = sorted(modules)
    by_risk = {r: [f for f in findings if f[0] == r] for r in RISK_KINDS}
    ctx.risks = [r for r in RISK_KINDS if by_risk[r]]
    notes = [f"  {where}  {what}" for _r, where, what in findings]
    notes += [f"  unparsed: {u}" for u in unparsed]
    detail = {"modules": ctx.imports, "files": [rel for rel, _f in files],
              "findings": {r: [f"{w}  {t}" for _r, w, t in by_risk[r]] for r in ctx.risks},
              "unparsed": unparsed, "notes": notes}

    if not ctx.risks:
        return "pass", (
            f"{len(files)} source file{'' if len(files) == 1 else 's'}, using "
            f"{', '.join(ctx.imports) or 'nothing'}; no network, subprocess, "
            "filesystem-write, dynamic-exec or environment use found."
        ), detail

    parts = []
    for risk in ctx.risks:
        items = by_risk[risk]
        shown = "; ".join(f"{what} at {where}" for _r, where, what in items[:3])
        if len(items) > 3:
            shown += f" (+{len(items) - 3} more)"
        parts.append(f"{risk} — {shown}")
    return "warn", (
        "this tool uses " + ", and ".join(parts) + ". None of it is blocked: a tool is "
        "arbitrary code run with the privileges of whoever started PointillViz, so this is "
        "disclosure for a human to weigh, not a sandbox."
    ), detail


def _scan_r(rel: str, text: str):
    """(packages, findings) for one R file: a pattern scan, comments removed."""
    modules, findings = set(), []
    for lineno, line in enumerate(_strip_r_comments(text).splitlines(), 1):
        where = f"{rel}:{lineno}"
        for match in _R_LIBRARY.finditer(line):
            modules.add(match.group(1))
        for match in _R_NAMESPACE.finditer(line):
            modules.add(match.group(1))
        for risk, pattern in _R_CALL_RES:
            for match in pattern.finditer(line):
                name = "eval(parse())" if risk == "dynamic_exec" else f"{match.group(1)}()"
                findings.append((risk, where, name))
        if _R_FILE_WRITE.search(line):
            findings.append(("filesystem_write", where, "file(..., \"w\")"))
    for package in sorted(modules):
        if package in _R_NETWORK_PACKAGES:
            findings.append(("network", rel, f"the {package} package"))
        if package in _R_SUBPROCESS_PACKAGES:
            findings.append(("subprocess", rel, f"the {package} package"))
    return modules, findings


def _strip_r_comments(text: str) -> str:
    """R source with every `#` comment removed, and strings left alone."""
    out = []
    for line in text.splitlines():
        quote, escaped = None, False
        for i, ch in enumerate(line):
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in "\"'`":
                quote = ch
            elif ch == "#":
                line = line[:i]
                break
        out.append(line)
    return "\n".join(out)


class _Scan(ast.NodeVisitor):
    """One Python file's top-level imports, and every risky construct with its line."""

    def __init__(self, rel: str):
        self.rel = rel
        self.modules = set()
        self.findings = []                   # (risk, "file:line", what)
        self._alias = {}                     # local name -> real module
        self._from = {}                      # local name -> "module.attr"

    # -- helpers ---------------------------------------------------------
    def _add(self, risk: str, node, what: str) -> None:
        self.findings.append((risk, f"{self.rel}:{getattr(node, 'lineno', 0)}", what))

    def _module_risk(self, top: str, node, what: str) -> None:
        if top in _NETWORK_MODULES:
            self._add("network", node, what)
        if top in _SUBPROCESS_MODULES:
            self._add("subprocess", node, what)
        if top in _DYNAMIC_MODULES:
            self._add("dynamic_exec", node, what)

    def _dotted(self, node) -> str:
        """"os.path.join" for an Attribute/Name chain, else ""."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        parts.append(node.id)
        parts.reverse()
        head = self._alias.get(parts[0], parts[0])
        if len(parts) == 1:
            return self._from.get(parts[0], head)
        return ".".join([head] + parts[1:])

    # -- imports ---------------------------------------------------------
    def visit_Import(self, node):
        for alias in node.names:
            top = alias.name.split(".")[0]
            self.modules.add(top)
            self._alias[alias.asname or top] = alias.name if alias.asname else top
            self._module_risk(top, node, f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level:                       # `from . import helper` — in-bundle
            self.generic_visit(node)
            return
        module = node.module or ""
        top = module.split(".")[0]
        if top:
            self.modules.add(top)
            self._module_risk(top, node, f"from {module} import "
                              + ", ".join(a.name for a in node.names))
        for alias in node.names:
            self._from[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # -- calls -----------------------------------------------------------
    def visit_Call(self, node):
        name = self._dotted(node.func)
        last = name.rsplit(".", 1)[-1] if name else ""
        if name in ("open", "io.open", "codecs.open", "gzip.open", "builtins.open"):
            writes, mode = _open_mode(node)
            if writes:
                shown = f"open(..., {mode!r})" if mode else "open() with a computed mode"
                self._add("filesystem_write", node, shown)
        if name in _WRITE_CALLS:
            self._add("filesystem_write", node, f"{name}()")
        elif last in _WRITE_METHODS:
            self._add("filesystem_write", node, f".{last}()")
        if name in _DYNAMIC_CALLS or last in ("import_module", "reload"):
            self._add("dynamic_exec", node, f"{name or last}()")
        if name in ("os.system", "os.popen", "os.fork", "os.posix_spawn") \
                or name.startswith(("os.exec", "os.spawn")):
            self._add("subprocess", node, f"{name}()")
        if name in ("os.putenv", "os.unsetenv"):
            self._add("env", node, f"{name}()")
        if name.startswith("os.environ.") and last in _ENV_METHODS:
            self._add("env", node, f"{name}()")
        self.generic_visit(node)

    # -- os.environ["X"] = ... -------------------------------------------
    def visit_Assign(self, node):
        for target in node.targets:
            self._env_target(target, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._env_target(node.target, node)
        self.generic_visit(node)

    def visit_Delete(self, node):
        for target in node.targets:
            self._env_target(target, node, verb="del ")
        self.generic_visit(node)

    def _env_target(self, target, node, verb: str = "") -> None:
        if isinstance(target, ast.Subscript) and self._dotted(target.value) == "os.environ":
            self._add("env", node, f"{verb}os.environ[...]")


def _open_mode(node):
    """(does this open() write?, the literal mode if there was one)."""
    mode = None
    if len(node.args) >= 2:
        mode = node.args[1]
    for kw in node.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return False, "r"
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(c in mode.value for c in "wax+"), mode.value
    return True, None                        # computed: report rather than guess


# ===========================================================================
# Gate: fixture
# ===========================================================================
def _gate_fixture(ctx: _Context):
    m = ctx.require_manifest()
    ctx.require("entry", "the entry file did not load, so there is nothing to run.")
    dataset = ctx.dataset()
    try:
        ctx.params = _default_params(m, dataset)
    except ManifestError as exc:
        return "fail", (
            f"no default parameter set could be built: {exc} Give the parameter a "
            "`default` (or mark it optional) — the generated form opens on the defaults, "
            "so a tool that cannot run on them cannot be run by a first-time user either."
        ), None

    run = ctx.run(ctx.params)
    ctx.first = run
    code = run.get("code")
    detail = {"params": ctx.params, "seconds": _tool_seconds(run),
              "logs": (run.get("logs") or [])[:20]}

    # A GPU tool is held to its CPU fallback here, as in CI; say so when it fails.
    cpu_note = (f' The tool declares "gpu": "{m["gpu"]}", and the gates run it on the CPU '
                "(as CI does), so it needs a CPU fallback."
                if m.get("gpu") and ctx.device == "cpu" else "")

    if code == "cancelled":
        return "fail", "the run was cancelled before it finished.", detail
    if code in ("no_result", "no_subprocess"):
        return "fail", (
            run["message"] + " A segfault in compiled code, a hard exit (os._exit() in "
            "Python, quit() in R), or an interpreter that died while loading the tool all "
            "look like this."
        ), dict(detail, notes=(run.get("stderr") or "").splitlines()[-10:])
    if run["status"] == "error":
        if code in RESULT_CODES:
            # The tool ran; its RETURN VALUE is what the result contract
            # rejected. That is the result gate's verdict, and reporting it
            # twice would send an author looking for two bugs.
            return "pass", (
                f"the tool ran to completion in {_tool_seconds(run):.2f} s, but its return "
                "value was rejected — see the result gate."
            ), detail
        if code == "crash":
            return "fail", (
                f"the tool crashed after {_tool_seconds(run):.2f} s on its own defaults: "
                f"{run['message']}{cpu_note}"
            ), dict(detail, notes=(run.get("traceback") or "").splitlines()[-14:])
        info = ctx.fixture_info
        return "fail", (
            f"the tool refused its own defaults after {_tool_seconds(run):.2f} s: "
            f"{run['message']} (code: {code}). The fixture has {info['n_cells']} cells, "
            f"{info['n_sections']} sections, {len(info['cell_types'])} cell types, one extra "
            f"categorical, two continuous columns and {info['n_genes']} genes, so a refusal "
            f"here is a refusal on almost any dataset.{cpu_note}"
        ), detail

    got = [k for k in OUTPUT_KINDS if k in (run.get("result") or {})]
    n_logs = len(run.get("logs") or [])
    logged = f" and {n_logs} log line{'' if n_logs == 1 else 's'}." if n_logs else "."
    swept = ""
    if m.get("sweep"):
        # A declared sweep is a promise the viewer builds a control on, so it
        # is exercised here — across the parameter's whole declared range —
        # through the same driver a user's sweep takes.
        status, note, sweep_detail = _check_sweep(ctx, m)
        detail = dict(detail, sweep=sweep_detail)
        if status == "fail":
            return "fail", note, detail
        swept = " " + note
    return "pass", (
        f"ran on the {ctx.fixture_info['n_cells']}-cell fixture in "
        f"{_tool_seconds(run):.2f} s, returning {', '.join(got)}{logged}{swept}"
    ), detail


def _check_sweep(ctx: _Context, m: dict):
    """Run the declared sweep at its minimum, a middle value and its maximum.

    Every step must be a result that honours the manifest's outputs, or a
    refusal with a sentence (PluginError) — never a crash or a malformed
    value — and at least one step must succeed.
    """
    spec, values = sweep_values(m, None, None, 3)
    run = ctx.run(ctx.params, sweep={"param": m["sweep"], "values": values})
    detail = {"param": spec["name"], "values": values, "seconds": _tool_seconds(run)}
    if run.get("status") != "ok":
        return "fail", (
            f"the tool declares a sweep over {spec['name']}, but a sweep at "
            f"{', '.join(str(v) for v in values)} did not complete "
            f"({run.get('code')}: {run.get('message')})."
        ), detail
    sweep = (run.get("result") or {}).get("sweep") or {}
    steps = sweep.get("steps") or []
    if sweep.get("param") != spec["name"] or len(steps) != len(values):
        return "fail", (f"a sweep over {spec['name']} returned {len(steps)} steps for "
                        f"{len(values)} values."), detail
    refused = [values[i] for i, s in enumerate(steps) if "error" in s]
    for value, step in zip(values, steps):
        if "error" in step:
            continue
        missing, extra = output_mismatch(m, step)
        if missing or extra:
            return "fail", (
                f"at {spec['name']} = {value} the sweep returned "
                f"{', '.join(k for k in OUTPUT_KINDS if k in step) or 'nothing'}; the "
                f"manifest declares {', '.join(m['outputs'])}."
            ), dict(detail, missing=missing, extra=extra)
    detail["refused"] = refused
    return "pass", (
        f"A {len(values)}-step sweep over {spec['name']} ({values[0]} to {values[-1]}) "
        f"returned every step in {_tool_seconds(run):.2f} s"
        + (f", refusing {', '.join(str(v) for v in refused)} with a sentence." if refused else ".")
    ), detail


def _default_params(manifest: dict, dataset_dir: str) -> dict:
    """The parameter dict the generated form would submit on first open.

    Manifest defaults where there are any; otherwise the fixture's OWN gene /
    field / level, because that is what the viewer's generated control does — a
    `gene` parameter is a search box over THIS dataset's panel, so a literal
    default in a manifest would be a gene name that is wrong on every dataset
    but the one its author had open. Filling from the dataset is not a
    convenience here, it is the same rule the form obeys.

    Two parameters of the same kind get DIFFERENT values, in declaration order.
    A tool that takes two cell types may correctly refuse when they are the
    same one ("random labelling has nothing to permute"), so a validator that
    could only ever hand out level 0 would fail a tool for being right. Values
    wrap when the dataset runs out.

    Coercion itself is still `manifest.coerce_params`; this only decides what
    to hand it.
    """
    from .dataset import Dataset
    ds = Dataset(dataset_dir)
    fields = [f["name"] for f in ds.categorical_fields()]
    continuous = [c["name"] for c in ds.continuous_fields()]
    genes = _expressed_genes(ds, 8)
    supplied: dict = {}
    handed: dict = {}

    def take(key, options):
        """The next unused option for this kind, wrapping when they run out."""
        if not options:
            return None
        i = handed.get(key, 0)
        handed[key] = i + 1
        return options[i % len(options)]

    def field_for(p):
        named = p.get("field_param")
        if named:
            chosen = supplied.get(named)
            if chosen is None:
                other = next((q for q in manifest["params"] if q["name"] == named), None)
                chosen = (other or {}).get("default")
            if chosen:
                return str(chosen)
        return fields[0] if fields else None

    for p in manifest["params"]:
        if p.get("default") not in (None, ""):
            continue
        ptype, name = p["type"], p["name"]
        if ptype == "gene":
            supplied[name] = take("gene", genes)
        elif ptype == "genes":
            picked = [take("gene", genes) for _ in range(min(3, p.get("max_items") or 3))]
            supplied[name] = [g for g in picked if g]
        elif ptype == "categorical_field":
            supplied[name] = take("categorical_field", fields)
        elif ptype == "continuous_field":
            supplied[name] = take("continuous_field", continuous)
        elif ptype == "category_value":
            field = field_for(p)
            if field:
                levels, _codes = ds.categorical(field)
                supplied[name] = take(("level", field), list(levels))
    return coerce_params(manifest, {k: v for k, v in supplied.items() if v is not None})


def _expressed_genes(ds, n: int) -> list:
    """The first `n` genes that are expressed in at least one cell.

    The fixture's first gene is called "Silent" and is expressed in nothing, on
    purpose. Handing that to a tool as its DEFAULT would fail every tool that
    correctly refuses an empty gene — the awkward cases are there to be met by
    a tool that goes looking for them, not to be the opening move.
    """
    import numpy as np
    out = []
    for name in ds.gene_names:
        if np.count_nonzero(ds.gene(name, "raw")):
            out.append(name)
            if len(out) >= n:
                return out
    return out or ds.gene_names[:max(1, n)]


def _blank_outputs(payload: dict) -> list:
    """Painted outputs holding no value at all.

    The result contract cannot catch this and should not try: a full-length
    array of NaN is a LEGAL result, and NaN is the documented "no value here"
    marker, so at the type level a blank map is indistinguishable from a
    computed one. The difference is in the meaning, and that is where the
    incident lived: at k >= the cells in a section, a hotspot statistic's null
    variance collapsed to exactly zero, the tool returned NaN for every cell,
    and reported "0 cells (0.0%) sit in a significant hotspot ... across 0
    tests". It neither crashed nor refused: it painted nothing and called it
    an answer.

    A tool with nothing to say at these parameters has exactly one correct
    move, and this is not it: raise PluginError (pv_error in R) so the user is
    told why and what to change.

    Two tiers, because certainty differs by where it happens. At the tool's OWN
    DEFAULTS a wholly blank output is a defect with no competing explanation —
    those are the parameters its author recommends, on a dataset built to be
    answerable — so the result gate fails on it. At a CORNER it is only
    reported: an extreme value may be unreachable on 600 synthetic cells while
    being ordinary on a real slide.

    "No value AT ALL", not "mostly NaN": partial coverage is normal and
    correct — the fixture's 8 cells with no coordinates make 592/600 the right
    answer for most tools. `vectors` is checked per ARROW, because an arrow is
    drawable only when all four of x, y, u and v are finite at that index.
    """
    blank = []
    cells = payload.get("cells")
    if cells is not None:
        values = cells.get("values") or []
        if not any(v is not None for v in values):
            blank.append(f"cells (0 of {len(values)} values are finite)")
    labels = payload.get("labels")
    if labels is not None:
        values = labels.get("values") or []
        if not any(v is not None for v in values):
            blank.append(f"labels (0 of {len(values)} cells have a label)")
    matrix = payload.get("matrix")
    if matrix is not None:
        rows = matrix.get("values") or []
        total = sum(len(r) for r in rows)
        if not any(v is not None for r in rows for v in r):
            blank.append(f"matrix (0 of {total} values are finite)")
    series = payload.get("series")
    if series is not None:
        lines = series.get("lines") or []
        total = sum(len(ln.get("y") or []) for ln in lines)
        if not any(y is not None for ln in lines for y in (ln.get("y") or [])):
            blank.append(f"series (0 of {total} points are finite)")
    radar = payload.get("radar")
    if radar is not None and not any(v is not None for p in radar.get("profiles", [])
                                     for v in p.get("values", [])):
        blank.append("radar (no finite profile values)")
    chord = payload.get("chord")
    if chord is not None and not any(link.get("value", 0) > 0 for link in chord.get("links", [])):
        blank.append("chord (no positive links)")
    images = payload.get("images")
    if images is not None and not images:
        blank.append("images (no image)")
    vectors = payload.get("vectors")
    if vectors is not None:
        total = len(vectors.get("x") or [])
        drawable = sum(1 for i in range(total)
                       if all(vectors.get(key) is not None
                              and vectors[key][i] is not None
                              for key in ("x", "y", "u", "v")))
        if not drawable:
            blank.append(f"vectors (0 of {total} arrows are drawable)")
    return blank


# ===========================================================================
# Gate: result
# ===========================================================================
def _gate_result(ctx: _Context):
    m = ctx.require_manifest()
    run = ctx.first
    if run is None:
        raise _Skip("skipped: the tool produced no return value to check.")
    if run["status"] == "error":
        if run.get("code") in RESULT_CODES:
            return "fail", (f"the return value was rejected: {run['message']} "
                            f"(code: {run['code']})"), {"code": run["code"]}
        raise _Skip("skipped: the run did not reach a return value.")

    payload = run.get("result") or {}
    seen = [k for k in OUTPUT_KINDS if k in payload]
    ctx.outputs_seen = seen
    declared = m["outputs"]
    missing, extra = output_mismatch(m, payload)
    detail = {"declared": declared, "seen": seen, "missing": missing, "extra": extra}

    # Schemas 2 and 3 are held to their outputs by the runner itself, so for
    # them this repeats what the run already enforced; a schema-1 tool is
    # only held to them here.
    if missing and extra:
        return "fail", (
            f"the manifest declares {', '.join(declared)} but the run returned "
            f"{', '.join(seen)}: {', '.join(missing)} never arrived and "
            f"{', '.join(extra)} was not declared."
        ), detail
    if missing:
        return "fail", (
            f"the manifest declares {', '.join(declared)} but the run returned only "
            f"{', '.join(seen) or 'nothing'}; the viewer would draw an empty "
            f"{missing[0]} section. Declare what the tool actually returns."
        ), detail
    if extra:
        return "fail", (
            f"the run returned {', '.join(extra)}, which the manifest does not declare. "
            "The manifest is the store listing a user chose from — a tool that paints the "
            "map when its card says 'table' is a tool nobody picked."
        ), detail
    blank = _blank_outputs(payload)
    if blank:
        return "fail", (
            f"the run returned {' and '.join(blank)} — a legal result that paints nothing, "
            "under a summary that reads as an answer. If the tool has no answer at these "
            "parameters, refuse with PluginError (pv_error in R) and say what to change; "
            "downstream, a blank map is indistinguishable from a computed one."
        ), dict(detail, blank=blank)
    return "pass", f"returned the declared outputs: {', '.join(seen)}.", detail


# ===========================================================================
# Gate: edge
# ===========================================================================
def _gate_edge(ctx: _Context):
    m = ctx.require_manifest()
    ctx.require("fixture", "the tool did not complete a default run, so a corner would only "
                           "repeat that failure.")
    base = ctx.params
    plans = [("minimum", _edge_params(m, base, "min"))]
    top = _edge_params(m, base, "max")
    # Both ends, not just the bottom: the top of a declared range is where a
    # neighbourhood becomes the whole section and a statistic can go
    # undefined, which only a run there reaches.
    if top != plans[0][1]:
        plans.append(("maximum", top))

    sentences, detail = [], {"corners": {}}
    for label, params in plans:
        run = ctx.run(params)
        text = _corner_text(params, base)
        status, sentence, notes = _corner_outcome(label, text, run)
        detail["corners"][label] = {"params": params, "status": status,
                                    "seconds": _tool_seconds(run),
                                    "blank": _blank_outputs(run.get("result") or {})}
        if status == "fail":
            # Stop here rather than paying for the other corner: a tool broken
            # at one end is a tool to fix and re-run.
            return "fail", sentence, dict(detail, notes=notes) if notes else detail
        sentences.append(sentence)

    if len(plans) == 1:
        sentences.append("there is no other corner — no numeric, boolean or select parameter "
                         "to move")
    return "pass", "; ".join(sentences) + ".", detail


def _corner_outcome(label: str, corner: str, run: dict):
    """(status, sentence, notes) for one end of the declared parameter space.

    The pass condition is the same at both ends, and it is not "did not
    crash": a PluginError is a tool REFUSING an input in words a user can act
    on, which is correct behaviour wherever it happens — "k must be at least 5"
    at the bottom and "no two categories have 5000 cells" at the top are both
    answers. A traceback or a malformed result is not.
    """
    code = run.get("code")
    if code == "cancelled":
        return "fail", f"the run at the {label} corner ({corner}) was cancelled", None
    if code in ("no_result", "no_subprocess"):
        return "fail", f"at the {label} corner ({corner}), {_lower_first(run['message'])}", \
               (run.get("stderr") or "").splitlines()[-10:]
    if run["status"] == "error":
        if code == "crash":
            return "fail", (f"at the {label} corner ({corner}) the tool crashed: "
                            f"{run['message']}. A tool that cannot answer at the end of its "
                            "own declared range must say so with PluginError, not with a "
                            "traceback."), (run.get("traceback") or "").splitlines()[-14:]
        if code in RESULT_CODES:
            return "fail", (f"at the {label} corner ({corner}) the return value was rejected: "
                            f"{run['message']} (code: {code})."), None
        return "pass", (f"at the {label} corner ({corner}) it refused in words rather than "
                        f"crashing: {run['message']} (code: {code})"), None
    blank = _blank_outputs(run.get("result") or {})
    if blank:
        # Reported, not failed: "no cell qualifies at this threshold" and "this
        # statistic means nothing here" arrive as an identical full-length
        # array of NaN, so failing on it would block the starter tool every
        # author begins from (its maximum threshold is unreachable on 600
        # cells) to catch a defect only its author can identify.
        return "pass", (
            f"ran at the {label} corner ({corner}) in {_tool_seconds(run):.2f} s but painted "
            f"nothing there — {' and '.join(blank)}. Correct if no cell can qualify at those "
            "parameters, a defect if the statistic is undefined; if it is the latter, refuse "
            "with PluginError so the user is told which"
        ), None
    return "pass", f"ran at the {label} corner ({corner}) in {_tool_seconds(run):.2f} s", None


def _corner_text(params: dict, base: dict) -> str:
    moved = {k: v for k, v in params.items() if base.get(k) != v}
    return ", ".join(f"{k}={v!r}" for k, v in sorted(moved.items())) or "the declared defaults"


def _edge_params(manifest: dict, base: dict, corner: str = "min") -> dict:
    """Every numeric parameter at one end of its declared range.

    min: numbers at `min`, bools False, selects at their first option.
    max: numbers at `max`, bools True, selects at their last option.

    Selects move too, and to the LAST option rather than a random one, because
    the option list is ordered by the author and the far end is the one least
    likely to have been the one they developed against.
    """
    lowest = corner == "min"
    out = dict(base)
    for p in manifest["params"]:
        if p["type"] in ("int", "float"):
            out[p["name"]] = p["min"] if lowest else p["max"]
        elif p["type"] == "bool":
            out[p["name"]] = not lowest
        elif p["type"] == "select":
            out[p["name"]] = (p["options"][0] if lowest else p["options"][-1])["value"]
    # Back through the same coercion the host applies, so `min` on an int
    # parameter reaches run() as an int and not as the float the manifest holds.
    return coerce_params(manifest, out)


# ===========================================================================
# Gate: no_mutation
# ===========================================================================
def _gate_no_mutation(ctx: _Context):
    if not ctx.ran_any or ctx.pre_hashes is None:
        raise _Skip("skipped: the tool never ran, so it cannot have changed anything.")
    after = _hash_tree(ctx.dataset())
    before = ctx.pre_hashes
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    removed = sorted(k for k in before if k not in after)
    added = sorted(k for k in after if k not in before)
    detail = {"changed": changed, "added": added, "removed": removed,
              "files": len(before)}
    if not (changed or removed or added):
        return "pass", (
            f"the fixture dataset's {len(before)} files are byte-identical after every run "
            "(plugin_cache/, where ds.knn caches, is excluded by design)."
        ), detail
    parts = []
    if changed:
        parts.append("modified " + ", ".join(changed[:3]))
    if added:
        parts.append("wrote " + ", ".join(added[:3]))
    if removed:
        parts.append("deleted " + ", ".join(removed[:3]))
    return "fail", (
        "the tool " + "; ".join(parts) + " inside the dataset directory. A dataset is "
        "read-only to a tool — several tools and several users share one, and only "
        "plugin_cache/ may be written."
    ), detail


def _hash_tree(root: str) -> dict:
    """sha256 per file under `root`, EXCLUDING plugin_cache/.

    `Dataset.knn` caches its neighbour graph there on purpose (two tools
    asking for knn(15) pay once), so a new file in plugin_cache is a tool
    using the API correctly and must not read as a mutation.
    """
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "plugin_cache"]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            digest = hashlib.sha256()
            try:
                with open(full, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(chunk)
            except OSError as exc:
                out[rel] = f"unreadable: {exc}"
                continue
            out[rel] = digest.hexdigest()
    return out


_GATES = {
    "bundle": _gate_bundle,
    "manifest": _gate_manifest,
    "environment": _gate_environment,
    "entry": _gate_entry,
    "imports": _gate_imports,
    "fixture": _gate_fixture,
    "result": _gate_result,
    "edge": _gate_edge,
    "no_mutation": _gate_no_mutation,
}


# ===========================================================================
# Small helpers
# ===========================================================================
def _lower_first(text: str) -> str:
    """Splice a driver sentence into the middle of a gate sentence."""
    text = str(text or "the run failed with no message.")
    return text[0].lower() + text[1:]


def _tool_seconds(run: dict) -> float:
    """The child's OWN measure of a run, falling back to the parent's clock.

    `launch.run` reports both: `seconds` is the wall clock a user would have
    waited, interpreter (and environment) start-up and all, and `tool_seconds`
    is what happened inside the runner. Gate messages quote the second one,
    because it is the part an author tuning their tool can actually make
    smaller.
    """
    value = run.get("tool_seconds")
    if value is None:
        value = run.get("seconds") or 0.0
    return round(float(value), 3)


def _size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} kB"
    return f"{n} B"


# ===========================================================================
# Human-readable report
# ===========================================================================
#: Column geometry for the per-gate lines. Wide enough for "environment" and
#: for a four-digit second count, so nothing in the message column ever shifts.
_STATUS_W, _ID_W, _SEC_W = 4, 12, 8
_INDENT = 2 + _STATUS_W + 2 + _ID_W + 2 + _SEC_W + 2
_WIDTH = 98


def format_report(report: dict) -> str:
    """The report as a tool author reads it while iterating.

    Plain text, no colour: this output is as likely to land in a CI log or a
    store page as in a terminal.
    """
    out = []
    title = " ".join(x for x in (report.get("plugin_id"), report.get("version")) if x)
    if report.get("name"):
        title = f"{title} — {report['name']}" if title else report["name"]
    if report.get("language"):
        kind = f"{_LANGUAGE[report['language']]} tool"
        if report.get("gpu"):
            kind += f", GPU {report['gpu']}"
        title = f"{title} ({kind})"
    out.append(title or "(the manifest did not load, so this tool has no identity yet)")
    out.append("")

    for gate in report["gates"]:
        head = (f"  {gate['status'].upper():<{_STATUS_W}}  {gate['id']:<{_ID_W}}  "
                f"{gate['seconds']:>{_SEC_W - 2}.2f} s  ")
        out.append(textwrap.fill(gate["message"], width=_WIDTH,
                                 initial_indent=head,
                                 subsequent_indent=" " * _INDENT))
        for note in ((gate.get("detail") or {}).get("notes") or [])[:14]:
            # Wrapped, not truncated: these carry file listings, parameter
            # lists and traceback lines, and a clipped one ("max_rad") reads
            # as a name rather than as an elision.
            out.append(textwrap.fill(str(note).rstrip(), width=_WIDTH,
                                     initial_indent=" " * (_INDENT - 2),
                                     subsequent_indent=" " * _INDENT))

    out.append("")
    out.append(_footer_line("imports", ", ".join(report["imports"]) or "nothing"))
    out.append(_footer_line(
        "risks", ", ".join(report["risks"]) + "  (disclosed, not blocked — a tool is "
        "arbitrary code)" if report["risks"] else "none found"))
    out.append(_footer_line("outputs", ", ".join(report["outputs_seen"]) or "none returned"))
    out.append("")

    counts = {}
    for gate in report["gates"]:
        counts[gate["status"]] = counts.get(gate["status"], 0) + 1
    required = [g for g in report["gates"] if g["required"]]
    n_pass = sum(1 for g in required if g["status"] == "pass")
    tail = (f"{n_pass}/{len(required)} required gates pass"
            + (f", {counts['warn']} warning" + ("s" if counts["warn"] > 1 else "")
               if counts.get("warn") else "")
            + (f", {counts['skip']} skipped" if counts.get("skip") else ""))
    if report["ok"]:
        verdict = f"PASSED in {report['seconds']:.1f} s — {tail}."
    else:
        blocking = [g["id"] for g in required if g["status"] != "pass"]
        verdict = (f"FAILED in {report['seconds']:.1f} s — {tail}; "
                   f"blocking: {', '.join(blocking)}.")
    out.append(textwrap.fill(verdict, width=_WIDTH, initial_indent="  ",
                             subsequent_indent="  "))
    return "\n".join(out)


def _footer_line(label: str, value: str) -> str:
    return textwrap.fill(value, width=_WIDTH,
                         initial_indent=f"  {label:<9} ",
                         subsequent_indent=" " * 12)


# ===========================================================================
# The entry probe's child
# ===========================================================================
def main(argv: list) -> int:
    """`python -m pointillviz_tool.validate --probe-entry <dir> <entry>`: the entry gate's child.

    Authors validate with `python -m pointillviz_tool validate`.
    """
    if len(argv) == 3 and argv[0] == "--probe-entry":
        return _probe_main(argv[1], argv[2])
    sys.stderr.write("usage: python -m pointillviz_tool validate <dir|bundle> "
                     "[--json] [--env own|app] [--device cpu|cuda]\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
