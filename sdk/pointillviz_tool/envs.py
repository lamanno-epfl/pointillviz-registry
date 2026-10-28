"""pixi, for tools that bring an environment: build it, launch into it, or share PointillViz's.

A tool with a `pixi.toml` runs in one of two places, and the person installing
it chooses which:

  * its OWN environment: `install_dedicated` builds it from the tool's
    pixi.lock inside the tool's folder, and `launcher(tool_dir, "own", device)`
    starts Python in it. It cannot disturb anything else, which is why the app
    recommends it.
  * POINTILLVIZ'S environment: `apply_shared` merges the tool's requirements
    into the app's own pixi workspace, and `launcher(tool_dir, "app", device)`
    is the interpreter running right now. One solve then covers everything,
    so `check_shared` first asks pixi what installing would change.

pixi does every install and writes every lock file. This module reads
pixi.toml but never writes it: every change goes through the pixi CLI, so the
manifest stays in the shape pixi itself would leave it.

Sharing, precisely
------------------
A tool's `[dependencies]`, `[pypi-dependencies]` and channels become the
feature `tool-<id>` of the app's manifest, and the default environment is
made of the features it already had plus that one. Platform-specific tables
and the tool's other environments (such as `gpu`) stay out: the shared
environment is the app's default one.

The answer `check_shared` gives follows the rule PointillViz promises its
users: warn only when the tool is INCONSISTENT with what is installed, meaning
the solve fails, or a package the app already has would change or disappear. A
tool that only adds packages is consistent, whatever their versions.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

from .dataset import PluginError

#: Longest pixi error kept for a message. A solver explanation leads with its
#: conclusion and can run to hundreds of lines, so the head is what is kept.
_MAX_MESSAGE = 8000
#: The first lines of pixi notices that the person watching an install cannot
#: act on, kept out of `log` (never out of an error): pixi 0.83's warnings
#: about the manifest it parses (the `[system-requirements]` deprecation that a
#: GPU tool's pixi.toml triggers, with a local path), and the note that a
#: feature serves no environment yet, which a shared install's own steps
#: provoke because `_wire` adds the requirements before the environment. pixi
#: starts that note with "⚠️" under a UTF-8 locale and with "warning:" otherwise.
_NOTICE = re.compile(r"\s*WARN Encountered \d+ warnings? while parsing the manifest"
                     r"|(?:warning:|⚠️?)\s*The feature \S+ is not used in any environment")
#: A line that starts a new record of pixi's log; a notice runs until one.
_RECORD = re.compile(r"\s*(TRACE|DEBUG|INFO|WARN|ERROR)\s")


def pixi_exe() -> str:
    """The pixi binary: `$POINTILLVIZ_PIXI`, else `pixi` on PATH, else the
    installer's default place (`$PIXI_HOME/bin`, which is `~/.pixi/bin`
    unless PIXI_HOME says otherwise)."""
    configured = os.environ.get("POINTILLVIZ_PIXI")
    if configured:
        if os.path.isfile(configured):
            return configured
        raise PluginError("no_pixi", f"POINTILLVIZ_PIXI is set to {configured}, "
                                     "which is not a file.")
    found = shutil.which("pixi")
    if found:
        return found
    home = os.environ.get("PIXI_HOME") or os.path.join(os.path.expanduser("~"), ".pixi")
    exe = os.path.join(home, "bin", "pixi.exe" if os.name == "nt" else "pixi")
    if os.path.isfile(exe):
        return exe
    raise PluginError("no_pixi", "pixi is not installed. Install it from "
                                 "https://pixi.sh, or set POINTILLVIZ_PIXI to its path.")


def has_pixi(tool_dir: str) -> bool:
    """Whether the tool declares an environment (a `pixi.toml` in its folder)."""
    return os.path.isfile(os.path.join(tool_dir, "pixi.toml"))


def tool_environments(tool_dir: str) -> list:
    """The pixi environments a tool declares: `default` first, then the
    others in order. Empty when it has no pixi.toml.

    pixi reads its own manifest (`pixi info`), so this works on Python 3.10,
    which has no tomllib, and agrees with what pixi will build.
    """
    if not has_pixi(tool_dir):
        return []
    names = [e.get("name") for e in _info(tool_dir).get("environments_info") or []]
    return ["default"] + [name for name in names if name and name != "default"]


def launcher(tool_dir: str, mode: str, device: str = "cpu") -> list:
    """The argv prefix that starts Python for a run of this tool.

    "own" is the tool's own environment: `gpu` when the run is on the GPU and
    the tool declares that environment, `default` otherwise. `--frozen` takes
    pixi.lock exactly as it is: a run must neither rewrite the lock the gates
    checked nor wait on a solver (pixi still installs the environment first
    if it is missing). `--quiet` keeps pixi's own notices out of the tool's
    log: pixi 0.83 warns about `[system-requirements]`, the usual way to ask
    for a CUDA build, in twelve lines on every run. Its errors still come
    through. "app" is the interpreter running this code, which is
    PointillViz's environment for a tool installed into it or a legacy tool.
    """
    if mode == "app":
        return [sys.executable]
    if mode != "own":
        raise ValueError(f"mode must be 'own' or 'app', not {mode!r}.")
    env = "gpu" if device == "cuda" and "gpu" in tool_environments(tool_dir) else "default"
    return [pixi_exe(), "run", "--quiet", "--frozen", "--manifest-path", _manifest(tool_dir),
            "-e", env, "python"]


def install_dedicated(tool_dir: str, environment: str = "default", log=None) -> None:
    """Build one of the tool's own environments, exactly as its pixi.lock pins it.

    It lands in the tool's folder (`.pixi/envs/<environment>`), so removing
    the tool removes its environment with it. The folder must therefore
    already be where it will stay: a pixi environment does not survive being
    moved. pixi's progress goes to `log` line by line.
    """
    _pixi(["install", "--frozen", "--manifest-path", _manifest(tool_dir),
           "-e", environment], log)


def lock_check(tool_dir: str) -> tuple:
    """Whether the tool's pixi.lock is still what `pixi lock` would write for
    its pixi.toml, as `(ok, message)`. Solves only when the lock no longer
    satisfies the manifest; installs nothing."""
    if not os.path.isfile(os.path.join(tool_dir, "pixi.lock")):
        return False, ("pixi.lock is missing. Run `pixi lock` in the tool's "
                       "folder and include the file.")
    status, _out, err = _pixi(["lock", "--check", "--manifest-path",
                               _manifest(tool_dir)], check=False)
    if status == 0:
        return True, "pixi.lock matches pixi.toml."
    return False, ("pixi.lock does not match pixi.toml. Run `pixi lock` in the "
                   "tool's folder.\n" + err.strip()[:_MAX_MESSAGE])


def app_manifest():
    """PointillViz's own pixi.toml when this process runs in its pixi
    environment, else None.

    `pixi run` exports the manifest's path as PIXI_PROJECT_MANIFEST and, in
    newer releases, also as PIXI_WORKSPACE_MANIFEST. Shared installs edit that
    file, so outside a pixi environment there is nothing to share.
    """
    path = (os.environ.get("PIXI_WORKSPACE_MANIFEST")
            or os.environ.get("PIXI_PROJECT_MANIFEST"))
    if path and os.path.basename(path) == "pixi.toml" and os.path.isfile(path):
        return os.path.abspath(path)
    return None


def check_shared(app_manifest: str, tool_dir: str, tool_id: str) -> dict:
    """What installing this tool into PointillViz's environment would change.

    Returns `{consistent, changes:[{name, before, after}], error}`. The change
    is made on a temporary copy of the app's pixi.toml and pixi.lock, and
    `pixi lock --json` reports what the solve would do to the lock. `changes`
    lists the packages the app already has that would move to another version
    or build (`after` is None for one that would disappear); `error` is pixi's
    explanation when it cannot solve at all, the solver's conflict included.
    """
    pixi_exe()          # "pixi is missing" is not an answer about consistency
    tmp = tempfile.mkdtemp(prefix="pv_shared_")
    try:
        manifest = os.path.join(tmp, "pixi.toml")
        shutil.copyfile(app_manifest, manifest)
        lock = os.path.join(os.path.dirname(app_manifest), "pixi.lock")
        if os.path.isfile(lock):
            shutil.copyfile(lock, os.path.join(tmp, "pixi.lock"))
        try:
            _wire(manifest, tool_dir, tool_id)
            _status, out, _err = _pixi(["lock", "--json", "--manifest-path", manifest])
        except PluginError as exc:
            return {"consistent": False, "changes": [], "error": exc.message}
        changes = _changes(json.loads(out))
        return {"consistent": not changes, "changes": changes, "error": None}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def apply_shared(app_manifest: str, tool_dir: str, tool_id: str, log=None) -> None:
    """Install the tool's requirements into PointillViz's environment.

    The same change `check_shared` tried, made on the real manifest, then
    `pixi install`, which solves, writes pixi.lock and installs. Reinstalling
    a tool replaces its previous requirements. On failure pixi.toml and
    pixi.lock are put back as they were and the PluginError is raised; an
    install that stopped half way is brought back in line with them the next
    time pixi installs or runs the environment.
    """
    with _restored_on_failure(app_manifest):
        _wire(app_manifest, tool_dir, tool_id, log)
        _pixi(["install", "--manifest-path", app_manifest], log)


def remove_shared(app_manifest: str, tool_id: str, log=None) -> None:
    """Take a tool's requirements back out of PointillViz's environment.

    Packages no other requirement needs are uninstalled by the `pixi install`
    that follows. Nothing happens for a tool that was never shared.
    """
    feature = _feature(tool_id)
    if feature not in (_read_toml(app_manifest).get("feature") or {}):
        return
    with _restored_on_failure(app_manifest):
        _unwire(app_manifest, feature, log)
        _pixi(["install", "--manifest-path", app_manifest], log)


# ===========================================================================
# Helpers
# ===========================================================================
def _manifest(tool_dir: str) -> str:
    return os.path.join(os.path.abspath(tool_dir), "pixi.toml")


def _feature(tool_id: str) -> str:
    return "tool-" + tool_id


def _info(tool_dir: str) -> dict:
    """`pixi info --json` for the tool's workspace; {} when pixi cannot read it."""
    _status, out, _err = _pixi(["info", "--json", "--manifest-path", _manifest(tool_dir)],
                               check=False)
    try:
        return json.loads(out)
    except ValueError:
        return {}


def _read_toml(path: str) -> dict:
    # Only shared mode reads TOML itself (the specs `pixi info` does not
    # give), and it runs inside PointillViz, on Python 3.11. Imported here so
    # the rest of this module works on 3.10, which tool environments may run.
    try:
        import tomllib
    except ModuleNotFoundError:
        raise PluginError("env_failed", "Installing into PointillViz's environment "
                                        "needs Python 3.11 or newer.") from None
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise PluginError("env_failed", f"{os.path.basename(path)} is not valid TOML: {exc}.")


def _pixi(args: list, log=None, check: bool = True) -> tuple:
    """Run pixi; return `(status, stdout, stderr)`.

    stderr is where pixi reports progress, warnings and errors. It goes to
    `log` line by line as it arrives, because an install can take minutes and
    the person waiting should see it move, without the notices of `_NOTICE`.
    stdout is read whole: the one caller that wants it reads a JSON document
    from `pixi lock --json`. stdin is closed, so a pixi that wants to ask a
    question fails instead of waiting for an answer nobody will type.

    With `check`, a non-zero status raises PluginError("env_failed") carrying
    pixi's own explanation.
    """
    env = dict(os.environ, PIXI_COLOR="never")
    proc = subprocess.Popen([pixi_exe(), *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out: list = []
    reader = threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True)
    reader.start()
    err = []
    emit = _without_notices(log)
    for raw in proc.stderr:
        line = raw.decode("utf-8", "replace").rstrip()
        err.append(line)
        emit(line)
    reader.join()
    proc.wait()
    proc.stdout.close()
    proc.stderr.close()
    stdout = (out[0] if out else b"").decode("utf-8", "replace")
    stderr = "\n".join(err)
    if check and proc.returncode != 0:
        raise PluginError("env_failed", _error_text(stderr) or
                          f"pixi {args[0]} failed (status {proc.returncode}).")
    return proc.returncode, stdout, stderr


def _without_notices(log):
    """A line sink that passes pixi's lines on to `log`, except the notices of
    `_NOTICE`: each starts with one of those lines and runs over the indented
    and blank lines after it. Blank lines are never passed on."""
    skipping = False

    def emit(line: str) -> None:
        nonlocal skipping
        if _NOTICE.match(line):
            skipping = True
        elif skipping and (not line.strip() or line[:1].isspace()) and not _RECORD.match(line):
            pass
        else:
            skipping = False
            if log is not None and line.strip():
                log(line)
    return emit


def _error_text(stderr: str) -> str:
    """pixi's explanation of a failure, without the warnings printed before it."""
    text = stderr.strip()
    at = text.find("Error:")
    if at > 0:
        text = text[at:]
    return text if len(text) <= _MAX_MESSAGE else text[:_MAX_MESSAGE] + "\n(…)"


def _default_features(data: dict) -> list:
    """The features a parsed manifest's default environment is made of."""
    env = (data.get("environments") or {}).get("default")
    if isinstance(env, dict):
        env = env.get("features")
    return list(env or [])


def _feature_flags(features: list) -> list:
    return [flag for f in features for flag in ("--feature", f)]


def _wire(manifest: str, tool_dir: str, tool_id: str, log=None) -> None:
    """Put a tool's requirements into feature `tool-<id>` of `manifest`, and
    that feature into its default environment.

    Writes the manifest only (`--no-install --frozen`): nothing is solved or
    installed until the caller locks or installs. The requirements are copied
    exactly as the tool wrote them, with no pinning: the feature is not in any
    environment while they are added.
    """
    feature = _feature(tool_id)
    _unwire(manifest, feature, log)     # an update replaces the old requirements
    tool = _read_toml(_manifest(tool_dir))
    conda = [_conda_spec(n, v) for n, v in (tool.get("dependencies") or {}).items()]
    pypi = [_pypi_spec(n, v) for n, v in (tool.get("pypi-dependencies") or {}).items()]
    if not conda and not pypi:
        return
    table = tool.get("workspace") or tool.get("project") or {}
    channels = [c if isinstance(c, str) else c.get("channel")
                for c in table.get("channels") or []]
    edit = ["--manifest-path", manifest, "--feature", feature, "--no-install", "--frozen"]
    if conda:
        _pixi(["add", *edit, *conda], log)
    if pypi:
        _pixi(["add", "--pypi", *edit, *pypi], log)
    if any(channels):
        _pixi(["workspace", "channel", "add", *edit, *filter(None, channels)], log)
    features = [f for f in _default_features(_read_toml(manifest)) if f != feature]
    _pixi(["workspace", "environment", "add", "default", "--force",
           "--manifest-path", manifest, *_feature_flags(features + [feature])], log)


def _unwire(manifest: str, feature: str, log=None) -> None:
    """Take `feature` out of the default environment, then out of `manifest`.

    In that order: pixi asks for confirmation before removing a feature that
    an environment still uses, and there is nobody to answer. A default
    environment left with no features is removed, which is the manifest as it
    was before any tool was shared.
    """
    data = _read_toml(manifest)
    if feature not in (data.get("feature") or {}):
        return
    rest = [f for f in _default_features(data) if f != feature]
    if rest:
        _pixi(["workspace", "environment", "add", "default", "--force",
               "--manifest-path", manifest, *_feature_flags(rest)], log)
    elif "default" in (data.get("environments") or {}):
        _pixi(["workspace", "environment", "remove", "default",
               "--manifest-path", manifest], log)
    _pixi(["workspace", "feature", "remove", feature, "--manifest-path", manifest], log)


def _conda_spec(name: str, value) -> str:
    """A pixi.toml conda requirement as the MatchSpec `pixi add` takes, which
    pixi writes back as the same entry: `numpy >=1.24`, `conda-forge::pytorch * cuda*`."""
    if isinstance(value, str):
        return f"{name} {value}"
    extra = set(value) - {"version", "build", "channel"}
    if extra:
        raise PluginError("env_failed", _unshareable(name, extra))
    spec = f"{name} {value.get('version') or '*'}"
    if value.get("build"):
        spec += f" {value['build']}"
    return f"{value['channel']}::{spec}" if value.get("channel") else spec


def _pypi_spec(name: str, value) -> str:
    """A `[pypi-dependencies]` entry as the requirement `pixi add --pypi`
    takes: `tqdm>=4`, `rich[jupyter]`."""
    if isinstance(value, str):
        return name if value.strip() in ("", "*") else name + value
    extra = set(value) - {"version", "extras"}
    if extra:
        raise PluginError("env_failed", _unshareable(name, extra))
    req = name + (f"[{','.join(value['extras'])}]" if value.get("extras") else "")
    version = str(value.get("version") or "*")
    return req if version == "*" else req + version


def _unshareable(name: str, keys) -> str:
    return (f"{name} is declared with {', '.join(sorted(keys))}, which cannot be "
            "added to PointillViz's environment. Install the tool in its own "
            "environment instead.")


def _changes(report: dict) -> list:
    """The entries of a `pixi lock --json` report that touch packages the app
    already has.

    Additions (`before` is null) are left out: a tool bringing packages of its
    own is the point of installing it. Every other entry is an existing
    package moving to another version or build, or disappearing. Each change
    is listed once, though every platform in the lock reports it.
    """
    seen, out = set(), []
    for platforms in (report.get("environment") or {}).values():
        for entries in platforms.values():
            for entry in entries:
                before, after = entry.get("before"), entry.get("after")
                if not before:
                    continue
                b, a = _version(before), _version(after)
                if b == a:      # a rebuild of one version: the file is the difference
                    b, a = _version(before, exact=True), _version(after, exact=True)
                key = (entry.get("name"), b, a)
                if key not in seen:
                    seen.add(key)
                    out.append({"name": key[0], "before": b, "after": a})
    return out


def _version(record, exact: bool = False):
    """A lock record's version: its own field for PyPI, the file name for
    conda (`<name>-<version>-<build>.conda`). `exact` gives the file name."""
    if not record:
        return None
    filename = (record.get("conda") or record.get("pypi") or "").rsplit("/", 1)[-1]
    if exact:
        return filename or None
    if record.get("version"):
        return str(record["version"])
    for ext in (".conda", ".tar.bz2"):
        if filename.endswith(ext):
            parts = filename[:-len(ext)].rsplit("-", 2)
            if len(parts) == 3:
                return parts[1]
    return filename or None


@contextlib.contextmanager
def _restored_on_failure(manifest: str):
    """Put pixi.toml and pixi.lock back, byte for byte, if the change inside fails.

    A half-applied change is worse than either outcome: a manifest that names
    requirements whose packages never arrived makes the next `pixi run serve`
    solve again, and fail the same way, before PointillViz can start.
    """
    saved = {}
    for path in (manifest, os.path.join(os.path.dirname(manifest), "pixi.lock")):
        try:
            with open(path, "rb") as fh:
                saved[path] = fh.read()
        except FileNotFoundError:
            saved[path] = None
    try:
        yield
    except BaseException:
        for path, data in saved.items():
            if data is None:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
            else:
                with open(path, "wb") as fh:
                    fh.write(data)
        raise
