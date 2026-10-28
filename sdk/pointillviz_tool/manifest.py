"""`plugin.json` — parsing, validation, and the parameter coercion both sides share.

A manifest is the whole interface between a tool and the app. It says what the
tool needs from a dataset (`requires`), what the user gets to choose
(`params`), and what the tool will hand back (`outputs`). From it the viewer
generates the entire form, so a tool author writes no UI and the app never
learns a tool's name.

Three schemas are understood, and a manifest names its own:

  1. Native outputs and parameters only.
  2. Adds custom views, files, optional outputs and `json` parameters: the
     host validates the JSON envelope, the tool its domain-specific meaning.
  3. Adds native R tools (an entry ending in `.R`), the `labels` and `images`
     outputs, and `gpu` ("optional" or "required"; absent for a tool that
     never uses one). It drops `timeout_s`, `memory_mb` and `requires.scipy`:
     time and memory are the author's responsibility, and dependencies are
     declared by the tool's own pixi environment. Schema 3 refuses those keys;
     schemas 1 and 2 ignore them.

Keys this module does not know are dropped, so the normalised manifest is the
whole contract the host and the viewer act on.
"""

from __future__ import annotations

import json
import math
import os
import re

from .results import NATIVE_OUTPUTS, OUTPUT_KINDS

SCHEMAS = (1, 2, 3)
#: The output kinds each schema may declare.
SCHEMA_OUTPUTS = {
    1: NATIVE_OUTPUTS,
    2: NATIVE_OUTPUTS + ("views", "files"),
    3: OUTPUT_KINDS,
}
#: What a schema-3 `gpu` may say. "optional": the tool runs on either device
#: and the user chooses. "required": the tool is meant for a GPU, so the app
#: selects the GPU by default and says so when the machine has none. Either
#: way it must also run on the CPU: the gates run every tool there (CI has no
#: GPU), so a GPU method checks for CUDA and falls back.
GPU_MODES = ("optional", "required")
MAX_JSON_PARAM_BYTES = 64 * 1024

PARAM_TYPES = (
    "int", "float", "bool", "text", "select",
    "gene", "genes",                 # one gene name / a list of gene names
    "categorical_field",             # the NAME of a categorical column
    "category_value",                # one LEVEL of a categorical column
    "continuous_field",              # the name of a continuous column
    "json",                          # structured tool-specific settings
)
#: What a `genes` parameter's `preset` may name. "top_markers": the form opens
#: with the `preset_n` genes detected in the most cells, the Marker Dot Plot's
#: "Top markers" ranking (`Dataset.top_detected_genes`), and the gates run on
#: the same list. The user edits the list from there.
GENE_PRESETS = ("top_markers",)
# Used with fullmatch: `$` also matches before a trailing newline, and an id or
# a version ends up in directory names, release tags and URLs.
_ID_RE = re.compile(r"[a-z][a-z0-9-]{1,48}[a-z0-9]")
_SEMVER_RE = re.compile(r"\d+\.\d+\.\d+")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Keys schemas 1 and 2 carried and schema 3 dropped.
_REMOVED_KEYS = ("timeout_s", "memory_mb")

#: A SWEEP runs the tool at several values of ONE numeric parameter in a single
#: process, so the viewer can scrub between the steps instead of paying the
#: process start (and the tool's own set-up) per value. The step count is
#: capped because the whole sweep comes back as one reply.
MAX_SWEEP_STEPS = 12
MIN_SWEEP_STEPS = 2
DEFAULT_SWEEP_STEPS = 8
#: Kinds that carry one value per cell (or an arrow field): a sweep would
#: multiply them by the step count, so a tool returning them cannot declare one.
SWEEP_EXCLUDED_OUTPUTS = ("cells", "vectors", "labels")


class ManifestError(ValueError):
    """A manifest the host will not load, and the reason in one sentence."""


def load(plugin_dir: str) -> dict:
    """Read and validate `<plugin_dir>/plugin.json`. Raises ManifestError."""
    path = os.path.join(plugin_dir, "plugin.json")
    if not os.path.isfile(path):
        raise ManifestError("plugin.json is missing.")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except ValueError as exc:                 # bad JSON, or not UTF-8
        raise ManifestError(f"plugin.json is not valid JSON: {exc}")
    return validate(raw, plugin_dir)


def language(manifest: dict) -> str:
    """"r" for a native R tool, else "python". The entry's extension decides."""
    return "r" if manifest["entry"].lower().endswith(".r") else "python"


def _number(value, what: str) -> float:
    """`value` as a finite float, or a ManifestError naming the field.

    `float()` alone raises a bare ValueError on "ten", which the gates would
    blame on themselves rather than on the manifest.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ManifestError(f"{what} must be a number, not {value!r}.") from None
    if not math.isfinite(out):
        raise ManifestError(f"{what} must be a finite number, not {value!r}.")
    return out


def validate(raw: dict, plugin_dir: str = None) -> dict:
    """Validate a manifest dict and return it normalised (defaults filled in)."""
    if not isinstance(raw, dict):
        raise ManifestError("plugin.json must contain a JSON object.")
    schema = raw.get("schema", 1)
    if type(schema) is not int or schema not in SCHEMAS:
        raise ManifestError(f"Unsupported manifest schema {raw.get('schema')!r}; "
                            "this install understands schemas 1, 2 and 3. "
                            "Update PointillViz.")

    pid = str(raw.get("id") or "")
    if not _ID_RE.fullmatch(pid):
        raise ManifestError(
            f"id {pid!r} must be lowercase letters, digits and hyphens, 3-50 "
            "characters, starting with a letter — it is used as a directory "
            "name and as a URL segment.")
    version = str(raw.get("version") or "")
    if not _SEMVER_RE.fullmatch(version):
        raise ManifestError(f"version {version!r} must be MAJOR.MINOR.PATCH.")
    for key in ("name", "description", "author"):
        if not str(raw.get(key) or "").strip():
            raise ManifestError(f"{key} is required and must be non-empty.")

    entry = str(raw.get("entry") or "tool.py")
    # Confined to the plugin directory and to a plain source file: `entry` is
    # joined onto a path the host controls, and "../../etc/x.py" must not be a
    # way to make the host load something outside the bundle. The extension
    # decides the language, and native R tools exist from schema 3 on.
    suffixes = (".py", ".R", ".r") if schema >= 3 else (".py",)
    if os.path.basename(entry) != entry or not entry.endswith(suffixes):
        kinds = "a plain .py or .R filename" if schema >= 3 else "a plain .py filename"
        raise ManifestError(f"entry {entry!r} must be {kinds} inside the plugin "
                            "directory.")
    if plugin_dir and not os.path.isfile(os.path.join(plugin_dir, entry)):
        raise ManifestError(f"entry file {entry!r} does not exist in the bundle.")

    requires = raw.get("requires") or {}
    if not isinstance(requires, dict):
        raise ManifestError("requires must be an object.")
    if schema >= 3:
        removed = [k for k in _REMOVED_KEYS if k in raw]
        removed += ["requires.scipy"] if "scipy" in requires else []
        if removed:
            raise ManifestError(
                f"schema 3 has no {', '.join(removed)}: time and memory are the "
                "author's to manage, and dependencies belong in the tool's "
                "pixi.toml. Remove the key(s).")
    gpu = raw.get("gpu") if schema >= 3 else None
    if gpu is not None and gpu not in GPU_MODES:
        raise ManifestError(f"gpu {gpu!r} must be \"optional\" or \"required\", or "
                            "left out for a tool that never uses a GPU.")

    outputs = raw.get("outputs", [])
    optional_outputs = raw.get("optional_outputs", [])
    if not isinstance(outputs, list) or not isinstance(optional_outputs, list):
        raise ManifestError("outputs and optional_outputs must be lists.")
    allowed = SCHEMA_OUTPUTS[schema]
    bad = [str(o) for o in outputs + optional_outputs if o not in allowed]
    if bad:
        raise ManifestError(f"unknown output kind(s) {', '.join(bad)} for schema "
                            f"{schema}; allowed: " + ", ".join(allowed))
    if not outputs and not optional_outputs:
        raise ManifestError("outputs must name at least one of: " + ", ".join(allowed))
    if len(set(outputs + optional_outputs)) != len(outputs + optional_outputs):
        raise ManifestError("outputs and optional_outputs must not contain duplicate kinds.")
    if schema == 1 and optional_outputs:
        raise ManifestError("optional_outputs requires schema 2 or later.")

    params = [_validate_param(p, i) for i, p in enumerate(raw.get("params") or [])]
    if schema == 1 and any(p["type"] == "json" for p in params):
        raise ManifestError("json parameters require schema 2 or later.")
    swept = [p["name"] for p in params if p.get("sweep")]
    if len(swept) > 1:
        raise ManifestError(f"at most one parameter may be swept; {', '.join(swept)} all "
                            "declare sweep.")
    per_cell = [o for o in outputs + optional_outputs if o in SWEEP_EXCLUDED_OUTPUTS]
    if swept and per_cell:
        raise ManifestError(
            f"{swept[0]} declares a sweep, but the tool returns {', '.join(per_cell)}: a "
            "sweep returns one result per step, and per-cell outputs would multiply the "
            "reply by the number of steps. Sweep a tool whose outputs are matrices, "
            "series, tables, charts or views.")
    seen = set()
    for p in params:
        if p["name"] in seen:
            raise ManifestError(f"duplicate parameter name {p['name']!r}.")
        seen.add(p["name"])

    spatial = str(requires.get("spatial") or "any")
    if spatial not in ("any", "2d", "3d"):
        raise ManifestError("requires.spatial must be 'any', '2d' or '3d'.")

    return {
        "schema": schema,
        "id": pid,
        "name": str(raw["name"]).strip(),
        "version": version,
        "author": str(raw["author"]).strip(),
        "description": str(raw["description"]).strip(),
        "long_description": str(raw.get("long_description") or "").strip(),
        "icon": str(raw.get("icon") or "extension"),
        "tags": [str(t) for t in (raw.get("tags") or [])][:8],
        "entry": entry,
        "homepage": str(raw.get("homepage") or ""),
        "license": str(raw.get("license") or ""),
        "citation": str(raw.get("citation") or ""),
        "gpu": gpu,
        "outputs": outputs,
        "optional_outputs": optional_outputs,
        "params": params,
        "requires": {
            "genes": bool(requires.get("genes", False)),
            "cell_types": bool(requires.get("cell_types", False)),
            "categoricals": bool(requires.get("categoricals", False)),
            "spatial": spatial,
            "min_cells": int(_number(requires.get("min_cells") or 0, "requires.min_cells")),
        },
        # The one parameter the tool may be SWEPT over, or None — see
        # sweep_values(). Old hosts drop the key and the tool still runs.
        "sweep": swept[0] if swept else None,
    }


def _validate_param(p: dict, i: int) -> dict:
    if not isinstance(p, dict):
        raise ManifestError(f"params[{i}] must be an object.")
    name = str(p.get("name") or "")
    if not _NAME_RE.fullmatch(name):
        raise ManifestError(f"params[{i}].name {name!r} must be an identifier "
                            "(letters, digits and _, not starting with a digit) — "
                            "it becomes a key of the params dict or list.")
    ptype = str(p.get("type") or "")
    if ptype not in PARAM_TYPES:
        raise ManifestError(f"params[{i}] ({name}): unknown type {ptype!r}; "
                            "allowed: " + ", ".join(PARAM_TYPES))
    out = {
        "name": name,
        "type": ptype,
        "label": str(p.get("label") or name),
        "help": str(p.get("help") or ""),
        "default": p.get("default"),
        "optional": bool(p.get("optional", False)),
    }
    where = f"params[{i}] ({name})"
    if ptype in ("int", "float"):
        lo, hi = p.get("min"), p.get("max")
        if lo is None or hi is None:
            raise ManifestError(f"{where}: numeric parameters must "
                                "declare min and max, so the viewer can draw a "
                                "bounded control and the host can clamp.")
        out["min"], out["max"] = _number(lo, f"{where} min"), _number(hi, f"{where} max")
        if out["min"] > out["max"]:
            raise ManifestError(f"{where}: min is greater than max.")
        out["step"] = _number(p.get("step") or (1 if ptype == "int" else 0.01),
                              f"{where} step")
    if p.get("sweep"):
        if ptype not in ("int", "float"):
            raise ManifestError(f"{where}: only an int or float parameter "
                                "can be swept.")
        out["sweep"] = True
        steps = int(_number(p.get("sweep_steps") or DEFAULT_SWEEP_STEPS,
                            f"{where} sweep_steps"))
        if not (MIN_SWEEP_STEPS <= steps <= MAX_SWEEP_STEPS):
            raise ManifestError(f"{where}: sweep_steps must be between "
                                f"{MIN_SWEEP_STEPS} and {MAX_SWEEP_STEPS}.")
        out["sweep_steps"] = steps
    if ptype == "select":
        options = p.get("options") or []
        if not options:
            raise ManifestError(f"{where}: select needs options.")
        out["options"] = [{"value": str(o["value"]), "label": str(o.get("label") or o["value"])}
                          if isinstance(o, dict) else {"value": str(o), "label": str(o)}
                          for o in options]
    if ptype == "category_value":
        # Which column the level is drawn from. Naming another parameter keeps
        # the two menus consistent: change the field and the level list follows.
        out["field_param"] = p.get("field_param")
    if ptype == "genes":
        out["max_items"] = int(_number(p.get("max_items") or 20, f"{where} max_items"))
        if p.get("preset") is not None:
            if p["preset"] not in GENE_PRESETS:
                raise ManifestError(f"{where}: preset must be one of "
                                    + ", ".join(GENE_PRESETS) + ".")
            n = int(_number(p.get("preset_n") or out["max_items"], f"{where} preset_n"))
            if not 1 <= n <= out["max_items"]:
                raise ManifestError(f"{where}: preset_n must be from 1 to max_items "
                                    f"({out['max_items']}).")
            out["preset"], out["preset_n"] = p["preset"], n
    return out


# ---------------------------------------------------------------------------
# Coercion — run host-side, before the tool's process starts
# ---------------------------------------------------------------------------
def coerce_params(manifest: dict, supplied: dict) -> dict:
    """Turn the browser's JSON into the exact dict `run()` is promised.

    Missing values fall back to the declared default; numbers are clamped into
    their declared range rather than refused, because a slider that cannot
    produce an out-of-range value means an out-of-range value is a stale client
    and clamping is the recovery a user would want. Anything genuinely
    unanswerable — a required parameter with no default and no value, or a
    number that is not one — raises ManifestError.
    """
    supplied = supplied or {}
    out = {}
    for p in manifest["params"]:
        name, ptype = p["name"], p["type"]
        val = supplied.get(name, None)
        if val is None or val == "":
            val = p.get("default")
        if val is None or val == "":
            if p["optional"]:
                out[name] = None
                continue
            raise ManifestError(f"{p['label']} is required.")
        if ptype == "int":
            out[name] = int(max(p["min"], min(p["max"], _number(val, p["label"]))))
        elif ptype == "float":
            out[name] = float(max(p["min"], min(p["max"], _number(val, p["label"]))))
        elif ptype == "bool":
            out[name] = val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
        elif ptype == "select":
            allowed = {o["value"] for o in p["options"]}
            if str(val) not in allowed:
                raise ManifestError(f"{p['label']}: {val!r} is not one of "
                                    + ", ".join(sorted(allowed)))
            out[name] = str(val)
        elif ptype == "genes":
            items = val if isinstance(val, list) else [val]
            out[name] = [str(g) for g in items][:p["max_items"]]
        elif ptype == "json":
            try:
                # Accept JSON objects/arrays only: a string containing malformed
                # JSON must not silently become a valid tool setting.
                value = json.loads(val) if isinstance(val, str) else val
                encoded = json.dumps(value, allow_nan=False)
                if not isinstance(value, (dict, list)):
                    raise ValueError("expected an object or array")
                if len(encoded.encode("utf-8")) > MAX_JSON_PARAM_BYTES:
                    raise ValueError("exceeds the 64 KiB settings limit")
                out[name] = json.loads(encoded)
            except (ValueError, TypeError, RecursionError) as exc:
                raise ManifestError(f"{p['label']}: invalid JSON settings ({exc}).") from exc
        else:
            out[name] = str(val)
    return out


def sweep_values(manifest: dict, lo=None, hi=None, steps=None) -> tuple:
    """(parameter spec, values) for a sweep of the manifest's swept parameter.

    `steps` values from `lo` to `hi` (default: the parameter's declared range),
    evenly spaced, each clamped to the declared range and snapped to the
    declared step exactly as the slider would hold it, duplicates removed. So
    every value a sweep runs at is one the form could have submitted — and the
    same value the single run at that setting would have used.
    """
    name = manifest.get("sweep")
    spec = next((p for p in manifest["params"] if p["name"] == name), None)
    if spec is None:
        raise ManifestError("this tool declares no parameter that can be swept.")
    pmin, pmax, step = spec["min"], spec["max"], spec["step"]
    try:
        a = pmin if lo is None else float(lo)
        b = pmax if hi is None else float(hi)
        n = int(spec.get("sweep_steps") or DEFAULT_SWEEP_STEPS) if steps is None else int(steps)
    except (TypeError, ValueError):
        raise ManifestError("a sweep needs numeric from / to / steps.")
    if not (MIN_SWEEP_STEPS <= n <= MAX_SWEEP_STEPS):
        raise ManifestError(f"a sweep runs {MIN_SWEEP_STEPS} to {MAX_SWEEP_STEPS} steps, not {n}.")
    a, b = max(pmin, min(pmax, a)), max(pmin, min(pmax, b))
    if a > b:
        a, b = b, a
    values = []
    for i in range(n):
        v = a + (b - a) * i / (n - 1)
        v = pmin + round((v - pmin) / step) * step if step > 0 else v
        v = max(pmin, min(pmax, v))
        v = int(round(v)) if spec["type"] == "int" else float(f"{v:.10g}")
        if v not in values:
            values.append(v)
    if len(values) < MIN_SWEEP_STEPS:
        raise ManifestError(f"{spec['label']}: from {a:g} to {b:g} holds fewer than "
                            f"{MIN_SWEEP_STEPS} distinct values at step {step:g}.")
    return spec, values


def check_requires(manifest: dict, meta: dict) -> str:
    """"" if the tool can run on this dataset, else the reason it cannot.

    A string rather than a bool because the viewer shows it: "needs cell-type
    annotation" is a usable message and a greyed-out button is not.
    """
    req = manifest["requires"]
    if req["genes"] and not ((meta.get("genes") or {}).get("n_genes")):
        return "needs a gene expression matrix, which this dataset does not have."
    if req["cell_types"] and not meta.get("cell_types"):
        return "needs cell-type annotation, which this dataset does not have."
    if req["categoricals"] and not (meta.get("cell_types") or meta.get("extra_categorical")):
        return "needs at least one categorical annotation."
    mode = (meta.get("spatial") or {}).get("mode")
    if req["spatial"] == "2d" and mode != "2d":
        return "only applies to 2D datasets."
    if req["spatial"] == "3d" and mode != "3d":
        return "only applies to 3D datasets."
    if req["min_cells"] and int(meta.get("n_cells") or 0) < req["min_cells"]:
        return f"needs at least {req['min_cells']:,} cells; this dataset has " \
               f"{int(meta.get('n_cells') or 0):,}."
    return ""
