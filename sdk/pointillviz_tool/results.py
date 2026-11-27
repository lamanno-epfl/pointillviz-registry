"""What a tool hands back: the output vocabulary, its helpers, its one validator.

`run(ds, params)` returns a dict whose keys are output kinds:

  summary     Markdown text
  cells       one number per cell, painted on the map
  labels      one category per cell, painted on the map with a legend
  vectors     an arrow field over the tissue
  table, series, matrix, radar, chord     tables and charts
  views       self-contained HTML documents, shown in an opaque-origin iframe
  files       downloads
  images      static figures, PNG or SVG

The viewer draws each kind with a renderer of its own, so the shape of every
kind is a contract, and `check_result` is where it is enforced. The runner
calls it on every return value before anything is printed and the gates call
the same function, so a non-conforming tool fails in its own process with a
sentence rather than reaching the browser as a malformed payload.

Every per-cell kind follows the convention of `pointillviz_tool.dataset`: NaN
(or None) means "no value here" and survives as JSON null, while an infinity is
a bug in the tool and is reported as one. One kind breaks the per-cell shape,
and knowingly: `vectors` is an arrow field, and an arrow is a measurement AT A
PLACE that is not a cell, so it carries its own anchor coordinates in the
dataset's world units — see `_check_vectors`.

Result payloads travel as JSON over a pipe and then over HTTP, and a tool that
returns a 10 M-row table would take the browser down with it. The ceilings
below are generous for anything a person can read and firm enough that a
runaway tool fails with a sentence instead of a hung tab. HTML views, files
and images are open-ended, so they share one byte budget, which bounds the
pipe, the HTTP reply and the tab together.
"""

from __future__ import annotations

import base64
import binascii
import io
import re

import numpy as np

from .dataset import PluginError

#: The kinds every manifest schema may declare: schema 1's whole vocabulary,
#: each drawn by a renderer of the viewer's own.
NATIVE_OUTPUTS = ("summary", "cells", "vectors", "table", "series", "matrix",
                  "radar", "chord")
#: Every kind a result may hold. Schema 2 added views and files, schema 3
#: labels and images (`pointillviz_tool.manifest` says which schema may
#: declare which).
OUTPUT_KINDS = NATIVE_OUTPUTS + ("views", "files", "labels", "images")

#: The PluginError codes `check_result` raises: the tool RAN and its return
#: value is wrong, as opposed to the tool refusing an input. The gates
#: attribute them to the `result` gate, and a sweep stops on them (a refusal at
#: one value is that step's answer; a broken return value is not).
RESULT_CODES = frozenset({"bad_result", "bad_length", "non_finite",
                          "too_large", "empty_result"})

MAX_TABLE_ROWS = 5000
MAX_SERIES_POINTS = 5000
MAX_MATRIX_CELLS = 10000
MAX_SUMMARY_CHARS = 20000
#: Log lines one run may send; the rest is cut with a note.
MAX_LOG_LINES = 200
#: Arrows in one `vectors` field. Five arrays of this length travel as JSON, so
#: the ceiling is a payload budget as much as a legibility one — but it is set
#: by the drawing, not by the wire: past roughly twenty thousand arrows a
#: section is solid ink and the overlay stops being readable at any zoom. The
#: viewer's own lattice sizing aims two orders below it; a tool that reaches
#: this number is emitting arrows nobody can tell apart.
MAX_VECTORS = 20000
#: Categories in one `labels` map, and characters in one category name: a
#: legend longer or wider than this is no longer something a person reads.
MAX_CATEGORIES = 256
MAX_CATEGORY_CHARS = 80
#: The budget views, files and images share, counted in wire bytes.
MAX_EXTENSION_BYTES = 8 * 1024 * 1024
MAX_VIEWS = 8
MAX_FILES = 16
MAX_IMAGES = 8

#: The eight bytes every PNG file starts with.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def output_mismatch(manifest, payload):
    """(missing, extra): declared outputs absent from a payload, and kinds
    present that the manifest never declared.

    The output contract the runner enforces for schema 2 and 3 tools: every
    kind in `outputs` is returned, and nothing outside `outputs` and
    `optional_outputs` is.
    """
    required = manifest["outputs"]
    allowed = required + manifest.get("optional_outputs", [])
    seen = [key for key in payload if payload[key] is not None]
    return ([key for key in required if key not in seen],
            [key for key in seen if key not in allowed])


# ===========================================================================
# Helpers a tool builds results with
# ===========================================================================
def html_view(html, *, id="view", title="Interactive view", height=480):
    """Build a self-contained HTML/CSS/JS view; embed assets, do not use CDNs."""
    return {"id": id, "title": title, "html": html, "height": height}


def download_file(name, data, *, media_type="application/octet-stream"):
    """Build a download from bytes or UTF-8 text (CSV, SVG, PDF, PNG, etc.)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return {"name": name, "media_type": media_type,
            "base64": base64.b64encode(data).decode("ascii")}


def image(data, title):
    """Build one `images` entry from PNG bytes, SVG text or a matplotlib Figure.

    Bytes are taken as PNG and must start with the PNG signature; a `str` is
    SVG markup (the viewer shows it as an image, never inlined into its page);
    a matplotlib Figure is saved as PNG at its own size and dpi. From any other
    plotting library, save a PNG or an SVG and pass its bytes or its text.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
        if not data.startswith(PNG_SIGNATURE):
            raise PluginError(
                "bad_result",
                f"image {title!r}: these bytes are not a PNG file. Pass PNG bytes, "
                "SVG text (a str) or a matplotlib Figure.")
        return {"title": title, "png": base64.b64encode(data).decode("ascii")}
    if isinstance(data, str):
        return {"title": title, "svg": data}
    try:
        from matplotlib.figure import Figure
    except ImportError:
        Figure = None
    if Figure is not None and isinstance(data, Figure):
        buf = io.BytesIO()
        data.savefig(buf, format="png")
        return {"title": title, "png": base64.b64encode(buf.getvalue()).decode("ascii")}
    raise PluginError(
        "bad_result",
        f"image {title!r}: expected PNG bytes, SVG text (a str) or a matplotlib "
        f"Figure, not {type(data).__name__}.")


# ===========================================================================
# Validation — one implementation, used by the runner AND the gates
# ===========================================================================
def _fail(message: str, code: str = "bad_result"):
    raise PluginError(code, message)


def _finite_list(values, what: str, cap: int) -> list:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size > cap:
        raise PluginError("too_large", f"{what} has {arr.size} values; the limit is {cap}.")
    # NaN and +/-inf are not JSON. NaN is the documented "no value" marker so
    # it becomes null; an infinity is a bug in the tool and is reported as one
    # rather than silently rendered as the top of a colour ramp.
    if np.isinf(arr).any():
        raise PluginError("non_finite",
                          f"{what} contains infinities. Use NaN for missing values.")
    return [None if np.isnan(v) else float(v) for v in arr]


def check_result(result, n_cells: int) -> dict:
    """Validate and normalise a tool's return value; raise PluginError if not.

    Returns a JSON-ready dict. Every key is optional and at least one is
    required — a tool that computes nothing renderable has not run.
    """
    if not isinstance(result, dict):
        raise PluginError("bad_result",
                          f"run() must return a dict, got {type(result).__name__}.")
    unknown = [str(k) for k in result if k not in OUTPUT_KINDS]
    if unknown:
        raise PluginError("bad_result",
                          f"Unknown result key(s): {', '.join(sorted(unknown))}. "
                          f"Allowed: {', '.join(OUTPUT_KINDS)}.")
    if not any(k in result and result[k] is not None for k in OUTPUT_KINDS):
        raise PluginError("empty_result",
                          "run() returned nothing to show. Provide at least one of "
                          + ", ".join(OUTPUT_KINDS) + ".")
    out = {}

    if result.get("summary") is not None:
        text = str(result["summary"])
        if len(text) > MAX_SUMMARY_CHARS:
            raise PluginError("too_large", f"summary is {len(text)} characters; "
                                           f"the limit is {MAX_SUMMARY_CHARS}.")
        out["summary"] = text

    cells = result.get("cells")
    if cells is not None:
        if not isinstance(cells, dict) or cells.get("values") is None:
            raise PluginError("bad_result", "cells must be a dict with a 'values' key.")
        arr = np.asarray(cells.get("values"), dtype=np.float64).ravel()
        if arr.size != n_cells:
            raise PluginError(
                "bad_length",
                f"cells.values has {arr.size} entries but the dataset has {n_cells} "
                "cells. Return one value per cell, in dataset order, with NaN where "
                "the tool has no answer.")
        out["cells"] = {
            "values": _finite_list(arr, "cells.values", n_cells),
            "label": str(cells.get("label") or "value"),
            "colormap": str(cells.get("colormap") or "viridis"),
            "diverging": bool(cells.get("diverging", False)),
            "domain": ([float(cells["domain"][0]), float(cells["domain"][1])]
                       if cells.get("domain") else None),
            "units": str(cells.get("units") or ""),
        }

    labels = result.get("labels")
    if labels is not None:
        out["labels"] = _check_labels(labels, n_cells)

    vectors = result.get("vectors")
    if vectors is not None:
        out["vectors"] = _check_vectors(vectors)

    table = result.get("table")
    if table is not None:
        cols = [str(c) for c in (table.get("columns") or [])]
        rows = table.get("rows") or []
        if not cols:
            raise PluginError("bad_result", "table needs a non-empty 'columns' list.")
        if len(rows) > MAX_TABLE_ROWS:
            raise PluginError("too_large", f"table has {len(rows)} rows; "
                                           f"the limit is {MAX_TABLE_ROWS}.")
        norm = []
        for r in rows:
            if len(r) != len(cols):
                raise PluginError("bad_result",
                                  f"a table row has {len(r)} cells but there are "
                                  f"{len(cols)} columns.")
            norm.append([_cell_value(v) for v in r])
        out["table"] = {"title": str(table.get("title") or ""),
                        "columns": cols, "rows": norm,
                        "sort_by": table.get("sort_by")}

    series = result.get("series")
    if series is not None:
        lines = []
        for ln in (series.get("lines") or []):
            x = np.asarray(ln.get("x"), dtype=np.float64).ravel()
            y = np.asarray(ln.get("y"), dtype=np.float64).ravel()
            if x.size != y.size:
                raise PluginError("bad_result",
                                  f"series line {ln.get('label')!r}: x has {x.size} "
                                  f"points, y has {y.size}.")
            lines.append({"label": str(ln.get("label") or ""),
                          "x": _finite_list(x, "series.x", MAX_SERIES_POINTS),
                          "y": _finite_list(y, "series.y", MAX_SERIES_POINTS),
                          "color": ln.get("color"),
                          "dashed": bool(ln.get("dashed", False)),
                          "band": bool(ln.get("band", False))})
        if not lines:
            raise PluginError("bad_result", "series needs at least one line.")
        out["series"] = {"title": str(series.get("title") or ""),
                         "x_label": str(series.get("x_label") or ""),
                         "y_label": str(series.get("y_label") or ""),
                         "lines": lines,
                         "rules": [float(v) for v in (series.get("rules") or [])],
                         "log_x": bool(series.get("log_x", False))}

    matrix = result.get("matrix")
    if matrix is not None:
        r_lab = [str(v) for v in (matrix.get("rows") or [])]
        c_lab = [str(v) for v in (matrix.get("cols") or [])]
        vals = np.asarray(matrix.get("values"), dtype=np.float64)
        if vals.ndim != 2 or vals.shape != (len(r_lab), len(c_lab)):
            raise PluginError("bad_result",
                              f"matrix.values is {vals.shape} but the labels say "
                              f"({len(r_lab)}, {len(c_lab)}).")
        if vals.size > MAX_MATRIX_CELLS:
            raise PluginError("too_large", f"matrix has {vals.size} cells; "
                                           f"the limit is {MAX_MATRIX_CELLS}.")
        out["matrix"] = {"title": str(matrix.get("title") or ""),
                         "rows": r_lab, "cols": c_lab,
                         "values": [_finite_list(row, "matrix.values", MAX_MATRIX_CELLS)
                                    for row in vals],
                         "diverging": bool(matrix.get("diverging", False)),
                         "colormap": str(matrix.get("colormap") or "viridis")}

    for kind in ("radar", "chord"):
        if result.get(kind) is not None:
            out[kind] = _check_radial(kind, result[kind])
    out.update(_check_extensions(result))
    return out


def _check_labels(labels, n_cells: int) -> dict:
    """Validate and normalise a `labels` payload: one category per cell.

    The categorical twin of `cells`, for a tool whose answer is a partition —
    niches, domains, clusters — rather than a number. `values[i]` is the index
    of cell *i*'s category in `categories`, or None (NaN in an array) for a
    cell the tool had no label for, which the viewer leaves unpainted. Codes
    are 0-based and must index a category: a negative code is not "missing"
    here, because a silently dropped cell and a mistyped code look the same.
    `colors` is optional; without it the viewer picks a palette.
    """
    if not isinstance(labels, dict) or labels.get("values") is None:
        _fail("labels must be a dict with 'values' and 'categories' keys.")
    cats = labels.get("categories")
    if cats is None or isinstance(cats, (str, bytes, dict)):
        _fail("labels.categories must be a list of category names.")
    try:
        categories = [str(c) for c in cats]
    except TypeError:
        _fail("labels.categories must be a list of category names.")
    if not categories:
        _fail("labels.categories must name at least one category.")
    if len(categories) > MAX_CATEGORIES:
        _fail(f"labels has {len(categories)} categories; the limit is "
              f"{MAX_CATEGORIES}.", "too_large")
    for name in categories:
        if len(name) > MAX_CATEGORY_CHARS:
            _fail(f"labels.categories: {name[:40]!r}... is {len(name)} characters; "
                  f"the limit is {MAX_CATEGORY_CHARS}.")
    if len(set(categories)) != len(categories):
        _fail("labels.categories must not repeat a name.")

    try:
        arr = np.asarray(labels["values"], dtype=np.float64).ravel()
    except (TypeError, ValueError):
        _fail("labels.values must hold integer category codes, with None where a "
              "cell has no label.")
    if arr.size != n_cells:
        _fail(f"labels.values has {arr.size} entries but the dataset has {n_cells} "
              "cells. Return one category code per cell, in dataset order, with "
              "None where the tool has no label.", "bad_length")
    if np.isinf(arr).any():
        _fail("labels.values contains infinities. Use None (or NaN) for a cell "
              "with no label.", "non_finite")
    known = arr[~np.isnan(arr)]
    bad = (known != np.floor(known)) | (known < 0) | (known >= len(categories))
    if bad.any():
        _fail(f"labels.values holds {known[bad][0]:g}, but each value must be a "
              f"whole number from 0 to {len(categories) - 1} (an index into "
              "categories), or None (NaN) for a cell with no label.")

    colors = labels.get("colors")
    if colors is not None:
        if isinstance(colors, (str, dict)):
            _fail("labels.colors must be a list with one #rrggbb colour per category.")
        colors = list(colors)
        if (len(colors) != len(categories)
                or not all(isinstance(c, str) and _HEX_COLOR.fullmatch(c) for c in colors)):
            _fail(f"labels.colors must hold one #rrggbb colour per category "
                  f"({len(categories)}), or be left out.")
    return {
        "values": [None if v != v else int(v) for v in arr.tolist()],
        "categories": categories,
        "colors": colors,
        "label": str(labels.get("label") or "label"),
    }


def _check_vectors(vectors) -> dict:
    """Validate and normalise a `vectors` payload — an arrow field over the tissue.

    The only kind that carries COORDINATES. Every other kind is either per-cell
    (the viewer already knows where cell *i* is) or coordinate-free; a vector
    field is neither, because an arrow is a measurement AT a place that is not
    a cell. So the tool hands back anchors in the dataset's own world units,
    and the viewer draws them there.

    Anchors and components, never endpoints. `(u, v)` is the quantity — a
    gradient, a flow, a displacement — in whatever units the tool names; the
    arrow's drawn LENGTH is presentation, and presentation is the viewer's
    half of the contract, exactly as a `cells` array declares values and lets
    the viewer choose the ramp. A tool that shipped endpoints would be
    choosing a length in data units for a canvas it cannot see, and two tools
    would then disagree about what a long arrow means.

    `pitch` is the data-space distance between neighbouring anchors, and it is
    a fact about the sampling rather than about the drawing: it is what lets
    the viewer scale the longest arrow to just under one lattice step, so
    arrows never overlap into an unreadable mat. Omit it — a tool whose
    anchors are not on a lattice has no pitch to state — and the viewer
    derives one from the anchors' own extent and count.

    NaN follows the convention the rest of this package rests on: an anchor
    with no finite value in any of x/y/u/v is one the tool had no answer for,
    and the viewer drops it rather than drawing an arrow of length zero at the
    origin. An infinity is a bug and is reported as one.
    """
    if not isinstance(vectors, dict):
        raise PluginError("bad_result", "vectors must be a dict.")
    parts = {}
    for key in ("x", "y", "u", "v"):
        if vectors.get(key) is None:
            raise PluginError("bad_result",
                              f"vectors needs an {key!r} array of anchor "
                              "coordinates and components.")
        parts[key] = np.asarray(vectors[key], dtype=np.float64).ravel()
    n = parts["x"].size
    for key, arr in parts.items():
        if arr.size != n:
            raise PluginError(
                "bad_result",
                f"vectors.{key} has {arr.size} entries but vectors.x has {n}. "
                "x, y, u and v are one array each, of the same length: one "
                "arrow per index.")
    if n == 0:
        raise PluginError("bad_result",
                          "vectors is empty. Return no `vectors` key at all "
                          "rather than an arrow field with no arrows.")
    if n > MAX_VECTORS:
        raise PluginError("too_large",
                          f"vectors has {n} arrows; the limit is {MAX_VECTORS}. "
                          "Thin the lattice — an overlay this dense is solid ink.")

    value = vectors.get("value")
    if value is not None:
        value = np.asarray(value, dtype=np.float64).ravel()
        if value.size != n:
            raise PluginError("bad_result",
                              f"vectors.value has {value.size} entries but "
                              f"there are {n} arrows.")

    pitch = vectors.get("pitch")
    if pitch is not None:
        pitch = float(pitch)
        if not np.isfinite(pitch) or pitch <= 0:
            raise PluginError("bad_result",
                              f"vectors.pitch is {pitch}; it is the spacing "
                              "between neighbouring arrows and must be a "
                              "positive number, or absent.")

    return {
        "title": str(vectors.get("title") or ""),
        "label": str(vectors.get("label") or "vector"),
        "x": _finite_list(parts["x"], "vectors.x", MAX_VECTORS),
        "y": _finite_list(parts["y"], "vectors.y", MAX_VECTORS),
        "u": _finite_list(parts["u"], "vectors.u", MAX_VECTORS),
        "v": _finite_list(parts["v"], "vectors.v", MAX_VECTORS),
        "value": (None if value is None
                  else _finite_list(value, "vectors.value", MAX_VECTORS)),
        "units": str(vectors.get("units") or ""),
        "pitch": pitch,
        "colormap": str(vectors.get("colormap") or "viridis"),
        "diverging": bool(vectors.get("diverging", False)),
        "domain": ([float(vectors["domain"][0]), float(vectors["domain"][1])]
                   if vectors.get("domain") else None),
    }


def _cell_value(v):
    """One table cell: a string, or a JSON-safe number, or None."""
    if v is None:
        return None
    if isinstance(v, str):
        return v[:200]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)[:200]
    if np.isnan(f):
        return None
    if np.isinf(f):
        raise PluginError("non_finite", "a table cell is infinite.")
    return f


def _check_radial(kind, payload):
    """Validate a radar or chord payload: bounded and coordinate-free."""
    def fail(message):
        raise PluginError('bad_result', f'{kind}: {message}')

    def color(value):
        if value is not None and (not isinstance(value, str) or not _HEX_COLOR.fullmatch(value)):
            fail('colors must be six-digit hex colors.')
        return value

    def numbers(value, name, limit):
        try:
            a = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            fail(f'{name} must contain numbers.')
        if a.ndim != 1:
            fail(f'{name} must be a one-dimensional array.')
        return _finite_list(a, f'{kind}.{name}', limit)

    if not isinstance(payload, dict):
        fail('output must be an object.')
    out = {k: str(payload.get(k) or '')[:20000] for k in ('title', 'units', 'note')}
    if kind == 'radar':
        axes, profiles = payload.get('axes'), payload.get('profiles')
        if not isinstance(axes, list) or not 3 <= len(axes) <= 100:
            fail('axes must have 3–100 labels.')
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= 100:
            fail('profiles must have 1–100 entries.')
        domain = numbers(payload.get('domain', [0, 1]), 'domain', 2)
        if len(domain) != 2 or None in domain or domain[0] >= domain[1]:
            fail('domain must be two finite increasing bounds.')
        out.update(axes=[str(a) for a in axes], domain=domain, profiles=[])
        for p in profiles:
            if not isinstance(p, dict):
                fail('each profile must be an object.')
            values = numbers(p.get('values'), 'values', 100)
            if len(values) != len(axes):
                fail('each profile must have one value per axis.')
            if any(v is not None and not domain[0] <= v <= domain[1] for v in values):
                fail('values must lie inside the declared domain.')
            out['profiles'].append({'label': str(p.get('label') or ''), 'color': color(p.get('color')), 'values': values})
        layout = payload.get('layout', 'multiples')
        if layout not in ('overlay', 'multiples'):
            fail('layout must be overlay or multiples.')
        out['layout'] = layout
    else:
        nodes, links = payload.get('nodes'), payload.get('links')
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 100:
            fail('nodes must have 1–100 entries.')
        if not isinstance(links, list) or len(links) > 2000:
            fail('links must be an array of at most 2,000 entries.')
        out.update(nodes=[], links=[], directed=bool(payload.get('directed', False)))
        for node in nodes:
            if not isinstance(node, dict):
                fail('each node must be an object.')
            out['nodes'].append({'label': str(node.get('label') or ''), 'color': color(node.get('color')), 'group': str(node.get('group') or '')})
        for link in links:
            if not isinstance(link, dict):
                fail('each link must be an object.')
            a, b = link.get('source'), link.get('target')
            if any(type(i) is not int or not 0 <= i < len(nodes) for i in (a, b)):
                fail('link endpoints must be valid integer node indices.')
            values = numbers([link.get('value'), link.get('score')], 'link', 2)
            if values[0] is None or values[0] < 0:
                fail('link values must be finite and nonnegative.')
            out['links'].append({'source': a, 'target': b, 'value': values[0], 'score': values[1], 'color': color(link.get('color'))})
    return out


def _check_extensions(result):
    """Views, files and images: open-ended envelopes, never host-side renderers.

    HTML is interpreted only in an opaque-origin iframe; files and PNGs are
    base64 bytes, never server paths; SVG is shown as an image, never inlined.
    The three share one byte budget, counted on the wire before anything is
    decoded, so a runaway payload is refused before it is allocated twice.
    """
    out = {}
    budget = 0

    def spend(n_bytes):
        nonlocal budget
        budget += n_bytes
        if budget > MAX_EXTENSION_BYTES:
            _fail("views, files and images together exceed the 8 MiB output budget.",
                  "too_large")

    for kind, limit in (("views", MAX_VIEWS), ("files", MAX_FILES),
                        ("images", MAX_IMAGES)):
        items = result.get(kind)
        if items is None:
            continue
        if not isinstance(items, list) or not items:
            _fail(f"{kind} must be a non-empty list; omit an optional output when absent.")
        if len(items) > limit:
            _fail(f"{kind} allows at most {limit} items.", "too_large")
        normalized, seen = [], set()
        for item in items:
            if not isinstance(item, dict):
                _fail(f"Each {kind} item must be an object.")
            if kind == "images":
                normalized.append(_check_image(item, spend))
                continue
            key = item.get("id" if kind == "views" else "name")
            if not isinstance(key, str) or not key or len(key) > 160:
                _fail(f"{kind} needs a non-empty id/name of at most 160 characters.")
            if key in seen:
                _fail(f"Duplicate {kind} id/name: {key}.")
            seen.add(key)
            if kind == "views":
                html = item.get("html")
                if not isinstance(html, str) or not html.strip():
                    _fail("views.html must be non-empty HTML text.")
                height = item.get("height", 480)
                if type(height) is not int or not 160 <= height <= 1600:
                    _fail("views.height must be an integer from 160 to 1600 pixels.")
                title = item.get("title", key)
                if not isinstance(title, str) or not title.strip() or len(title) > 200:
                    _fail("views.title must be non-empty text, at most 200 characters.")
                spend(len(html.encode("utf-8")))
                normalized.append({"id": key, "title": title, "height": height, "html": html})
            else:
                if key in (".", "..") or any(c in key for c in '/\\:') or any(ord(c) < 32 for c in key):
                    _fail("files.name must be a plain filename, not a path.")
                media = item.get("media_type", "application/octet-stream")
                if not isinstance(media, str) or not re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", media):
                    _fail("files.media_type must be a MIME type without parameters.")
                encoded = item.get("base64")
                if not isinstance(encoded, str):
                    _fail("files.base64 must be a base64 string; server file paths are not accepted.")
                spend(len(encoded))
                try:
                    data = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error):
                    _fail("files.base64 is not valid base64.")
                normalized.append({"name": key, "media_type": media,
                                   "base64": encoded, "size": len(data)})
        out[kind] = normalized
    return out


def _check_image(item: dict, spend) -> dict:
    """One `images` entry: a title and exactly one of png (base64) or svg (text)."""
    title = item.get("title")
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        _fail("images.title must be non-empty text, at most 200 characters.")
    formats = [f for f in ("png", "svg") if item.get(f) is not None]
    if len(formats) != 1:
        _fail(f"image {title!r} needs exactly one of png or svg.")
    if formats[0] == "png":
        encoded = item["png"]
        if not isinstance(encoded, str):
            _fail(f"image {title!r}: png must be base64 text; image(png_bytes, title) "
                  "builds it.")
        spend(len(encoded))
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            _fail(f"image {title!r}: png is not valid base64.")
        if not data.startswith(PNG_SIGNATURE):
            _fail(f"image {title!r}: png does not hold a PNG file.")
        return {"title": title, "png": encoded}
    svg = item["svg"]
    if not isinstance(svg, str) or "<svg" not in svg:
        _fail(f"image {title!r}: svg must be SVG markup, the text of an <svg> "
              "element, not a file name.")
    spend(len(svg.encode("utf-8")))
    return {"title": title, "svg": svg}
