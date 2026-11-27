"""check_result: what the viewer will and will not be handed.

A result that would break the viewer is refused HERE, in a sentence, rather
than reaching the browser as a malformed payload. Every refusal names the
offending part and, where there is one, the limit.
"""

import base64
import json
import sys
import types

import numpy as np
import pytest

from pointillviz_tool import PluginError, check_result, download_file, html_view, image
from pointillviz_tool.results import (
    MAX_CATEGORIES, MAX_EXTENSION_BYTES, MAX_IMAGES, MAX_TABLE_ROWS, MAX_VECTORS,
    NATIVE_OUTPUTS, OUTPUT_KINDS, PNG_SIGNATURE, RESULT_CODES, output_mismatch,
)

PNG = PNG_SIGNATURE + b"\x00\x00\x00\rIHDR" + bytes(17)   # a signature is all we check


def refusal(result, n_cells=10):
    with pytest.raises(PluginError) as exc:
        check_result(result, n_cells)
    assert exc.value.code in RESULT_CODES, \
        f"{exc.value.code!r} is not a result code; the gates would blame the wrong gate"
    return exc.value


def test_the_vocabulary():
    assert OUTPUT_KINDS[:len(NATIVE_OUTPUTS)] == NATIVE_OUTPUTS
    assert set(OUTPUT_KINDS) - set(NATIVE_OUTPUTS) == {"views", "files", "labels", "images"}


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------
def test_a_result_must_say_something():
    assert refusal({}).code == "empty_result"
    assert refusal({"summary": None}).code == "empty_result"
    assert refusal([1, 2, 3]).code == "bad_result"
    exc = refusal({"cellz": 1})
    assert exc.code == "bad_result"
    assert "cellz" in str(exc) and "summary" in str(exc) and "labels" in str(exc), \
        "the message must name the bad key AND list the allowed ones"


def test_a_whole_result_round_trips_through_json():
    res = check_result({
        "summary": "**hi**",
        "cells": {"values": np.linspace(0, 1, 20)},
        "labels": {"values": np.arange(20) % 2, "categories": ["a", "b"]},
        "vectors": {"x": np.arange(4), "y": np.arange(4), "u": np.ones(4),
                    "v": np.zeros(4), "pitch": 2.0},
        "table": {"columns": ["t", "n"], "rows": [["a", 1]]},
        "series": {"lines": [{"label": "L", "x": np.arange(5), "y": np.arange(5) * 1.0}],
                   "rules": [0.0]},
        "matrix": {"rows": ["a", "b"], "cols": ["x"], "values": np.zeros((2, 1))},
        "images": [image(PNG, "figure")],
    }, 20)
    json.dumps(res, allow_nan=False)
    assert set(res) == {"summary", "cells", "labels", "vectors", "table", "series",
                        "matrix", "images"}
    assert check_result(res, 20) == res, "a checked result checks again unchanged"


def test_output_mismatch_names_both_directions():
    manifest = {"outputs": ["table", "labels"], "optional_outputs": ["files"]}
    assert output_mismatch(manifest, {"table": {}, "labels": {}, "files": None}) == ([], [])
    assert output_mismatch(manifest, {"table": {}, "images": []}) == (["labels"], ["images"])


# ---------------------------------------------------------------------------
# cells, table, series, matrix
# ---------------------------------------------------------------------------
def test_cells_length_is_the_contract():
    exc = refusal({"cells": {"values": [1.0] * 9}})
    assert exc.code == "bad_length"
    assert "9" in str(exc) and "10" in str(exc) and "NaN" in str(exc), \
        "the message must state both counts and point at the NaN convention"
    assert len(check_result({"cells": {"values": [1.0] * 10}}, 10)["cells"]["values"]) == 10


def test_nan_is_null_and_infinity_is_a_bug():
    out = check_result({"cells": {"values": [1.0, float("nan"), 3.0]}}, 3)
    assert out["cells"]["values"] == [1.0, None, 3.0], \
        "NaN is the documented 'no value' marker and must survive as JSON null"
    exc = refusal({"cells": {"values": [1.0, float("inf"), 3.0]}}, 3)
    assert exc.code == "non_finite" and "NaN" in str(exc)


def test_cells_carries_its_presentation_with_defaults():
    out = check_result({"cells": {"values": [0.0] * 4}}, 4)["cells"]
    assert out["colormap"] == "viridis" and out["diverging"] is False and out["domain"] is None
    out2 = check_result({"cells": {"values": [0.0] * 4, "colormap": "magma",
                                   "diverging": True, "domain": [-1, 1],
                                   "label": "z", "units": "sd"}}, 4)["cells"]
    assert out2["domain"] == [-1.0, 1.0] and out2["label"] == "z" and out2["units"] == "sd"


def test_shapes_are_checked_not_trusted():
    refusal({"table": {"columns": [], "rows": []}}, 4)
    exc = refusal({"table": {"columns": ["a", "b"], "rows": [[1, 2], [3]]}}, 4)
    assert "1 cells" in str(exc) and "2 columns" in str(exc)
    refusal({"series": {"lines": [{"label": "l", "x": [1, 2], "y": [1]}]}}, 4)
    refusal({"series": {"lines": []}}, 4)
    refusal({"matrix": {"rows": ["a"], "cols": ["x", "y"], "values": [[1.0]]}}, 4)


def test_ceilings_are_enforced():
    exc = refusal({"table": {"columns": ["a"], "rows": [[1]] * (MAX_TABLE_ROWS + 1)}}, 4)
    assert exc.code == "too_large" and str(MAX_TABLE_ROWS) in str(exc)
    assert refusal({"summary": "x" * 20001}, 4).code == "too_large"


def test_table_cells_are_json_safe():
    out = check_result({"table": {"columns": ["a", "b", "c"],
                                  "rows": [["s", np.float32(1.5), float("nan")],
                                           [None, np.int64(3), True]]}}, 4)["table"]
    assert out["rows"][0] == ["s", 1.5, None]
    assert out["rows"][1] == [None, 3.0, True]
    json.dumps(out)


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------
def test_vectors_arrays_must_be_parallel():
    ok = check_result({"vectors": {"x": [0, 1], "y": [0, 1], "u": [1, 1], "v": [0, 1]}}, 10)
    assert len(ok["vectors"]["x"]) == 2 and ok["vectors"]["value"] is None
    exc = refusal({"vectors": {"x": [0, 1], "y": [0], "u": [1, 1], "v": [0, 1]}})
    assert "1" in str(exc) and "2" in str(exc), "the message must state both counts"
    for missing in ("x", "y", "u", "v"):
        payload = {"x": [0.0], "y": [0.0], "u": [1.0], "v": [1.0]}
        del payload[missing]
        refusal({"vectors": payload})
    refusal({"vectors": {"x": [], "y": [], "u": [], "v": []}})
    refusal({"vectors": [1, 2]})


def test_vectors_is_a_lattice_not_a_per_cell_array():
    ok = check_result({"vectors": {"x": [0, 1, 2], "y": [0, 0, 0],
                                   "u": [1, 1, 1], "v": [0, 0, 0]}}, 100000)
    assert len(ok["vectors"]["x"]) == 3


def test_vectors_ceiling_and_non_finite():
    n = MAX_VECTORS + 1
    exc = refusal({"vectors": {"x": [0.0] * n, "y": [0.0] * n, "u": [0.0] * n, "v": [0.0] * n}})
    assert exc.code == "too_large" and str(MAX_VECTORS) in str(exc)
    ok = check_result({"vectors": {"x": [0.0, np.nan], "y": [0.0, 0.0],
                                   "u": [1.0, np.nan], "v": [0.0, 0.0]}}, 10)
    assert ok["vectors"]["x"][1] is None and ok["vectors"]["u"][1] is None
    assert refusal({"vectors": {"x": [0.0, np.inf], "y": [0.0, 0.0],
                                "u": [1.0, 1.0], "v": [0.0, 0.0]}}).code == "non_finite"


def test_vectors_value_and_pitch():
    exc = refusal({"vectors": {"x": [0, 1], "y": [0, 1], "u": [1, 1], "v": [0, 1],
                               "value": [1.0]}})
    assert "1" in str(exc) and "2" in str(exc)
    for bad in (0, -3.0, float("inf"), float("nan")):
        refusal({"vectors": {"x": [0.0], "y": [0.0], "u": [1.0], "v": [0.0], "pitch": bad}})
    ok = check_result({"vectors": {"x": [0.0], "y": [0.0], "u": [1.0], "v": [0.0],
                                   "value": [2.5], "pitch": 4.0, "units": "per 100 um",
                                   "label": "grad", "title": "T", "diverging": True,
                                   "domain": [-1, 1], "colormap": "icefire"}}, 10)
    assert ok["vectors"]["pitch"] == 4.0 and ok["vectors"]["value"] == [2.5]
    assert ok["vectors"]["diverging"] is True and ok["vectors"]["domain"] == [-1.0, 1.0]


# ---------------------------------------------------------------------------
# radar and chord
# ---------------------------------------------------------------------------
def test_radar_domain_missing_values_and_lengths():
    radar = {"axes": ["x", "y", "z"], "domain": [-2, 2],
             "profiles": [{"label": "a", "values": [-1, None, 1]}]}
    assert check_result({"radar": radar}, 0)["radar"]["profiles"][0]["values"] == [-1, None, 1]
    for values in ([0, 1], [0, float("inf"), 1], [0, 3, 1]):
        radar["profiles"][0]["values"] = values
        refusal({"radar": radar}, 0)
    refusal({"radar": dict(radar, axes=["x", "y"])}, 0)
    refusal({"radar": dict(radar, domain=[1, 1])}, 0)


def test_chord_rejects_bad_endpoints_widths_and_colours():
    nodes = [{"label": "A"}, {"label": "B", "color": "#00ff00"}]
    ok = check_result({"chord": {"nodes": nodes, "links": [
        {"source": 0, "target": 1, "value": 2, "score": None}]}}, 0)["chord"]
    assert ok["links"][0]["value"] == 2.0 and ok["nodes"][1]["color"] == "#00ff00"
    for link in ({"source": 0, "target": 2, "value": 1},
                 {"source": 0, "target": 1, "value": -1},
                 {"source": True, "target": 1, "value": 1},
                 {"source": 0, "target": 1, "value": 1, "color": "green"}):
        refusal({"chord": {"nodes": nodes, "links": [link]}}, 0)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def test_labels_are_codes_into_categories_with_null_for_no_label():
    out = check_result({"labels": {"values": [0, 2, None, 1, np.nan],
                                   "categories": ["N1", "N2", "N3"],
                                   "colors": ["#e6194b", "#3cb44b", "#FFE119"],
                                   "label": "Niche"}}, 5)["labels"]
    assert out == {"values": [0, 2, None, 1, None], "categories": ["N1", "N2", "N3"],
                   "colors": ["#e6194b", "#3cb44b", "#FFE119"], "label": "Niche"}
    assert all(type(v) is int for v in out["values"] if v is not None)


def test_labels_accept_arrays_and_fill_defaults():
    floats = np.array([0.0, 1.0, np.nan, 1.0])
    out = check_result({"labels": {"values": floats, "categories": np.array([7, 8])}}, 4)["labels"]
    assert out["values"] == [0, 1, None, 1] and out["categories"] == ["7", "8"]
    assert out["colors"] is None and out["label"] == "label"
    ints = check_result({"labels": {"values": np.zeros(4, dtype=np.int8),
                                    "categories": ("only",)}}, 4)["labels"]
    assert ints["values"] == [0, 0, 0, 0]


def test_labels_must_cover_every_cell():
    exc = refusal({"labels": {"values": [0] * 9, "categories": ["a"]}})
    assert exc.code == "bad_length" and "9" in str(exc) and "10" in str(exc)


@pytest.mark.parametrize("bad", [-1, 3, 1.5, "x"])
def test_labels_codes_must_index_a_category(bad):
    values = [0] * 9 + [bad]
    exc = refusal({"labels": {"values": values, "categories": ["a", "b", "c"]}})
    assert exc.code == "bad_result"
    if bad != "x":
        assert "None" in str(exc), "the message must say how to mark a cell with no label"


def test_labels_infinity_is_a_bug():
    assert refusal({"labels": {"values": [0] * 9 + [np.inf],
                               "categories": ["a"]}}).code == "non_finite"


def test_labels_categories_are_bounded_and_unique():
    ok = [0] * 10
    assert refusal({"labels": {"values": ok}}).code == "bad_result"
    assert refusal({"labels": {"values": ok, "categories": "ab"}}).code == "bad_result"
    assert refusal({"labels": {"values": ok, "categories": []}}).code == "bad_result"
    exc = refusal({"labels": {"values": ok,
                              "categories": [f"c{i}" for i in range(MAX_CATEGORIES + 1)]}})
    assert exc.code == "too_large" and str(MAX_CATEGORIES) in str(exc)
    check_result({"labels": {"values": ok,
                             "categories": [f"c{i}" for i in range(MAX_CATEGORIES)]}}, 10)
    assert refusal({"labels": {"values": ok, "categories": ["x" * 81]}}).code == "bad_result"
    check_result({"labels": {"values": ok, "categories": ["x" * 80]}}, 10)
    assert refusal({"labels": {"values": ok, "categories": ["a", "a"]}}).code == "bad_result"
    assert refusal({"labels": [0] * 10}).code == "bad_result"


def test_labels_colors_are_one_hex_per_category():
    base = {"values": [0] * 10, "categories": ["a", "b"]}
    for colors in (["#000000"], ["#000000", "red"], ["#000000", "#12345"], "#000000"):
        assert refusal({"labels": dict(base, colors=colors)}).code == "bad_result"


# ---------------------------------------------------------------------------
# views, files, images
# ---------------------------------------------------------------------------
def test_multiple_views_and_arbitrary_downloads():
    payload = check_result({
        "views": [html_view("<svg></svg>", id="polar"),
                  html_view("<canvas></canvas>", id="network")],
        "files": [download_file("plot.pdf", b"%PDF-1.7\x00", media_type="application/pdf"),
                  download_file("scores.csv", "cell,value\n1,2\n", media_type="text/csv")]}, 20)
    assert len(payload["views"]) == 2
    assert base64.b64decode(payload["files"][0]["base64"]) == b"%PDF-1.7\x00"
    assert payload["files"][0]["size"] == 9
    assert check_result(payload, 20) == payload
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("result", [
    dict(views=[]), dict(files=[]), dict(views=[html_view(42)]),
    dict(views=[html_view("a", height=0)]),
    dict(views=[html_view("a"), html_view("b")]),
    dict(views=[html_view("x" * (MAX_EXTENSION_BYTES + 1))]),
    dict(files=[download_file("../secret", "data")]),
    dict(files=[download_file("C:\\secret", "data")]),
    dict(files=[{"name": "x", "base64": "!"}]),
    dict(files=[download_file("x", "a", media_type="text/html\nX: 1")]),
])
def test_malformed_and_oversized_envelopes(result):
    refusal(result)


def test_image_builds_png_and_svg_entries():
    png = image(PNG, "UMAP")
    assert png == {"title": "UMAP", "png": base64.b64encode(PNG).decode("ascii")}
    assert image(bytearray(PNG), "again")["png"] == png["png"]
    svg = '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
    assert image(svg, "Scale") == {"title": "Scale", "svg": svg}
    out = check_result({"images": [png, image(svg, "Scale")]}, 0)["images"]
    assert out == [png, {"title": "Scale", "svg": svg}]


def test_image_refuses_what_it_cannot_show():
    with pytest.raises(PluginError) as exc:
        image(b"GIF89a....", "gif")
    assert exc.value.code == "bad_result" and "PNG" in str(exc.value)
    with pytest.raises(PluginError):
        image(12, "a number")


def test_image_saves_a_matplotlib_figure_as_png(monkeypatch):
    class Figure:                           # stands in for matplotlib.figure.Figure
        def savefig(self, fh, format):
            assert format == "png"
            fh.write(PNG)
    fake = types.ModuleType("matplotlib.figure")
    fake.Figure = Figure
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.figure", fake)
    assert image(Figure(), "fig") == {"title": "fig",
                                      "png": base64.b64encode(PNG).decode("ascii")}


def test_image_with_real_matplotlib():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2, 2))
    fig.add_subplot().plot([0, 1], [1, 0])
    out = check_result({"images": [image(fig, "line")]}, 0)["images"][0]
    assert base64.b64decode(out["png"]).startswith(PNG_SIGNATURE)


@pytest.mark.parametrize("item", [
    {"png": base64.b64encode(PNG).decode()},                               # no title
    {"title": " ", "png": base64.b64encode(PNG).decode()},
    {"title": "t" * 201, "png": base64.b64encode(PNG).decode()},
    {"title": "both", "png": base64.b64encode(PNG).decode(), "svg": "<svg/>"},
    {"title": "neither"},
    {"title": "bytes", "png": PNG},
    {"title": "not base64", "png": "!!!"},
    {"title": "not a png", "png": base64.b64encode(b"GIF89a").decode()},
    {"title": "a path", "svg": "figures/umap.svg"},
    "not an object",
])
def test_malformed_images_are_refused(item):
    refusal({"images": [item]})


def test_images_are_capped_and_share_the_budget():
    one = image(PNG, "x")
    assert refusal({"images": []}).code == "bad_result"
    assert refusal({"images": [one] * (MAX_IMAGES + 1)}).code == "too_large"
    check_result({"images": [one] * MAX_IMAGES}, 0)
    half = MAX_EXTENSION_BYTES // 2
    big_svg = "<svg>" + " " * half + "</svg>"
    check_result({"images": [image(big_svg, "big")]}, 0)
    exc = refusal({"views": [html_view("x" * half)], "images": [image(big_svg, "big")]})
    assert exc.code == "too_large" and "images" in str(exc), \
        "views, files and images draw on one budget"
