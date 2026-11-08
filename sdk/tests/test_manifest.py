"""plugin.json: the generated form's source of truth, in schemas 1, 2 and 3."""

import json
import os

import pytest

from pointillviz_tool.manifest import (
    MAX_JSON_PARAM_BYTES, PARAM_TYPES, SCHEMAS, ManifestError, check_requires,
    coerce_params, language, load, sweep_values, validate, visible_params,
)

BASE = {"id": "demo-tool", "name": "Demo", "version": "1.0.0", "author": "a",
        "description": "d", "outputs": ["summary"]}
V2 = dict(BASE, schema=2, outputs=["views"])
V3 = dict(BASE, schema=3)


def refused(raw, why=None, **kwargs):
    with pytest.raises(ManifestError) as exc:
        validate(raw, **kwargs)
    if why is not None:
        assert why in str(exc.value), f"refused, but for: {exc.value}"
    return exc.value


def test_vocabulary():
    assert SCHEMAS == (1, 2, 3)
    assert len(PARAM_TYPES) == 11


# ---------------------------------------------------------------------------
# Identity, entry, params (every schema)
# ---------------------------------------------------------------------------
def test_identity_fields_are_strict():
    for bad in ("Demo", "d", "demo_tool", "1demo", "demo-", "demo-tool\n"):
        refused({**BASE, "id": bad})
    for bad in ("1.0", "v1.0.0", "1.0.0-rc1", "1.0.0\n"):
        refused({**BASE, "version": bad})
    refused({**BASE, "params": [{"name": "k\n", "type": "bool"}]})
    refused({**BASE, "name": "  "})
    refused({**BASE, "schema": 999}, "schemas 1, 2 and 3")
    refused({**BASE, "schema": "3"})
    refused([BASE])


def test_entry_cannot_escape_the_bundle():
    assert "plain .py filename" in str(refused({**BASE, "entry": "../../backend/serve.py"}))
    refused({**BASE, "entry": "sub/tool.py"})
    refused({**BASE, "entry": "tool.sh"})
    refused({**V3, "entry": "../tool.R"})
    refused({**V3, "entry": "tool.Rmd"})


def test_entry_must_exist_when_a_directory_is_given(tmp_path):
    refused(BASE, "does not exist", plugin_dir=str(tmp_path))
    (tmp_path / "tool.py").write_text("def run(ds, params):\n    return {}\n")
    assert validate(BASE, str(tmp_path))["entry"] == "tool.py"


def test_numeric_params_must_be_bounded():
    assert "min and max" in str(refused({**BASE, "params": [
        {"name": "k", "type": "int", "default": 5}]})), \
        "an unbounded numeric has no drawable control and nothing to clamp against"
    refused({**BASE, "params": [{"name": "k", "type": "int", "min": 9, "max": 3}]})
    ok = validate({**BASE, "params": [{"name": "k", "type": "int", "min": 3, "max": 9}]})
    assert ok["params"][0]["step"] == 1.0


@pytest.mark.parametrize("param, requires", [
    ({"name": "k", "type": "int", "min": "three", "max": 9}, None),
    ({"name": "k", "type": "int", "min": 3, "max": [9]}, None),
    ({"name": "k", "type": "float", "min": 0, "max": 1, "step": "fine"}, None),
    ({"name": "k", "type": "float", "min": float("nan"), "max": 1}, None),
    ({"name": "k", "type": "float", "min": 0, "max": float("inf")}, None),
    ({"name": "k", "type": "int", "min": 1, "max": 9, "sweep": True, "sweep_steps": "x"}, None),
    ({"name": "g", "type": "genes", "max_items": "many"}, None),
    (None, {"min_cells": "lots"}),
])
def test_a_non_numeric_number_is_a_manifest_error(param, requires):
    raw = dict(BASE, params=[param] if param else [], requires=requires or {})
    refused(raw, "must be a")


def test_a_genes_preset_is_kept_and_checked():
    gs = {"name": "gs", "type": "genes", "max_items": 30, "optional": True}
    got = validate({**V3, "params": [dict(gs, preset="top_markers", preset_n=20)]})["params"][0]
    assert (got["preset"], got["preset_n"], got["max_items"]) == ("top_markers", 20, 30)
    got = validate({**V3, "params": [dict(gs, preset="top_markers")]})["params"][0]
    assert got["preset_n"] == 30                         # no preset_n: as many as max_items
    assert "preset" not in validate({**V3, "params": [gs]})["params"][0]
    refused({**V3, "params": [dict(gs, preset="variable")]}, "preset must be one of top_markers")
    refused({**V3, "params": [dict(gs, preset="top_markers", preset_n=31)]}, "preset_n must be from 1")
    refused({**V3, "params": [dict(gs, preset="top_markers", preset_n=-1)]}, "preset_n must be from 1")


def test_param_names_and_types_are_closed():
    refused({**BASE, "params": [{"name": "2k", "type": "int"}]})
    refused({**BASE, "params": [{"name": "k-1", "type": "int", "min": 0, "max": 1}]})
    refused({**BASE, "params": [{"name": "k", "type": "colour"}]})
    refused({**BASE, "params": [{"name": "s", "type": "select"}]})
    refused({**BASE, "params": [{"name": "k", "type": "bool"}, {"name": "k", "type": "text"}]})
    refused({**BASE, "outputs": ["heatmap"]})
    refused({**BASE, "outputs": []})
    refused({**BASE, "outputs": ["summary", "summary"]})


def test_unknown_keys_are_dropped_and_defaults_filled():
    m = validate({**BASE, "colour": "red", "tags": [str(i) for i in range(12)]})
    assert "colour" not in m and len(m["tags"]) == 8
    assert m["icon"] == "extension" and m["entry"] == "tool.py" and m["license"] == ""
    assert m["requires"] == {"genes": False, "cell_types": False, "categoricals": False,
                             "spatial": "any", "min_cells": 0}


# ---------------------------------------------------------------------------
# Schemas 1 and 2
# ---------------------------------------------------------------------------
def test_schema_one_and_two_output_contracts():
    assert validate(V2)["schema"] == 2
    assert validate(dict(V2, schema=1, outputs=["table"]))["schema"] == 1
    for edits in [dict(schema=1), dict(outputs="views"), dict(optional_outputs=["views"]),
                  dict(outputs=["new-plot"]), dict(outputs=["labels"]), dict(outputs=["images"]),
                  dict(schema=1, outputs=["table"], optional_outputs=["summary"])]:
        refused(dict(V2, **edits))
    assert validate(dict(V2, outputs=[], optional_outputs=["views"]))["outputs"] == []


def test_resource_keys_are_ignored_before_schema_three():
    for schema in (1, 2):
        m = validate({**BASE, "schema": schema, "timeout_s": 0, "memory_mb": "lots",
                      "gpu": "required", "requires": {"scipy": True}})
        assert "timeout_s" not in m and "memory_mb" not in m
        assert "scipy" not in m["requires"] and m["gpu"] is None


def test_json_settings_need_schema_two():
    refused({**BASE, "params": [{"name": "s", "type": "json"}]}, "schema 2")
    manifest = validate(dict(V2, params=[dict(name="settings", type="json",
                                               default={"groups": [1, 2]})]))
    assert coerce_params(manifest, {})["settings"] == {"groups": [1, 2]}
    assert coerce_params(manifest, {"settings": '[{"a": true}]'})["settings"] == [{"a": True}]
    for value in ['{', 'null', '"text"', '{"a": NaN}', {"a": float('inf')},
                  {"huge": "x" * MAX_JSON_PARAM_BYTES}]:
        with pytest.raises(ManifestError):
            coerce_params(manifest, {"settings": value})


# ---------------------------------------------------------------------------
# Schema 3
# ---------------------------------------------------------------------------
def test_schema_three_runs_python_or_r():
    assert language(validate(V3)) == "python"
    m = validate({**V3, "entry": "tool.R"})
    assert m["entry"] == "tool.R" and language(m) == "r"
    assert language(validate({**V3, "entry": "niches.r"})) == "r"
    refused({**BASE, "entry": "tool.R"}, "plain .py filename")
    refused({**V2, "entry": "tool.R"}, "plain .py filename")


def test_schema_three_declares_a_gpu_in_one_key():
    assert validate(V3)["gpu"] is None
    assert validate({**V3, "gpu": None})["gpu"] is None
    for mode in ("optional", "required"):
        assert validate({**V3, "gpu": mode})["gpu"] == mode
    for bad in (True, "cuda", "yes", 1):
        refused({**V3, "gpu": bad}, "optional")


@pytest.mark.parametrize("edit, key", [
    ({"timeout_s": 120}, "timeout_s"),
    ({"memory_mb": 4096}, "memory_mb"),
    ({"requires": {"scipy": True}}, "requires.scipy"),
    ({"requires": {"scipy": False}}, "requires.scipy"),
])
def test_schema_three_refuses_the_removed_keys(edit, key):
    refused({**V3, **edit}, key)


def test_schema_three_outputs():
    m = validate({**V3, "outputs": ["labels", "table"], "optional_outputs": ["images", "files"]})
    assert m["outputs"] == ["labels", "table"] and m["optional_outputs"] == ["images", "files"]
    assert "timeout_s" not in m and "memory_mb" not in m and "scipy" not in m["requires"]


# ---------------------------------------------------------------------------
# Coercion and requirements
# ---------------------------------------------------------------------------
def test_coerce_gives_run_the_type_it_was_promised():
    m = validate({**BASE, "params": [
        {"name": "k", "type": "int", "min": 3, "max": 50, "default": 15},
        {"name": "r", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
        {"name": "on", "type": "bool", "default": True},
        {"name": "mode", "type": "select", "options": ["a", "b"], "default": "a"},
        {"name": "gs", "type": "genes", "max_items": 2, "default": ["x"]},
    ]})
    got = coerce_params(m, {"k": "7", "r": "0.25", "on": "false", "mode": "b",
                            "gs": ["a", "b", "c"], "undeclared": 1})
    assert got == {"k": 7, "r": 0.25, "on": False, "mode": "b", "gs": ["a", "b"]}
    assert isinstance(got["k"], int) and isinstance(got["r"], float)


def test_out_of_range_clamps_but_an_unknown_select_or_non_number_refuses():
    m = validate({**BASE, "params": [
        {"name": "k", "type": "int", "min": 3, "max": 50, "default": 15},
        {"name": "mode", "type": "select", "options": ["a", "b"], "default": "a"}]})
    assert coerce_params(m, {"k": 9999})["k"] == 50, \
        "a slider cannot produce an out-of-range value, so one means a stale client " \
        "and clamping is the recovery a user would want"
    assert coerce_params(m, {"k": -5})["k"] == 3
    with pytest.raises(ManifestError):
        coerce_params(m, {"mode": "z"})
    with pytest.raises(ManifestError):
        coerce_params(m, {"k": "seven"})
    bad_default = validate({**BASE, "params": [
        {"name": "k", "type": "int", "min": 3, "max": 50, "default": "fifteen"}]})
    with pytest.raises(ManifestError):
        coerce_params(bad_default, {})


def test_a_missing_required_param_refuses_by_its_label():
    m = validate({**BASE, "params": [{"name": "g", "type": "gene", "label": "Gene"}]})
    with pytest.raises(ManifestError) as exc:
        coerce_params(m, {})
    assert "Gene" in str(exc.value), "the message names the LABEL the user saw, not the key"
    m2 = validate({**BASE, "params": [{"name": "g", "type": "gene", "optional": True}]})
    assert coerce_params(m2, {})["g"] is None


def test_requires_explains_itself():
    m = validate({**BASE, "requires": {"genes": True, "cell_types": True,
                                       "spatial": "3d", "min_cells": 1000}})
    assert check_requires(m, {"genes": {"n_genes": 5}, "cell_types": ["a"],
                              "spatial": {"mode": "3d"}, "n_cells": 5000}) == ""
    for meta, word in (({"cell_types": ["a"], "spatial": {"mode": "3d"}, "n_cells": 5000}, "gene"),
                       ({"genes": {"n_genes": 5}, "spatial": {"mode": "3d"}, "n_cells": 5000}, "cell-type"),
                       ({"genes": {"n_genes": 5}, "cell_types": ["a"], "spatial": {"mode": "2d"}, "n_cells": 5000}, "3D"),
                       ({"genes": {"n_genes": 5}, "cell_types": ["a"], "spatial": {"mode": "3d"}, "n_cells": 10}, "1,000")):
        reason = check_requires(m, meta)
        assert reason and word in reason, f"expected {word!r} in the refusal, got {reason!r}"


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
RADIUS = {"name": "radius", "type": "float", "label": "Radius", "default": 60.0,
          "min": 5.0, "max": 1000.0, "step": 5.0, "sweep": True, "sweep_steps": 8}
SWEPT = dict(BASE, outputs=["matrix", "table", "summary"], params=[RADIUS])


def test_only_one_number_may_be_swept_and_never_on_a_per_cell_tool():
    m = validate(SWEPT)
    assert m["sweep"] == "radius"
    assert m["params"][0]["sweep"] is True and m["params"][0]["sweep_steps"] == 8
    refused(dict(SWEPT, params=[RADIUS, {"name": "f", "type": "text", "sweep": True}]),
            "only an int or float")
    refused(dict(SWEPT, params=[RADIUS, {"name": "k", "type": "int", "min": 1, "max": 9,
                                         "sweep": True}]), "at most one parameter")
    for per_cell in ("cells", "vectors"):
        refused(dict(SWEPT, outputs=["matrix", per_cell]), "per-cell outputs")
    refused(dict(SWEPT, schema=3, outputs=["matrix", "labels"]), "per-cell outputs")
    refused(dict(SWEPT, params=[dict(RADIUS, sweep_steps=40)]), "sweep_steps")
    assert validate(dict(SWEPT, params=[dict(RADIUS, sweep=False)]))["sweep"] is None


def test_sweep_values_stay_in_range_on_the_step():
    m = validate(SWEPT)
    assert sweep_values(m, 5, 60, 4)[1] == [5.0, 25.0, 40.0, 60.0]
    assert sweep_values(m, -100, 99999, 3)[1] == [5.0, 505.0, 1000.0]
    assert sweep_values(m, 60, 5, 2)[1] == [5.0, 60.0], "a reversed range is the same range"
    assert sweep_values(m, 5, 10, 12)[1] == [5.0, 10.0], "duplicates after snapping collapse"
    for lo, hi, n in ((5, 5, 3), (5, 60, 1), (5, 60, 13), ("a", 60, 3)):
        with pytest.raises(ManifestError):
            sweep_values(m, lo, hi, n)
    im = validate(dict(SWEPT, params=[{"name": "k", "type": "int", "default": 5,
                                       "min": 1, "max": 50, "sweep": True}]))
    _, vals = sweep_values(im, 1, 50, 4)
    assert vals == [1, 17, 34, 50] and all(isinstance(v, int) for v in vals)
    with pytest.raises(ManifestError):
        sweep_values(validate(BASE))


# ---------------------------------------------------------------------------
# show_when
# ---------------------------------------------------------------------------
def when(param, *values):
    return {"param": param, "in": list(values)}


SOURCE = {"name": "source", "type": "select", "label": "Source", "default": "category",
          "options": ["category", "gene", "continuous"]}
BRANCHES = [
    SOURCE,
    {"name": "field", "type": "categorical_field", "show_when": when("source", "category")},
    {"name": "level", "type": "category_value", "field_param": "field",
     "show_when": when("source", "category")},
    {"name": "gene", "type": "gene", "label": "Gene", "show_when": when("source", "gene")},
    {"name": "column", "type": "continuous_field", "show_when": when("source", "continuous")},
    {"name": "sigma", "type": "float", "default": 1.5, "min": 0.5, "max": 3.0,
     "show_when": when("source", "gene", "continuous")},
]
BRANCHY = dict(V3, params=BRANCHES)


def with_param(i, **edits):
    """BRANCHY with params[i] changed; an edit to None removes that key."""
    params = [dict(p) for p in BRANCHES]
    params[i] = {k: v for k, v in dict(params[i], **edits).items() if v is not None}
    return dict(BRANCHY, params=params)


def without_rules(params):
    return [{k: v for k, v in p.items() if k != "show_when"} for p in params]


@pytest.mark.parametrize("rule, why", [
    ("gene", 'exactly the keys "param" and "in"'),
    ({"param": "source"}, 'exactly the keys "param" and "in"'),
    ({"param": "source", "values": ["gene"]}, 'exactly the keys "param" and "in"'),
    ({"param": "source", "in": ["gene"], "also": 1}, 'exactly the keys "param" and "in"'),
    ({"param": 3, "in": ["gene"]}, '"param" must be the name of another parameter, not 3'),
    ({"param": "source", "in": []}, '"in" must be a non-empty list'),
    ({"param": "source", "in": "gene"}, '"in" must be a non-empty list'),
])
def test_show_when_has_one_shape(rule, why):
    assert str(refused(with_param(3, show_when=rule), why)).startswith("params[3] (gene): ")


def test_show_when_names_another_select_or_bool():
    refused(with_param(3, show_when=when("gene", "gene")), "show_when names gene itself")
    refused(with_param(3, show_when=when("sorce", "gene")), "'sorce', which is not a parameter")
    refused(with_param(3, show_when=when("sigma", 1.5)),
            "names sigma, a float parameter; only a select or a bool")
    refused(with_param(3, show_when=when("column", "x")), "a continuous_field parameter")


def test_show_when_values_are_the_controlling_parameters_own():
    refused(with_param(3, show_when=when("source", "genes")),
            "lists 'genes', which is not among source's options (category, gene, continuous)")
    refused(with_param(3, show_when=when("source", "gene", "x", "y")), "'x', 'y', which are not")
    smooth = {"name": "smooth", "type": "bool", "default": False}
    sigma = {"name": "sigma", "type": "float", "default": 1.0, "min": 0.5, "max": 3.0}
    for bad in ("true", 1, None):
        refused(dict(V3, params=[smooth, dict(sigma, show_when=when("smooth", True, bad))]),
                f'smooth is a bool, so show_when "in" must hold true and/or false, not {bad!r}')


def test_show_when_values_are_normalised():
    m = validate(dict(V3, params=[
        {"name": "k", "type": "select", "default": "5", "options": [5, 10]},
        {"name": "g", "type": "gene", "show_when": when("k", 10, "10", 5)},
        {"name": "on", "type": "bool", "default": True},
        {"name": "r", "type": "float", "default": 1, "min": 0, "max": 2,
         "show_when": when("on", True, False, True)}]))
    assert m["params"][1]["show_when"] == {"param": "k", "in": ["10", "5"]}
    assert m["params"][3]["show_when"] == {"param": "on", "in": [True, False]}
    assert coerce_params(m, {"k": 10, "g": "Actb"})["g"] == "Actb", "10 is the option '10'"


def test_show_when_adds_its_key_and_changes_nothing_else():
    plain = validate(dict(BRANCHY, params=without_rules(BRANCHES)))
    assert not any("show_when" in p for p in plain["params"])
    ruled = validate(BRANCHY)
    assert [p["name"] for p in ruled["params"] if "show_when" in p] == [
        "field", "level", "gene", "column", "sigma"]
    # Byte for byte, so index.json and the app see no change in a tool that
    # does not use the key.
    assert json.dumps(dict(ruled, params=without_rules(ruled["params"]))) == json.dumps(plain)
    assert validate(dict(BRANCHY, params=[dict(p, show_when=None) for p in without_rules(
        BRANCHES)])) == plain, "a null show_when is no show_when"


def test_show_when_refuses_a_cycle():
    a = {"name": "a", "type": "bool", "default": True, "show_when": when("c", "x")}
    b = {"name": "b", "type": "bool", "default": True, "show_when": when("a", True)}
    c = {"name": "c", "type": "select", "default": "x", "options": ["x", "y"],
         "show_when": when("b", False)}
    err = refused(dict(V3, params=[a, b, c]), "cycle")
    assert str(err).startswith("params[0] (a): show_when forms a cycle (a → c → b → a)")
    # A parameter that only leads into the cycle is not part of it.
    lead = {"name": "d", "type": "gene", "show_when": when("a", True)}
    refused(dict(V3, params=[lead, a, b, c]), "params[1] (a): show_when forms a cycle (a → c → b → a)")
    validate(dict(V3, params=[{k: v for k, v in a.items() if k != "show_when"}, b, c, lead]))


def test_a_parameter_comes_after_the_choice_that_shows_it():
    refused(dict(V3, params=[BRANCHES[3], SOURCE]),
            "params[0] (gene): show_when names source, which is declared after it; put source "
            "above gene, because the form shows a parameter below the choice that reveals it.")


def test_a_swept_parameter_is_always_shown():
    toggle = {"name": "on", "type": "bool", "default": True}
    refused(dict(SWEPT, params=[toggle, dict(RADIUS, show_when=when("on", True))]),
            "a swept parameter cannot declare show_when")
    m = validate(dict(SWEPT, params=[toggle, RADIUS, {"name": "k", "type": "int", "default": 5,
                                                      "min": 1, "max": 9,
                                                      "show_when": when("on", True)}]))
    assert m["sweep"] == "radius", "the other parameters of a swept tool may declare show_when"


def test_a_level_is_hidden_whenever_its_field_is():
    err = refused(with_param(2, show_when=None),
                  "params[2] (level): its levels come from field, which is shown only while "
                  "source is category, so level must be hidden whenever field is")
    assert 'give it the same "show_when": {"param": "source", "in": ["category"]}' in str(err)
    refused(with_param(2, show_when=when("source", "category", "gene")),
            "must be hidden whenever field is")
    # Fewer values than the field's are fine, and so is a chain that passes
    # through the field's controlling parameter with values that show it.
    wide = [dict(p) for p in BRANCHES]
    wide[1] = dict(wide[1], show_when=when("source", "category", "gene"))
    validate(dict(BRANCHY, params=wide))
    mode = {"name": "mode", "type": "select", "default": "fine", "options": ["fine", "coarse"],
            "show_when": when("source", "category")}
    validate(dict(BRANCHY, params=BRANCHES[:2] + [mode, dict(BRANCHES[2], show_when=when(
        "mode", "fine"))] + BRANCHES[3:]))
    refused(dict(BRANCHY, params=BRANCHES[:2] + [dict(mode, show_when=None), dict(
        BRANCHES[2], show_when=when("mode", "fine"))] + BRANCHES[3:]),
        "must be hidden whenever field is")


def test_a_level_without_a_field_param_follows_the_first_categorical_field():
    # The form's fallback: a level menu with no field_param that names a
    # parameter lists the levels of the first categorical_field's column.
    level = {"name": "level", "type": "category_value", "label": "Category"}
    err = refused(dict(V3, params=[SOURCE, BRANCHES[1], level]),
                  "params[2] (level): with no field_param, the form lists the levels of the first "
                  "categorical_field, field, which is shown only while source is category, so "
                  "level must be hidden whenever field is")
    assert 'give it the same "show_when": {"param": "source", "in": ["category"]}' in str(err)
    refused(dict(V3, params=[SOURCE, BRANCHES[1], dict(level, field_param="fild")]),
            "params[2] (level): its field_param 'fild' names no parameter, and the form then "
            "lists the levels of the first categorical_field, field, which is shown only while")
    validate(dict(V3, params=[SOURCE, BRANCHES[1], dict(level, show_when=when("source",
                                                                          "category"))]))
    # Only the first categorical_field is the fallback; with none, there is
    # no field to follow.
    validate(dict(V3, params=[SOURCE, {"name": "group", "type": "categorical_field"},
                              BRANCHES[1], level]))
    validate(dict(V3, params=[SOURCE, level]))


def test_a_hidden_parameter_gets_its_default_or_none_and_is_never_required():
    m = validate(BRANCHY)
    stale = {"field": "cell_type", "level": "B", "gene": "Actb", "column": "depth", "sigma": 2.5}
    assert coerce_params(m, stale) == {"source": "category", "field": "cell_type", "level": "B",
                                       "gene": None, "column": None, "sigma": 1.5}, \
        "the gene picked before switching away from it never reaches the tool"
    # Shown, the same parameters follow the usual rules.
    assert coerce_params(m, dict(stale, source="gene", sigma="2.5")) == {
        "source": "gene", "field": None, "level": None, "gene": "Actb", "column": None,
        "sigma": 2.5}
    with pytest.raises(ManifestError, match=r"^Gene is required while Source is gene\.$"):
        coerce_params(m, {"source": "gene"})


def test_a_required_message_names_the_choice_that_shows_the_parameter():
    gradient = {"name": "source", "type": "select", "label": "Gradient of", "default": "category",
                "options": [{"value": "category", "label": "Cell type or category"},
                            {"value": "gene", "label": "Gene"},
                            {"value": "continuous", "label": "Continuous column"}]}
    m = validate(dict(V3, params=[
        gradient,
        {"name": "field", "type": "categorical_field", "label": "Column", "optional": True,
         "show_when": when("source", "category")},
        {"name": "level", "type": "category_value", "label": "Category", "field_param": "field",
         "show_when": when("source", "category")},
        {"name": "gene", "type": "gene", "label": "Gene", "show_when": when("source", "gene")},
        {"name": "smooth", "type": "bool", "label": "Smooth", "default": True,
         "show_when": when("source", "gene")},
        {"name": "width", "type": "text", "label": "Width", "show_when": when("smooth", True)},
        {"name": "rough", "type": "text", "label": "Roughness",
         "show_when": when("smooth", False)}]))

    def refusal(values):
        with pytest.raises(ManifestError) as exc:
            coerce_params(m, values)
        return str(exc.value)

    # Vector Field's case: a gene sent without the choice that shows it is
    # told what the default choice wants, by the option's label.
    assert refusal({"gene": "Cd79a"}) == \
        "Category is required while Gradient of is Cell type or category."
    assert refusal({"source": "gene"}) == "Gene is required while Gradient of is Gene."
    # A bool is on or off; down a chain, the immediate controller is named.
    assert refusal({"source": "gene", "gene": "Cd79a"}) == "Width is required while Smooth is on."
    assert refusal({"source": "gene", "gene": "Cd79a", "smooth": "false"}) == \
        "Roughness is required while Smooth is off."
    # Without show_when, the sentence is the one it always was.
    plain = validate(dict(V3, params=[{"name": "g", "type": "gene", "label": "Gene"}]))
    with pytest.raises(ManifestError) as exc:
        coerce_params(plain, {})
    assert str(exc.value) == "Gene is required."


def test_a_hidden_default_is_coerced_like_any_default():
    m = validate(dict(V3, params=[
        {"name": "on", "type": "bool", "default": False},
        {"name": "k", "type": "int", "default": 99, "min": 1, "max": 50,
         "show_when": when("on", True)},
        {"name": "mode", "type": "select", "default": "a", "options": ["a", "b"],
         "show_when": when("on", True)}]))
    # A value that would be refused if shown is ignored while hidden.
    assert coerce_params(m, {"k": "seven", "mode": "zzz"}) == {"on": False, "k": 50, "mode": "a"}
    for on in (True, "true", "1", "yes", "ON"):
        assert coerce_params(m, {"on": on, "k": 7})["k"] == 7
    for off in (False, "false", "0", "no", "", None):
        assert coerce_params(m, {"on": off, "k": 7})["k"] == 50


def test_a_parameter_under_a_hidden_one_is_hidden():
    m = validate(dict(V3, params=[
        SOURCE,
        {"name": "smooth", "type": "bool", "default": True, "show_when": when("source", "gene")},
        {"name": "sigma", "type": "float", "default": 1.0, "min": 0.5, "max": 3.0,
         "show_when": when("smooth", True)}]))
    assert visible_params(m, {}) == {"source"}
    assert visible_params(m, {"source": "gene"}) == {"source", "smooth", "sigma"}
    assert visible_params(m, {"source": "gene", "smooth": "false"}) == {"source", "smooth"}
    # smooth holds true, but is hidden with source, and sigma with it.
    assert coerce_params(m, {"smooth": True, "sigma": 2.0}) == {
        "source": "category", "smooth": True, "sigma": 1.0}
    assert coerce_params(m, {"source": "gene", "sigma": 2.0})["sigma"] == 2.0


def test_visible_params_reads_values_as_coercion_does():
    m = validate(BRANCHY)
    assert visible_params(m, {}) == visible_params(m, None) == {"source", "field", "level"}
    assert visible_params(m, {"source": ""}) == {"source", "field", "level"}, "'' is no value"
    assert visible_params(m, {"source": "continuous"}) == {"source", "column", "sigma"}
    assert visible_params(m, {"source": "nonsense"}) == {"source"}, "not an option: no match"
    # A slice of a manifest, as the gates probe one default at a time: the
    # rule names a parameter that is not there, so the parameter is shown.
    gene = next(p for p in m["params"] if p["name"] == "gene")
    assert visible_params(dict(m, params=[gene]), {}) == {"gene"}
    # A select with no value and no default matches nothing; a bool with none
    # reads as false, as the viewer draws a switch with no default: off.
    bare = validate(dict(V3, params=[dict(SOURCE, default=None, optional=True), BRANCHES[3]]))
    assert visible_params(bare, {}) == {"source"}
    assert coerce_params(bare, {"gene": "Actb"}) == {"source": None, "gene": None}
    switch = validate(dict(V3, params=[
        {"name": "on", "type": "bool", "optional": True},
        {"name": "off_only", "type": "text", "default": "x", "show_when": when("on", False)},
        {"name": "on_only", "type": "text", "default": "y", "show_when": when("on", True)}]))
    assert visible_params(switch, {}) == visible_params(switch, {"on": ""}) == {"on", "off_only"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_load_reads_and_refuses_in_sentences(tmp_path):
    with pytest.raises(ManifestError, match="missing"):
        load(str(tmp_path))
    (tmp_path / "plugin.json").write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load(str(tmp_path))
    (tmp_path / "plugin.json").write_bytes(b'{"name": "\xff"}')
    with pytest.raises(ManifestError, match="not valid JSON"):
        load(str(tmp_path))
    (tmp_path / "plugin.json").write_text(json.dumps({**V3, "entry": "tool.R"}))
    (tmp_path / "tool.R").write_text("run <- function(ds, params) list(summary = 'ok')\n")
    assert load(str(tmp_path))["entry"] == "tool.R"
    os.remove(tmp_path / "tool.R")
    with pytest.raises(ManifestError, match="does not exist"):
        load(str(tmp_path))
