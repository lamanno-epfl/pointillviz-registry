"""`new`: a scaffolded tool is valid as written, and passes every gate in its own environment.

The last two tests build real pixi environments (python, numpy and scipy, and
R for the R template): they need pixi, and network access or a warm package
cache, and take a minute or two the first time.
"""

import json
import os
import re
import subprocess

import pytest

from pointillviz_tool import envs
from pointillviz_tool.dataset import PluginError
from pointillviz_tool.manifest import ManifestError, load
from pointillviz_tool.scaffold import TEMPLATES, scaffold
from pointillviz_tool.validate import format_report, validate


@pytest.mark.parametrize("language,entry,outputs", [
    ("python", "tool.py", ["summary", "cells"]),
    ("r", "tool.R", ["summary", "labels"]),
])
def test_new_writes_a_complete_schema_3_tool(tmp_path, language, entry, outputs):
    out = scaffold("my-new-tool", str(tmp_path), language=language)
    assert out == os.path.join(str(tmp_path), "my-new-tool")
    assert sorted(os.listdir(out)) == sorted(["README.md", "pixi.toml", "plugin.json", entry])
    m = load(out)
    assert (m["schema"], m["id"], m["name"], m["entry"]) == (3, "my-new-tool", "My New Tool", entry)
    assert m["outputs"] == outputs and m["gpu"] is None
    # Public-registry hygiene: an author field with no email address in it.
    assert "@" not in m["author"]
    for name in os.listdir(out):
        text = open(os.path.join(out, name), encoding="utf-8").read()
        assert "{{" not in text, f"{name} still holds a placeholder"
    toml = open(os.path.join(out, "pixi.toml"), encoding="utf-8").read()
    declared = re.findall(r"^([a-z-]+) = ", toml.split("[dependencies]")[1], re.M)
    assert declared == ["python", "numpy", "scipy"] + (["r-base", "r-jsonlite"]
                                                        if language == "r" else [])
    assert 'name = "my-new-tool"' in toml


def test_new_refuses_a_bad_id_or_an_existing_folder(tmp_path):
    with pytest.raises(ManifestError):
        scaffold("Not_An_Id", str(tmp_path))
    assert os.listdir(tmp_path) == []
    scaffold("taken", str(tmp_path))
    with pytest.raises(FileExistsError):
        scaffold("taken", str(tmp_path), language="r")
    with pytest.raises(ValueError):
        scaffold("other", str(tmp_path), language="julia")


def test_the_templates_are_the_only_source(tmp_path):
    # What `new` writes is the template folder, placeholders filled in, and nothing else.
    for language in ("python", "r"):
        out = scaffold(f"from-{language}", str(tmp_path), language=language)
        assert sorted(os.listdir(out)) == sorted(os.listdir(os.path.join(TEMPLATES, language)))


def _pixi_or_skip():
    try:
        return envs.pixi_exe()
    except PluginError:
        pytest.skip("pixi is not installed")


@pytest.mark.parametrize("language", ["python", "r"])
def test_a_scaffolded_tool_passes_every_gate_in_its_own_environment(tmp_path, language):
    pixi = _pixi_or_skip()
    out = scaffold(f"starter-{language}", str(tmp_path), language=language)
    locked = subprocess.run([pixi, "lock", "--manifest-path", os.path.join(out, "pixi.toml")],
                            capture_output=True, text=True)
    assert locked.returncode == 0, locked.stderr
    report = validate(out)                 # own environment: there is a pixi.toml
    assert report["ok"], format_report(report)
    assert all(g["status"] == "pass" for g in report["gates"]), format_report(report)
    entry = next(g for g in report["gates"] if g["id"] == "entry")
    assert entry["message"].startswith("Built the default environment")
    assert os.path.isdir(os.path.join(out, ".pixi", "envs", "default"))
    assert report["language"] == language
    json.dumps(report)
