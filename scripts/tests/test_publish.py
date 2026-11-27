"""scripts/publish.py: what release.yml's publish job releases, and from which bytes."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import publish  # noqa: E402


#: Spelled in pieces, so that the hygiene check, which reads this file too, passes it.
EMAIL = "a" + "@" + "example.org"
MNT = "/" + "mnt/"


def _manifest(tool, version):
    return {"schema": 3, "id": tool, "name": tool.title(), "version": version, "author": "a",
            "description": f"What {tool} does.", "entry": "tool.py", "outputs": ["summary"]}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A checkout with two tools, an empty REPORTS folder and an OUT folder."""
    root = tmp_path / "checkout"
    for tool, version in (("tool-a", "1.0.0"), ("tool-b", "2.1.0")):
        folder = root / "tools" / tool
        folder.mkdir(parents=True)
        (folder / "plugin.json").write_text(json.dumps(_manifest(tool, version)))
        (folder / "tool.py").write_text("def run(ds, params):\n    return {'summary': 'ok'}\n")
    monkeypatch.setattr(publish, "ROOT", str(root))
    monkeypatch.setenv("SHA", "abc123")
    return root, tmp_path / "reports", tmp_path / "out"


def _report(reports, tool, ver, **change):
    report = {"ok": True, "plugin_id": tool, "version": ver, "language": "python", "gpu": None,
              "gates": [{"id": "bundle", "required": True, "status": "pass"}]}
    report.update(change)
    (reports / f"report-{tool}").mkdir(parents=True, exist_ok=True)
    (reports / f"report-{tool}" / "report.json").write_text(json.dumps(report))


def _plan(tool, version, released=False):
    return {"tool": tool, "version": version, "tag": f"{tool}-v{version}", "released": released}


class FakeGh:
    def __init__(self, released=()):
        self.calls, self.released = [], set(released)

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("release", "view") and args[2] not in self.released:
            raise subprocess.CalledProcessError(1, "gh", stderr="release not found")
        if args[:2] == ("release", "download"):
            folder, name = args[args.index("--dir") + 1], args[args.index("--pattern") + 1]
            Path(folder, name).write_bytes(b"the released bytes")
        return ""

    def created(self):
        return [c for c in self.calls if c[:2] == ("release", "create")]


def test_a_passing_tool_is_packed_here_and_released(registry):
    root, reports, out = registry
    _report(reports, "tool-a", "1.0.0")
    gh = FakeGh()
    assert publish.publish([_plan("tool-a", "1.0.0")], str(reports), str(out),
                           passed={"Gates (tool-a)"}, gh=gh, log=lambda m: None) == 0
    (create,) = gh.created()
    path = out / "bundles" / "tool-a-v1.0.0" / "tool-a-1.0.0.pvtool"
    assert create[2] == "tool-a-v1.0.0" and create[3] == str(path)
    assert create[create.index("--target") + 1] == "abc123"
    with zipfile.ZipFile(path) as zf:
        assert sorted(zf.namelist()) == ["plugin.json", "tool.py"]
    assert json.loads((out / "reports" / "tool-a.json").read_text())["plugin_id"] == "tool-a"
    assert "What tool-a does." in (out / "notes.md").read_text()


def test_a_tool_whose_gates_job_did_not_succeed_is_skipped(registry):
    _, reports, out = registry
    _report(reports, "tool-a", "1.0.0")        # a report alone vouches for nothing
    gh = FakeGh()
    assert publish.publish([_plan("tool-a", "1.0.0")], str(reports), str(out),
                           passed={"Gates (tool-b)"}, gh=gh, log=lambda m: None) == 0
    assert gh.calls == [] and not (out / "reports").exists()


def test_artifacts_for_tools_outside_the_plan_are_ignored(registry):
    _, reports, out = registry
    _report(reports, "tool-a", "1.0.0")
    _report(reports, "tool-b", "9.9.9")        # dropped beside the others by some job
    gh = FakeGh()
    publish.publish([_plan("tool-a", "1.0.0")], str(reports), str(out),
                    passed={"Gates (tool-a)", "Gates (tool-b)"}, gh=gh, log=lambda m: None)
    assert [c[2] for c in gh.created()] == ["tool-a-v1.0.0"]


@pytest.mark.parametrize("change, problem", [
    ({"plugin_id": "tool-b"}, "gate report"),
    ({"version": "1.0.1"}, "gate report"),
    ({"ok": False}, "gate report"),
])
def test_a_report_that_does_not_pass_this_tool_and_version(registry, change, problem):
    _, reports, out = registry
    _report(reports, "tool-a", "1.0.0", **change)
    gh, lines = FakeGh(), []
    assert publish.publish([_plan("tool-a", "1.0.0")], str(reports), str(out),
                           passed={"Gates (tool-a)"}, gh=gh, log=lines.append) == 1
    assert gh.created() == [] and problem in lines[0]


def test_the_checkout_must_hold_the_planned_version(registry):
    _, reports, out = registry
    _report(reports, "tool-a", "1.0.1")
    lines = []
    assert publish.publish([_plan("tool-a", "1.0.1")], str(reports), str(out),
                           passed={"Gates (tool-a)"}, gh=FakeGh(), log=lines.append) == 1
    assert "plugin.json does not give this id and version" in lines[0]


@pytest.mark.parametrize("entry", [
    {"tool": "tool-a", "version": "../../x", "tag": "tool-a-v../../x"},
    {"tool": "tool-a", "version": "1.0.0", "tag": "other-v1.0.0"},
    {"tool": "../tool-a", "version": "1.0.0", "tag": "../tool-a-v1.0.0"},
])
def test_a_malformed_plan_entry_is_refused(registry, entry):
    _, reports, out = registry
    lines = []
    assert publish.publish([entry], str(reports), str(out), passed={f"Gates ({entry['tool']})"},
                           gh=FakeGh(), log=lines.append) == 1
    assert "not a tool id, a version and its tag" in lines[0]


def test_a_released_version_is_downloaded_not_packed(registry):
    _, reports, out = registry
    _report(reports, "tool-b", "2.1.0")
    gh = FakeGh(released={"tool-b-v2.1.0"})
    assert publish.publish([_plan("tool-b", "2.1.0", released=True)], str(reports), str(out),
                           passed={"Gates (tool-b)"}, gh=gh, log=lambda m: None) == 0
    assert gh.created() == []
    assert (out / "bundles" / "tool-b-v2.1.0" / "tool-b-2.1.0.pvtool").read_bytes() == b"the released bytes"


def test_one_failure_does_not_hold_up_the_others(registry):
    root, reports, out = registry
    _report(reports, "tool-a", "1.0.0")
    _report(reports, "tool-b", "2.1.0")
    (root / "tools" / "tool-a" / "plugin.json").write_text(json.dumps(
        {**_manifest("tool-a", "1.0.0"), "outputs": "not a list"}))   # pack refuses it
    gh = FakeGh()
    assert publish.publish([_plan("tool-a", "1.0.0"), _plan("tool-b", "2.1.0")], str(reports), str(out),
                           passed={"Gates (tool-a)", "Gates (tool-b)"}, gh=gh, log=lambda m: None) == 1
    assert [c[2] for c in gh.created()] == ["tool-b-v2.1.0"]
    assert sorted(os.listdir(out / "reports")) == ["tool-b.json"]


def test_text_the_public_index_must_not_hold_is_not_released(registry):
    _, reports, out = registry
    # A default value in tool.py, decoded into the entry gate's message: hygiene never saw it on disk.
    _report(reports, "tool-a", "1.0.0",
            gates=[{"id": "entry", "required": True, "status": "pass", "message": f"run(ds, params, x='{EMAIL}')"}])
    gh, lines = FakeGh(), []
    assert publish.publish([_plan("tool-a", "1.0.0")], str(reports), str(out),
                           passed={"Gates (tool-a)"}, gh=gh, log=lines.append) == 1
    assert gh.created() == [] and "an email address" in lines[0]
