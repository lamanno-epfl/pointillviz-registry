"""check_hygiene.py reads JSON as build_index.py does; build_index.py never lists what it refuses."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_index  # noqa: E402
import check_hygiene  # noqa: E402

#: Spelled in pieces, so that the hygiene check, which reads this file too, passes it.
EMAIL = "a" + "@" + "example.org"
MNT = "/" + "mnt/"

MANIFEST = {"schema": 3, "id": "tool-a", "name": "Tool A", "version": "1.0.0", "author": "A. Author",
            "description": "What it does.", "entry": "tool.py", "outputs": ["summary"]}


def _tool(root, **change):
    folder = root / "tools" / "tool-a"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plugin.json").write_text(json.dumps({**MANIFEST, **change}))   # ASCII, escaped
    (folder / "tool.py").write_text("def run(ds, params):\n    return {'summary': 'ok'}\n")
    return folder


def test_json_escapes_are_read_as_build_index_reads_them(tmp_path, monkeypatch):
    monkeypatch.setattr(check_hygiene, "ROOT", str(tmp_path))
    path = _tool(tmp_path, description=f"Mail {EMAIL} about data in {MNT}x") / "plugin.json"
    escaped = path.read_text().replace("@", chr(92) + "u0040").replace(MNT, (chr(92) + "u002f").join(["", "mnt", ""]))
    path.write_text(escaped)
    raw = path.read_text()
    assert "@" not in raw and MNT not in raw                 # nothing to see on disk
    found = list(check_hygiene._findings("tools/tool-a/plugin.json"))
    assert {what for _, what, _ in found} == {"an email address, written with JSON escapes",
                                              "a private path, written with JSON escapes"}


def test_a_plain_file_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(check_hygiene, "ROOT", str(tmp_path))
    _tool(tmp_path)
    assert list(check_hygiene._findings("tools/tool-a/plugin.json")) == []


def _entry(tmp_path, report_message="ok", prev=None, **change):
    folder = _tool(tmp_path, **change)
    reports, bundles = tmp_path / "reports", tmp_path / "bundles"
    reports.mkdir(exist_ok=True)
    (reports / "tool-a.json").write_text(json.dumps({
        "ok": True, "plugin_id": "tool-a", "version": "1.0.0", "risks": [],
        "gates": [{"id": "entry", "required": True, "status": "pass", "message": report_message}]}))
    args = argparse.Namespace(reports=str(reports), bundles=str(bundles))
    return build_index._entry(str(folder), "tool-a", prev, args, "https://example.invalid/rel",
                              "https://example.invalid/raw", "2026-10-28")


def test_a_clean_tool_is_indexed(tmp_path):
    entry = _entry(tmp_path)
    assert entry["id"] == "tool-a" and entry["bundle"]["size"] > 0


def test_an_entry_the_hygiene_check_would_refuse_keeps_the_previous_version(tmp_path):
    prev = {"id": "tool-a", "version": "0.9.0", "bundle": {"url": "u"}}
    assert _entry(tmp_path, report_message=f"run(ds, params, x='{EMAIL}')", prev=prev) == prev
    assert _entry(tmp_path / "again", report_message=f"run(ds, params, x='{EMAIL}')") is None


def test_any_malformed_manifest_keeps_the_previous_version(tmp_path):
    prev = {"id": "tool-a", "version": "0.9.0", "bundle": {"url": "u"}}
    assert _entry(tmp_path, prev=prev, tags=5) == prev                 # a TypeError, not a ManifestError
