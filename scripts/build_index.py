#!/usr/bin/env python3
"""Build the registry's index.json: each tool's latest version, and the Coming soon placeholders.

    python scripts/build_index.py --bundles DIR [--reports DIR] [--base-url URL]
                                  [--raw-base URL] [--out PATH]

One entry per folder tools/<id>/, for the version its plugin.json names. The
SDK reads the manifest, so an entry, and the `manifest` it carries, hold the
manifest as PointillViz itself reads it, with every default filled in.

A released bundle never changes. A version that the existing index (the file
at --out) already lists keeps its entry as it was, the bundle's sha256 and
size, the gate report, the date and the manifest included, and is never
packed again: packing the same files again would not give the same bytes,
because a zip records each file's time and a fresh checkout gives every file a
new one. Only its links (bundle, README, preview and source) are rebuilt. Only
a new version is hashed, from its bundle in DIR, which is laid out as the
registry's GitHub releases are (<DIR>/<id>-v<version>/<id>-<version>.pvtool).
A bundle that is not there is packed there first.

With --reports, a new version needs its gate report (<reports>/<id>.json, as
`python -m pointillviz_tool validate --json` writes it), and it is listed only
when that report is ok. Otherwise the index keeps the version it listed before,
if there was one: a tool that fails is not released, and it does not hold up
the others. Without --reports, new versions are listed with no report.

URLs may be relative: PointillViz reads them against the index's own location,
a URL or a folder. A relative --base-url must lead from the index's folder to
DIR. With a relative --raw-base, every listed tool's files are copied to
<index folder>/<raw-base>/tools/<id>/, so that a local registry folder (the
index, its bundles and the tools' READMEs) works offline.

The output is deterministic: tools in id order, keys in a fixed order, and the
same inputs give the same bytes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sdk"))

from pointillviz_tool import bundle, manifest  # noqa: E402  (the SDK of this checkout)
import check_hygiene  # noqa: E402  (beside this file)

REPOSITORY = "https://github.com/lamanno-epfl/pointillviz-registry"
BASE_URL = REPOSITORY + "/releases/download"
RAW_BASE = "https://raw.githubusercontent.com/lamanno-epfl/pointillviz-registry/main"
_GITHUB_RAW = re.compile(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/(.+)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_index.py",
                                 description="Build the registry's index.json.")
    ap.add_argument("--bundles", required=True,
                    help="bundles as <DIR>/<id>-v<version>/<id>-<version>.pvtool; a new "
                         "version's bundle is packed here when it is missing")
    ap.add_argument("--reports", default=None,
                    help="gate reports as <DIR>/<id>.json; with them, a new version is listed "
                         "only when its report is ok")
    ap.add_argument("--base-url", default=BASE_URL,
                    help=f"where bundles are downloaded from (default: {BASE_URL})")
    ap.add_argument("--raw-base", default=RAW_BASE,
                    help=f"where README.md, preview.svg and the source are read from "
                         f"(default: {RAW_BASE})")
    ap.add_argument("--out", default=os.path.join(ROOT, "index.json"),
                    help="the index to write; the versions it already lists are carried over "
                         "(default: index.json in the registry)")
    args = ap.parse_args(argv)
    base_url, raw_base = args.base_url.rstrip("/"), args.raw_base.rstrip("/")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not _absolute(base_url) and (os.path.realpath(os.path.join(out_dir, base_url))
                                    != os.path.realpath(args.bundles)):
        ap.error(f"a relative --base-url is read against the index's folder, so --bundles must "
                 f"be {os.path.join(out_dir, base_url)}.")

    previous = _previous(args.out)
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    folder = os.path.join(ROOT, "tools")
    tools = []
    for tool in sorted(os.listdir(folder) if os.path.isdir(folder) else []):
        tool_dir = os.path.join(folder, tool)
        if not os.path.isdir(tool_dir):
            continue
        entry = _entry(tool_dir, tool, previous.get(tool), args, base_url, raw_base, today)
        if entry is None:
            continue
        tools.append(entry)
        if not _absolute(raw_base):
            _copy_tool(tool_dir, os.path.join(out_dir, raw_base, "tools", tool))

    index = {"schema": 1, "registry": _registry(raw_base), "tools": tools,
             "placeholders": _placeholders()}
    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, args.out)
    print(f"{args.out}: {len(tools)} tools, {len(index['placeholders'])} placeholders",
          file=sys.stderr)
    return 0


def _entry(tool_dir, tool, prev, args, base_url, raw_base, today):
    """The index entry for one tool folder, or the previous one when its version cannot be listed."""
    def keep(why):
        kept = f"; keeping {prev['version']}" if prev else "; it is left out"
        print(f"{tool}: {why}{kept}", file=sys.stderr)
        return prev

    try:
        m = manifest.load(tool_dir)
    except Exception as exc:            # any malformed plugin.json: the other tools go on
        return keep(f"plugin.json cannot be read ({exc})")
    if m["id"] != tool:
        return keep(f"plugin.json gives the id {m['id']!r}; a tool lives in tools/<its id>/")
    tag, name = f"{m['id']}-v{m['version']}", f"{m['id']}-{m['version']}.pvtool"

    files = {rel for rel, _full in bundle.files(tool_dir)}
    links = {
        "readme_url": f"{raw_base}/tools/{tool}/README.md" if "README.md" in files else None,
        "preview_url": f"{raw_base}/tools/{tool}/preview.svg" if "preview.svg" in files else None,
        "source_url": _source_url(raw_base, tool),
    }
    if prev and prev["version"] == m["version"]:
        # Released: the entry stays the one that describes the released bundle,
        # never packed again. Only its links may move. Files changed without a
        # new version are not released (validate.yml's version check says so),
        # so nothing here is read from them.
        print(f"{tool} {m['version']}: carried over", file=sys.stderr)
        return dict(prev, bundle=dict(prev["bundle"], url=f"{base_url}/{tag}/{name}"), **links)

    report = None
    if args.reports:
        report = _report(os.path.join(args.reports, f"{tool}.json"), m)
        if report is None:
            return keep(f"{m['version']} has no passing gate report")
    path = os.path.join(args.bundles, tag, name)
    try:
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            bundle.pack(tool_dir, path)
        with open(path, "rb") as fh:
            data = fh.read()
        _check_bundle(data, m)
    except (bundle.BundleError, manifest.ManifestError, ValueError, KeyError) as exc:
        return keep(f"the bundle of {m['version']} is unusable ({exc})")
    sha256, size, published = hashlib.sha256(data).hexdigest(), len(data), today
    entry = {
        "id": m["id"],
        "name": m["name"],
        "version": m["version"],
        "description": m["description"],
        "author": m["author"],
        "icon": m["icon"],
        "tags": m["tags"],
        "license": m["license"],
        "citation": m["citation"],
        "language": manifest.language(m),
        "gpu": m["gpu"],
        "outputs": m["outputs"],
        "optional_outputs": m["optional_outputs"],
        "requires": m["requires"],
        "manifest": m,
        "bundle": {"url": f"{base_url}/{tag}/{name}", "sha256": sha256, "size": size},
        **links,
        "report": report,
        "published": published,
    }
    # Its text is its author's, merged without review: what the hygiene check
    # refuses would stop index.json from being committed for every tool.
    for _line, what, match in check_hygiene.findings_in(json.dumps(entry, ensure_ascii=False),
                                                         "index.json"):
        return keep(f"its entry would put {what} in the public index ({match[:40]})")
    print(f"{tool} {m['version']}: new, {size:,} bytes", file=sys.stderr)
    return entry


def _report(path: str, m: dict):
    """The gates of an ok report for exactly this version, trimmed for the index, else None."""
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError):
        return None
    if not (report.get("ok") is True and report.get("plugin_id") == m["id"]
            and report.get("version") == m["version"]):
        return None
    return {"ok": True,
            "gates": [{"id": g["id"], "required": g["required"], "status": g["status"],
                       "message": g["message"]} for g in report["gates"]],
            "risks": list(report.get("risks") or [])}


def _check_bundle(data: bytes, m: dict) -> None:
    """A bundle the host will open, holding this tool at this version."""
    bundle.inspect(data)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        inside = json.loads(zf.read("plugin.json"))
    if (inside.get("id"), inside.get("version")) != (m["id"], m["version"]):
        raise ValueError(f"it holds {inside.get('id')} {inside.get('version')}")


def _previous(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return {t["id"]: t for t in json.load(fh)["tools"]}
    except FileNotFoundError:
        return {}


def _placeholders() -> list:
    """placeholders.json: tools listed as Coming soon, each its id, name and one line."""
    try:
        with open(os.path.join(ROOT, "placeholders.json"), encoding="utf-8") as fh:
            items = json.load(fh)
    except FileNotFoundError:
        return []
    keys = ("id", "name", "description")
    if not isinstance(items, list) or not all(
            isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k].strip() for k in keys)
            for p in items):
        sys.exit("placeholders.json must be a list of objects, each with a non-empty id, name "
                 "and description.")
    return [{k: p[k] for k in keys} for p in items]


def _copy_tool(tool_dir: str, dest: str) -> None:
    """A tool's own files (as `pack` sees them) at `dest`, for a local registry folder."""
    if os.path.isdir(dest) and os.path.samefile(tool_dir, dest):
        return
    shutil.rmtree(dest, ignore_errors=True)
    for name, full in bundle.files(tool_dir):
        target = os.path.join(dest, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(full, target)


def _registry(raw_base: str) -> str:
    found = _GITHUB_RAW.fullmatch(raw_base)
    return f"https://github.com/{found[1]}/{found[2]}" if found else REPOSITORY


def _source_url(raw_base: str, tool: str) -> str:
    """The tool's folder: on GitHub its tree page, otherwise the folder beside the README."""
    found = _GITHUB_RAW.fullmatch(raw_base)
    if found:
        return f"https://github.com/{found[1]}/{found[2]}/tree/{found[3]}/tools/{tool}"
    return f"{raw_base}/tools/{tool}/"


def _absolute(url: str) -> bool:
    return re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", url) is not None


if __name__ == "__main__":
    sys.exit(main())
