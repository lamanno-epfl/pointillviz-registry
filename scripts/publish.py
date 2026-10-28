#!/usr/bin/env python3
"""Release the planned tools whose gates passed: the publish job of release.yml.

    python scripts/publish.py REPORTS OUT

The plan job's list of {tool, version, tag, released} comes in the PLAN
environment variable. A planned tool is released only when

  - its job "Gates (<id>)" in this workflow run succeeded, as GitHub's API
    reports it, so that no job can vouch for another's tool, and
  - the report that job uploaded, REPORTS/report-<id>/report.json, passes for
    this id and version, and so does the plugin.json of this checkout.

The bundle is never one a gates job made: tools run in those jobs, as
unreviewed code, and could rewrite whatever lies beside them. A new version is
packed here, from this checkout, by this checkout's SDK; a version released by
an earlier run whose index commit was lost is downloaded from its release.
OUT receives bundles/<tag>/<id>-<version>.pvtool and reports/<id>.json, which
scripts/build_index.py reads.

Runs in the SDK's environment (for pack), with the gh CLI and the workflow's
GH_TOKEN, GH_REPO, GITHUB_RUN_ID and SHA (the commit to release).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sdk"))

from pointillviz_tool import bundle  # noqa: E402  (the SDK of this checkout)
import check_hygiene  # noqa: E402  (beside this file)

ID_RE = re.compile(r"[a-z][a-z0-9-]{1,48}[a-z0-9]")
SEMVER_RE = re.compile(r"\d+\.\d+\.\d+")


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def passed_jobs() -> set:
    """The names of this run's jobs that succeeded, in their latest attempt."""
    out = gh("api", "--paginate", f"repos/{os.environ['GH_REPO']}/actions/runs/"
             f"{os.environ['GITHUB_RUN_ID']}/jobs?filter=latest",
             "--jq", '.jobs[] | select(.conclusion == "success") | .name')
    return set(out.splitlines())


def publish(plan: list, reports: str, out: str, *, passed: set, gh=gh, log=print) -> int:
    """Release and stage every planned tool that qualifies; 1 if one could not be."""
    failed = 0
    for entry in plan:
        tool, version, tag = entry.get("tool", ""), entry.get("version", ""), entry.get("tag", "")
        if f"Gates ({tool})" not in passed:
            continue                    # its gates job failed and said why
        problem = _problem(tool, version, tag, reports)
        if problem:
            log(f"::error::{tool} {version} is not released: {problem}")
            failed = 1
            continue
        with open(os.path.join(ROOT, "tools", tool, "plugin.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        with open(os.path.join(reports, f"report-{tool}", "report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        folder = os.path.join(out, "bundles", tag)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{tool}-{version}.pvtool")
        try:
            if entry.get("released"):
                gh("release", "download", tag, "--pattern", f"{tool}-{version}.pvtool", "--dir", folder)
            else:
                bundle.pack(os.path.join(ROOT, "tools", tool), path)
                if not _already_released(tag, gh):
                    with open(os.path.join(out, "notes.md"), "w", encoding="utf-8") as fh:
                        fh.write(_notes(manifest, report, tool, version))
                    gh("release", "create", tag, path, "--target", os.environ["SHA"],
                       "--title", f"{manifest['name']} {version}",
                       "--notes-file", os.path.join(out, "notes.md"))
        except Exception as err:        # one tool's failure must not hold up the others
            detail = getattr(err, "stderr", "") or str(err)
            log(f"::error::{tool} {version} could not be released, so it is not indexed: {detail}")
            failed = 1
            continue
        os.makedirs(os.path.join(out, "reports"), exist_ok=True)
        shutil.copyfile(os.path.join(reports, f"report-{tool}", "report.json"),
                        os.path.join(out, "reports", f"{tool}.json"))
        log(f"{tool} {version}: {'indexed again' if entry.get('released') else 'released'}")
    return failed


def _problem(tool: str, version: str, tag: str, reports: str):
    """Why this planned tool cannot be released, or None."""
    if not ID_RE.fullmatch(tool) or not SEMVER_RE.fullmatch(version) or tag != f"{tool}-v{version}":
        return f"the plan entry {tool!r} {version!r} {tag!r} is not a tool id, a version and its tag"
    try:
        with open(os.path.join(ROOT, "tools", tool, "plugin.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        with open(os.path.join(reports, f"report-{tool}", "report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError) as err:
        return f"its plugin.json or gate report cannot be read ({err})"
    if not isinstance(manifest, dict) or (manifest.get("id"), manifest.get("version")) != (tool, version):
        return "plugin.json does not give this id and version"
    if not isinstance(report, dict) or (report.get("ok"), report.get("plugin_id"), report.get("version")) \
            != (True, tool, version):
        return "its gate report does not pass this id and version"
    # The release notes and the index entry are made of these; build_index.py would refuse them.
    for _line, what, match in check_hygiene.findings_in(
            json.dumps({"manifest": manifest, "report": report}, ensure_ascii=False), "index.json"):
        return f"its manifest or gate report holds {what} ({match[:40]})"
    return None


def _already_released(tag: str, gh) -> bool:
    try:
        gh("release", "view", tag)
        return True
    except subprocess.CalledProcessError:
        return False


def _notes(manifest: dict, report: dict, tool: str, version: str) -> str:
    kind = "An R tool" if report.get("language") == "r" else "A Python tool"
    gpu = f', with "gpu": "{report["gpu"]}"' if report.get("gpu") else ""
    passed = [g["id"] for g in report.get("gates", []) if g.get("required") and g.get("status") == "pass"]
    return (f"{manifest.get('description', '')}\n\n{kind} for PointillViz{gpu}. It passed every "
            f"required gate: {', '.join(passed)}.\n\nInstall it in PointillViz: Tools → Store → "
            f"Install. Or download {tool}-{version}.pvtool below and use Install from file.\n")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    reports, out = argv
    return publish(json.loads(os.environ.get("PLAN") or "[]"), reports, out, passed=passed_jobs())


if __name__ == "__main__":
    sys.exit(main())
