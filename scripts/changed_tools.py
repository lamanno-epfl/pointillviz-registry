#!/usr/bin/env python3
"""Which tools CI has to look at, as a JSON list for a job matrix.

    python3 scripts/changed_tools.py BASE [HEAD]                    tools to validate
    python3 scripts/changed_tools.py BASE [HEAD] --examples         examples to validate
    python3 scripts/changed_tools.py BASE [HEAD] --check-versions   the version-bump check
    python3 scripts/changed_tools.py --release                      tools to release

The first form prints the tools the diff BASE..HEAD touches: every folder
under tools/ with a changed file that still exists at HEAD. A change under
sdk/, scripts/ or .github/workflows/ changes how every tool is validated,
packed or indexed, so it means all of them. So does a BASE that cannot be
diffed against: empty, all zeros (the first push of a branch) or a commit
this clone does not have (after a force push). --examples does the same for
the folders under examples/, which are validated but never released.

--check-versions fails when a tool whose own folder changed keeps a version
that is not above the one BASE's index.json lists, or a version that was
released before BASE (its tag is in BASE's history) while the index does not
list it yet. A released bundle never changes, so changed files need a new
version.

--release prints what the release workflow has to handle, as matrix entries
{tool, version, tag, released}: each tool whose tag <id>-v<version> does not
exist yet (released: false), and each tool whose tag exists but whose version
index.json does not list (released: true: the release was made and the index
was not updated, so the released bundle is validated and indexed again). Tags
are read from this clone, so fetch them first.

Only the standard library and git are used, so CI runs this with the runner's
own python3.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: What every tool goes through: the SDK, these scripts and the workflows.
SHARED = ("sdk/", "scripts/", ".github/workflows/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="changed_tools.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("base", nargs="?", help="the commit the change is compared with")
    ap.add_argument("head", nargs="?", default="HEAD", help="the changed commit (default: HEAD)")
    ap.add_argument("--examples", action="store_true",
                    help="list the examples the change touches instead of the tools")
    ap.add_argument("--check-versions", action="store_true",
                    help="fail when a changed tool's version is not above the indexed one")
    ap.add_argument("--release", action="store_true",
                    help="list the tools whose version is not released or not indexed")
    args = ap.parse_args(argv)
    if args.release:
        print(json.dumps(_release_plan()))
        return 0
    if args.base is None:
        ap.error("BASE is required, except with --release.")

    base = args.base if _is_commit(args.base) else None
    if args.check_versions:
        if base is None:
            print(f"No base commit to compare with ({args.base or 'none given'}), so there is "
                  "no version to check.", file=sys.stderr)
            return 0
        return _check_versions(base, args.head)
    folder = "examples" if args.examples else "tools"
    print(json.dumps(_changed(base, args.head, own=False, folder=folder)))
    return 0


def _changed(base, head: str, *, own: bool, folder: str = "tools") -> list:
    """The folders under `folder`/ at `head` that the diff base..head touches.

    `own` counts only changes inside a tool's own folder; otherwise a change
    to SHARED, or no usable base, means every one.
    """
    present = _tools_at(head, folder)
    if base is None:
        return present
    paths = _git("diff", "--name-only", "--no-renames", "-z", base, head, "--").split("\0")
    if not own and any(p.startswith(SHARED) for p in paths):
        return present
    touched = {p.split("/")[1] for p in paths if p.startswith(folder + "/") and p.count("/") >= 2}
    return [tool for tool in present if tool in touched]


def _check_versions(base: str, head: str) -> int:
    try:
        index = json.loads(_git("show", f"{base}:index.json"))
    except subprocess.CalledProcessError:
        index = {}                      # nothing is indexed yet
    listed = {t["id"]: t["version"] for t in index.get("tools", [])}
    failed = False
    for tool in _changed(base, head, own=True):
        try:
            version = str(json.loads(_git("show", f"{head}:tools/{tool}/plugin.json"))["version"])
        except (subprocess.CalledProcessError, ValueError, KeyError, TypeError):
            continue                    # the validation reports a missing or broken manifest
        if tool in listed and not _newer(version, listed[tool]):
            released = listed[tool]
        elif _released_before(f"{tool}-v{version}", base):
            # Released, and the index commit after it was lost: the next release
            # run indexes the released bundle, not these files.
            released = version
        else:
            continue
        failed = True
        # A GitHub annotation on the file; read as plain text elsewhere.
        print(f"::error file=tools/{tool}/plugin.json::tools/{tool} changed, and its version "
              f"{version} is not above {released}, which is already released. A "
              f"released bundle never changes: raise \"version\" in "
              f"tools/{tool}/plugin.json above {released}.")
    return 1 if failed else 0


def _released_before(tag: str, base: str) -> bool:
    """Whether the release tag `tag` exists and points into BASE's history.

    A tag on a later commit is the release of this very change, made while
    this check ran (on a push to main, release.yml runs alongside it).
    """
    ok = subprocess.run(["git", "-C", ROOT, "rev-parse", "-q", "--verify",
                         f"refs/tags/{tag}^{{commit}}"], capture_output=True).returncode == 0
    return ok and subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor",
                                  f"refs/tags/{tag}", base], capture_output=True).returncode == 0


def _release_plan() -> list:
    try:
        with open(os.path.join(ROOT, "index.json"), encoding="utf-8") as fh:
            indexed = {(t["id"], t["version"]) for t in json.load(fh).get("tools", [])}
    except FileNotFoundError:
        indexed = set()
    tags = set(_git("tag", "--list").split())
    folder = os.path.join(ROOT, "tools")
    plan = []
    for tool in sorted(os.listdir(folder) if os.path.isdir(folder) else []):
        if not os.path.isdir(os.path.join(folder, tool)):
            continue
        try:
            with open(os.path.join(folder, tool, "plugin.json"), encoding="utf-8") as fh:
                version = str(json.load(fh)["version"])
        except (OSError, ValueError, KeyError, TypeError):
            # Listed, so that its job fails and says why; it cannot be released.
            plan.append({"tool": tool, "version": "", "tag": "", "released": False})
            continue
        tag = f"{tool}-v{version}"
        if tag not in tags:
            plan.append({"tool": tool, "version": version, "tag": tag, "released": False})
        elif (tool, version) not in indexed:
            plan.append({"tool": tool, "version": version, "tag": tag, "released": True})
    return plan


def _tools_at(rev: str, folder: str = "tools") -> list:
    out = _git("ls-tree", "-d", "--name-only", "-z", rev, folder + "/")
    return sorted(p.split("/", 1)[1] for p in out.split("\0") if p.startswith(folder + "/"))


def _newer(version: str, than: str) -> bool:
    def parts(v):
        return tuple(int(x) for x in v.split(".")) if re.fullmatch(r"\d+\.\d+\.\d+", v) else None
    a, b = parts(version), parts(than)
    return a > b if a and b else version != than


def _is_commit(rev) -> bool:
    if not rev or set(rev) == {"0"}:
        return False
    return subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"{rev}^{{commit}}"],
                          capture_output=True).returncode == 0


def _git(*args) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True,
                          check=True).stdout


if __name__ == "__main__":
    sys.exit(main())
