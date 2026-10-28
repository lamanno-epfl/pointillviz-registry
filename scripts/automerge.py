#!/usr/bin/env python3
"""Merge each pull request that publishes or changes one tool, once its checks pass.

    python3 scripts/automerge.py [--dry-run]

The automerge workflow runs this when `validate` passes on a pull request,
every hour, and on demand. It reads the open pull requests into main through
the GitHub API, runs no code from them, and merges one (squashed, at the very
commit `validate` checked) when:

  - every file it adds, changes, renames or removes is inside one folder
    tools/<id>/, <id> is a valid tool id, and the plugin.json there gives that
    id (unless the pull request removes it);
  - <id> is not reserved for a Coming soon tool (placeholders.json);
  - the tool is new, or the pull request's author is one of its owners in
    owners.json. The author of a new tool becomes its owner. A tool in tools/
    with no owners (the registry's own tools) is changed by maintainers only;
  - it is not a draft, and `validate` passed on its latest commit.

A pull request that changes anything else (the SDK, the scripts, the
workflows, another tool, index.json, owners.json) waits for a maintainer, and
once its checks have passed, a comment says why it was not merged.

A merge made with the workflow's own token starts no workflow, so once it has
merged something, the script starts release.yml itself.

Needs GITHUB_REPOSITORY (owner/name) and, except with --dry-run, GH_TOKEN.
Only the standard library is used, so CI runs this with the runner's own
python3.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
#: The SDK's rule for a tool id (sdk/pointillviz_tool/manifest.py).
ID_RE = re.compile(r"[a-z][a-z0-9-]{1,48}[a-z0-9]")
#: A bundle holds at most 300 files, so a pull request that changes more is not one tool's.
MAX_FILES = 300
#: Marks the comment on a pull request's commit, so that it is written once.
MARK = "<!-- automerge {sha} -->"


class GitHub:
    """The few REST calls this script makes, on one repository."""

    def __init__(self, repo: str, token: str | None):
        self.repo, self.token = repo, token

    def call(self, method: str, path: str, body=None):
        """(status, decoded JSON or None). HTTP errors are returned, not raised."""
        req = urllib.request.Request(f"{API}/repos/{self.repo}/{path}", method=method,
                                     data=None if body is None else json.dumps(body).encode())
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as err:
            raw = err.read()
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, {"message": raw.decode("utf-8", "replace")[:300]}

    def get(self, path: str):
        """The JSON at `path`, or None when it does not exist."""
        status, data = self.call("GET", path)
        if status == 404:
            return None
        if status != 200:
            raise RuntimeError(f"GET {path}: {status} {(data or {}).get('message', '')}")
        return data

    def get_all(self, path: str) -> list:
        """Every item of a paged list."""
        out, page = [], 1
        sep = "&" if "?" in path else "?"
        while True:
            chunk = self.get(f"{path}{sep}per_page=100&page={page}") or []
            out += chunk
            if len(chunk) < 100:
                return out
            page += 1


# --------------------------------------------------------------------------- the rules

def tool_of(paths: list) -> tuple:
    """(the one tool folder every path is in, None), or (None, why not)."""
    outside = [p for p in paths if not re.fullmatch(r"tools/[^/]+/.+", p)]
    if outside:
        return None, f"it changes files outside tools/<id>/: {_few(outside)}"
    tools = sorted({p.split("/")[1] for p in paths})
    if len(tools) != 1:
        return None, f"it changes more than one tool: {', '.join(tools)}"
    if not ID_RE.fullmatch(tools[0]):
        return None, (f"tools/{tools[0]}/ is not named after a valid tool id (lowercase letters, "
                      "digits and hyphens, 3 to 50 characters)")
    return tools[0], None


def refusal(tool: str, author_id: int, *, head_id, removed: bool, on_main: set, owners: dict,
            reserved: set):
    """Why a pull request that changes only tools/<tool>/ cannot be merged, or None.

    `head_id` is the id the pull request's plugin.json gives, and `removed` says
    that the pull request deletes the tool's whole folder, which takes it out of
    the index (its releases stay).
    """
    if tool in reserved:
        return f"the id {tool} is reserved for a tool that is coming soon"
    if not removed and head_id != tool:
        if head_id is None:
            return f"tools/{tool}/plugin.json is missing or not a JSON object with an id"
        return f"tools/{tool}/plugin.json gives the id {head_id!r}, but a tool lives in tools/<its id>/"
    listed = owners.get(tool) or []
    if listed:
        if author_id not in {o.get("id") for o in listed}:
            names = ", ".join("@" + str(o.get("login")) for o in listed)
            return f"tools/{tool}/ belongs to {names}, and only its owners can change it"
    elif tool in on_main:
        return f"tools/{tool}/ is one of the registry's own tools, which its maintainers change"
    elif removed:
        return f"tools/{tool}/ is not in the registry, so there is nothing to remove"
    return None


def _few(paths: list) -> str:
    shown = ", ".join(sorted(paths)[:3])
    return shown + (f" and {len(paths) - 3} more" if len(paths) > 3 else "")


# --------------------------------------------------------------------------- reading GitHub

def _decode(content: dict):
    return base64.b64decode(content["content"]) if content and content.get("content") else None


def load_state(gh: GitHub) -> dict:
    """What main holds now: its tool folders, the owners, and the reserved ids.

    The folder listing holds at most 1,000 entries; past that many tools, read
    the tree through the git trees API instead.
    """
    listing = gh.get("contents/tools?ref=main") or []
    owners_file = gh.get("contents/owners.json?ref=main")
    raw = _decode(owners_file)
    placeholders = json.loads(_decode(gh.get("contents/placeholders.json?ref=main")) or b"[]")
    return {"on_main": {e["name"] for e in listing if e.get("type") == "dir"},
            "owners": json.loads(raw) if raw else {},
            "owners_sha": owners_file["sha"] if owners_file else None,
            "reserved": {p["id"] for p in placeholders}}


def head_manifest(gh: GitHub, tool: str, sha: str) -> tuple:
    """(the id that tools/<tool>/plugin.json gives at `sha`, whether the whole folder is gone there)."""
    content = gh.get(f"contents/tools/{tool}/plugin.json?ref={sha}")
    if content is None:
        return None, gh.get(f"contents/tools/{tool}?ref={sha}") is None
    try:
        manifest = json.loads(_decode(content) or b"")
    except ValueError:
        return None, False
    pid = manifest.get("id") if isinstance(manifest, dict) else None
    return pid if isinstance(pid, str) else None, False


def validate_run(gh: GitHub, sha: str):
    """The latest run of validate.yml for this commit of a pull request, or None."""
    data = gh.get(f"actions/workflows/validate.yml/runs?head_sha={sha}&event=pull_request&per_page=10")
    runs = (data or {}).get("workflow_runs") or []
    return max(runs, key=lambda r: (r.get("created_at") or "", r.get("id") or 0)) if runs else None


# --------------------------------------------------------------------------- one pass

def sweep(gh: GitHub, *, dry_run: bool = False, trigger_sha: str | None = None, log=print) -> int:
    """Merge what qualifies; 0 when every pull request was handled, 1 after an error."""
    state = load_state(gh)
    merged, errors = [], 0
    for pr in sorted(gh.get_all("pulls?state=open&base=main"), key=lambda p: p["number"]):
        try:
            done = _handle(gh, pr, state, dry_run=dry_run, trigger_sha=trigger_sha, log=log)
        except Exception as err:            # one pull request must not hold up the others
            log(f"::error::#{pr['number']}: {err}")
            errors += 1
            continue
        if done:
            merged.append(done)
    if merged and not dry_run:
        status, data = gh.call("POST", "actions/workflows/release.yml/dispatches", {"ref": "main"})
        if status != 204:
            log(f"::error::merged {', '.join(merged)}, but release.yml did not start: "
                f"{status} {(data or {}).get('message', '')}")
            return 1
        log(f"release.yml started for {', '.join(merged)}")
    return 1 if errors else 0


def _handle(gh: GitHub, pr: dict, state: dict, *, dry_run: bool, trigger_sha, log):
    """Merge one pull request if it qualifies; its tool's id when it was merged."""
    n, sha = pr["number"], pr["head"]["sha"]
    if pr.get("draft"):
        log(f"#{n}: a draft, left alone")
        return None
    run = validate_run(gh, sha)
    if run is None or run.get("status") != "completed":
        log(f"#{n}: validate has not finished on {sha[:7]}")
        return None
    if run.get("conclusion") != "success":
        log(f"#{n}: validate did not pass on {sha[:7]} ({run.get('conclusion')})")
        return None

    files = gh.get_all(f"pulls/{n}/files")
    paths = [p for f in files for p in (f["filename"], f.get("previous_filename")) if p]
    tool, reason = (None, f"it changes {len(files)} files, more than a tool may hold") \
        if len(files) > MAX_FILES else tool_of(paths)
    if tool:
        head_id, removed = head_manifest(gh, tool, sha)
        reason = refusal(tool, pr["user"]["id"], head_id=head_id, removed=removed,
                         on_main=state["on_main"], owners=state["owners"], reserved=state["reserved"])
    if reason:
        log(f"#{n}: left for a maintainer: {reason}")
        if sha == trigger_sha and not dry_run:
            _explain(gh, n, sha, reason)
        return None
    if dry_run:
        log(f"#{n}: would merge {tool} at {sha[:7]}")
        return None

    # The owner first: a merge without its owner would leave the tool to the
    # maintainers. If the merge then fails, the author keeps the id for a retry.
    if not state["owners"].get(tool):
        _record_owner(gh, state, tool, pr["user"])
    status, data = gh.call("PUT", f"pulls/{n}/merge", {"sha": sha, "merge_method": "squash"})
    if status != 200:
        # 409: a commit was pushed after validate ran; its own run decides. 405: not mergeable.
        log(f"#{n}: not merged: {status} {(data or {}).get('message', '')}")
        return None
    log(f"#{n}: merged {tool} at {sha[:7]}")
    if removed:
        state["on_main"].discard(tool)
    else:
        state["on_main"].add(tool)
    return tool


def _record_owner(gh: GitHub, state: dict, tool: str, user: dict) -> None:
    """Make the author of a new tool its owner, in owners.json on main, before it is merged.

    A maintainer may have changed owners.json since this pass read it; then the
    file is read again and the write retried once.
    """
    for attempt in (1, 2):
        if state["owners"].get(tool):
            raise RuntimeError(f"tools/{tool}/ was claimed by someone else meanwhile; not merged")
        owners = dict(state["owners"], **{tool: [{"login": user["login"], "id": user["id"]}]})
        text = json.dumps(owners, indent=2, sort_keys=True) + "\n"
        body = {"message": f"Record the owner of {tool} [skip ci]", "branch": "main",
                "content": base64.b64encode(text.encode()).decode("ascii")}
        if state["owners_sha"]:
            body["sha"] = state["owners_sha"]
        status, data = gh.call("PUT", "contents/owners.json", body)
        if status in (200, 201):
            state["owners"], state["owners_sha"] = owners, data["content"]["sha"]
            return
        if attempt == 1 and status in (409, 422):
            fresh = load_state(gh)
            state["owners"], state["owners_sha"] = fresh["owners"], fresh["owners_sha"]
            continue
        raise RuntimeError(f"the owner of {tool} could not be recorded, so it was not merged: "
                           f"{status} {(data or {}).get('message', '')}")


def _explain(gh: GitHub, n: int, sha: str, reason: str) -> None:
    """Say once per commit why a pull request whose checks passed was not merged."""
    mark = MARK.format(sha=sha)
    if any(mark in (c.get("body") or "") for c in gh.get_all(f"issues/{n}/comments")):
        return
    gh.call("POST", f"issues/{n}/comments", {"body": (
        f"{mark}\nThe checks passed, but this pull request was not merged automatically: {reason}.\n\n"
        "A pull request is merged automatically when it adds or changes one tool, in its folder "
        "`tools/<id>/` and nowhere else, and the tool is new or yours (see Publishing in "
        "CONTRIBUTING.md). A maintainer can still merge this one.")})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="say what would be merged, and change nothing")
    args = ap.parse_args(argv)
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not repo:
        ap.error("GITHUB_REPOSITORY is not set (owner/name)")
    if not token and not args.dry_run:
        ap.error("GH_TOKEN is not set; --dry-run reads a public repository without one")
    return sweep(GitHub(repo, token), dry_run=args.dry_run,
                 trigger_sha=os.environ.get("TRIGGER_SHA") or None)


if __name__ == "__main__":
    sys.exit(main())
