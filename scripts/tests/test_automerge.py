"""scripts/automerge.py: which pull requests are merged, against a GitHub held in memory."""

import base64
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import automerge  # noqa: E402
from automerge import refusal, sweep, tool_of  # noqa: E402

ALICE = {"login": "alice", "id": 1}
BOB = {"login": "bob", "id": 2}


# --------------------------------------------------------------------------- the rules

def test_one_tool_folder_and_nothing_else():
    assert tool_of(["tools/my-tool/tool.py", "tools/my-tool/plugin.json"]) == ("my-tool", None)
    assert tool_of(["tools/my-tool/sub/helper.py"]) == ("my-tool", None)
    for paths in (["tools/my-tool/tool.py", "sdk/pointillviz_tool/validate.py"],
                  ["tools/my-tool/tool.py", ".github/workflows/release.yml"],
                  ["tools/my-tool/tool.py", "owners.json"],
                  ["index.json"], ["tools/stray-file"]):
        tool, reason = tool_of(paths)
        assert tool is None and "outside tools/<id>/" in reason
    tool, reason = tool_of(["tools/my-tool/tool.py", "tools/banksy-niches/tool.py"])
    assert tool is None and "more than one tool" in reason
    tool, reason = tool_of(["tools/My_Tool/tool.py"])
    assert tool is None and "valid tool id" in reason


def _refusal(tool="my-tool", author=1, head_id="my-tool", removed=False, on_main=(), owners=None,
             reserved=()):
    return refusal(tool, author, head_id=head_id, removed=removed, on_main=set(on_main),
                   owners=owners or {}, reserved=set(reserved))


def test_a_new_tool_is_anyone_s():
    assert _refusal() is None


def test_an_owned_tool_is_its_owners_only():
    owners = {"my-tool": [ALICE]}
    assert _refusal(author=1, on_main=["my-tool"], owners=owners) is None
    assert "only its owners" in _refusal(author=2, on_main=["my-tool"], owners=owners)
    assert "@alice" in _refusal(author=2, on_main=["my-tool"], owners=owners)
    # A removed tool keeps its owners, so nobody else can take its id over.
    assert "only its owners" in _refusal(author=2, owners=owners)


def test_the_registry_s_own_tools_are_left_to_maintainers():
    assert "registry's own tools" in _refusal(tool="banksy-niches", head_id="banksy-niches",
                                              on_main=["banksy-niches"])


def test_the_manifest_id_must_be_the_folder_s():
    assert "gives the id 'banksy-niches'" in _refusal(head_id="banksy-niches")
    assert "missing or not a JSON object" in _refusal(head_id=None)


def test_reserved_ids_cannot_be_taken():
    assert "reserved" in _refusal(tool="coming-soon", head_id="coming-soon", reserved=["coming-soon"])


def test_removing_a_tool():
    owners = {"my-tool": [ALICE]}
    assert _refusal(head_id=None, removed=True, on_main=["my-tool"], owners=owners) is None
    assert "only its owners" in _refusal(author=2, head_id=None, removed=True, on_main=["my-tool"],
                                         owners=owners)
    assert "nothing to remove" in _refusal(head_id=None, removed=True)


# --------------------------------------------------------------------------- a pass over GitHub

class FakeGitHub(automerge.GitHub):
    """Just enough of GitHub's REST API for the script."""

    def __init__(self, *, tools=(), owners=None, reserved=()):
        super().__init__("lab/registry", "token")
        self.tools = set(tools)
        self.owners = owners
        self.owners_sha = "o0" if owners is not None else None
        self.reserved = list(reserved)
        self.prs, self.files, self.heads, self.runs = {}, {}, {}, {}
        self.comments, self.merged, self.dispatched = {}, [], 0
        self.merge_status, self.broken = {}, set()
        self.owner_conflicts = 0

    def add_pr(self, n, user, files, *, head_files=None, run="success", draft=False, sha=None):
        """`files` are the changed paths; `head_files` what tools/ holds at the head commit."""
        sha = sha or f"sha{n}"
        self.prs[n] = {"number": n, "user": user, "draft": draft, "head": {"sha": sha}}
        self.files[n] = [{"filename": f, "status": "modified"} for f in files]
        self.heads[sha] = head_files if head_files is not None else {
            f: json.dumps({"id": f.split("/")[1]}) for f in files if f.endswith("plugin.json")}
        if run:
            self.runs[sha] = {"id": n, "created_at": "t", "status": "completed",
                              "conclusion": run} if run != "pending" else {"id": n, "status": "in_progress"}
        return sha

    def call(self, method, path, body=None):
        page = re.search(r"[?&]page=(\d+)", path)
        path = re.sub(r"[?&](per_page|page)=\d+", "", path)
        if page and page.group(1) != "1":
            return 200, []              # everything fits on the first page
        m = re.fullmatch(r"pulls/(\d+)/files", path)
        if m:
            if int(m.group(1)) in self.broken:
                return 500, {"message": "boom"}
            return 200, self.files[int(m.group(1))]
        if method == "GET" and path == "pulls?state=open&base=main":
            return 200, [pr for n, pr in self.prs.items() if n not in self.merged]
        if path == "contents/tools?ref=main":
            return 200, [{"name": t, "type": "dir"} for t in sorted(self.tools)]
        if path == "contents/owners.json?ref=main":
            if self.owners is None:
                return 404, None
            return 200, {"sha": self.owners_sha, "content": _b64(json.dumps(self.owners))}
        if path == "contents/placeholders.json?ref=main":
            return 200, {"sha": "p", "content": _b64(json.dumps([{"id": i} for i in self.reserved]))}
        m = re.fullmatch(r"contents/(tools/[^?]+)\?ref=(\w+)", path)
        if m:
            held = self.heads.get(m.group(2), {})
            if m.group(1) in held:
                return 200, {"sha": "x", "content": _b64(held[m.group(1)])}
            if any(f.startswith(m.group(1) + "/") for f in held):
                return 200, [{"name": "a-file"}]
            return 404, {"message": "Not Found"}
        m = re.fullmatch(r"actions/workflows/validate\.yml/runs\?head_sha=(\w+)&event=pull_request", path)
        if m:
            run = self.runs.get(m.group(1))
            return 200, {"workflow_runs": [run] if run else []}
        m = re.fullmatch(r"pulls/(\d+)/merge", path)
        if m and method == "PUT":
            n = int(m.group(1))
            if n in self.merge_status:
                return self.merge_status[n], {"message": "Head branch was modified"}
            assert body == {"sha": self.prs[n]["head"]["sha"], "merge_method": "squash"}
            self.merged.append(n)
            return 200, {"merged": True}
        if path == "contents/owners.json" and method == "PUT":
            if self.owner_conflicts:
                self.owner_conflicts -= 1
                self.owners_sha = f"{self.owners_sha}+"     # someone else's commit
                return 409, {"message": "is at another sha"}
            assert body.get("sha") == self.owners_sha
            self.owners = json.loads(base64.b64decode(body["content"]))
            self.owners_sha = f"o{len(self.merged)}"
            return (201 if "sha" not in body else 200), {"content": {"sha": self.owners_sha}}
        m = re.fullmatch(r"issues/(\d+)/comments", path)
        if m:
            n = int(m.group(1))
            if method == "POST":
                self.comments.setdefault(n, []).append(body["body"])
                return 201, {}
            return 200, [{"body": b} for b in self.comments.get(n, [])]
        if path == "actions/workflows/release.yml/dispatches" and method == "POST":
            assert body == {"ref": "main"}
            self.dispatched += 1
            return 204, None
        raise AssertionError(f"unexpected call {method} {path}")


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _files(tool, *names):
    return [f"tools/{tool}/{n}" for n in names or ("plugin.json", "tool.py")]


def test_a_new_tool_is_merged_its_author_owns_it_and_the_release_starts():
    gh = FakeGitHub(tools=["banksy-niches"])
    gh.add_pr(7, ALICE, _files("my-tool"))
    assert sweep(gh, log=lambda m: None) == 0
    assert gh.merged == [7]
    assert gh.owners == {"my-tool": [ALICE]}
    assert gh.dispatched == 1


def test_an_owner_s_update_is_merged_and_a_stranger_s_is_explained_once():
    gh = FakeGitHub(tools=["my-tool"], owners={"my-tool": [ALICE]})
    gh.add_pr(1, ALICE, _files("my-tool"))
    stranger = gh.add_pr(2, BOB, _files("my-tool"))
    assert sweep(gh, trigger_sha=stranger, log=lambda m: None) == 0
    assert gh.merged == [1]
    assert gh.owners == {"my-tool": [ALICE]}        # unchanged, and not written again
    assert len(gh.comments[2]) == 1 and "only its owners can change it" in gh.comments[2][0]
    sweep(gh, trigger_sha=stranger, log=lambda m: None)
    assert len(gh.comments[2]) == 1                  # once per commit


def test_only_the_triggering_commit_gets_a_comment():
    gh = FakeGitHub()
    gh.add_pr(3, ALICE, ["sdk/pointillviz_tool/validate.py"])
    sweep(gh, trigger_sha="another-commit", log=lambda m: None)
    assert gh.comments == {} and gh.merged == []


def test_drafts_and_unfinished_or_failed_checks_wait():
    gh = FakeGitHub()
    gh.add_pr(1, ALICE, _files("tool-a"), draft=True)
    gh.add_pr(2, ALICE, _files("tool-b"), run="pending")
    gh.add_pr(3, ALICE, _files("tool-c"), run="failure")
    gh.add_pr(4, ALICE, _files("tool-d"), run=None)
    assert sweep(gh, log=lambda m: None) == 0
    assert gh.merged == [] and gh.dispatched == 0 and gh.comments == {}


def test_two_claims_on_one_new_id_in_one_pass():
    gh = FakeGitHub()
    gh.add_pr(1, ALICE, _files("my-tool"))
    second = gh.add_pr(2, BOB, _files("my-tool", "tool.R"), head_files={})
    sweep(gh, trigger_sha=second, log=lambda m: None)
    assert gh.merged == [1]
    assert gh.owners == {"my-tool": [ALICE]}
    assert "only its owners" in gh.comments[2][0]


def test_a_commit_pushed_after_the_checks_is_not_merged_but_the_author_keeps_the_id():
    gh = FakeGitHub()
    gh.add_pr(5, ALICE, _files("my-tool"))
    gh.merge_status[5] = 409
    assert sweep(gh, log=lambda m: None) == 0
    assert gh.merged == [] and gh.dispatched == 0
    assert gh.owners == {"my-tool": [ALICE]}      # recorded before the merge, so a retry works


def test_the_owner_is_written_before_the_merge_and_retried_once_on_a_conflict():
    gh = FakeGitHub(owners={"other-tool": [BOB]})
    gh.add_pr(1, ALICE, _files("my-tool"))
    gh.owner_conflicts = 1                          # a maintainer changed owners.json meanwhile
    assert sweep(gh, log=lambda m: None) == 0
    assert gh.merged == [1]
    assert gh.owners == {"other-tool": [BOB], "my-tool": [ALICE]}


def test_no_merge_when_the_owner_cannot_be_recorded():
    gh = FakeGitHub()
    gh.add_pr(1, ALICE, _files("my-tool"))
    gh.owner_conflicts = 5
    assert sweep(gh, log=lambda m: None) == 1
    assert gh.merged == [] and gh.dispatched == 0


def test_a_tool_whose_manifest_names_another_is_not_merged():
    gh = FakeGitHub(tools=["banksy-niches"])
    sha = gh.add_pr(6, BOB, _files("evil-tool"),
                    head_files={"tools/evil-tool/plugin.json": json.dumps({"id": "banksy-niches"})})
    sweep(gh, trigger_sha=sha, log=lambda m: None)
    assert gh.merged == [] and "gives the id 'banksy-niches'" in gh.comments[6][0]


def test_an_owner_can_remove_their_tool():
    gh = FakeGitHub(tools=["my-tool"], owners={"my-tool": [ALICE]})
    gh.add_pr(8, ALICE, _files("my-tool"), head_files={})
    sweep(gh, log=lambda m: None)
    assert gh.merged == [8] and gh.owners == {"my-tool": [ALICE]}


def test_one_broken_pull_request_does_not_hold_up_the_others():
    gh = FakeGitHub()
    gh.add_pr(1, ALICE, _files("tool-a"))
    gh.add_pr(2, BOB, _files("tool-b"))
    gh.broken.add(1)
    assert sweep(gh, log=lambda m: None) == 1
    assert gh.merged == [2] and gh.owners == {"tool-b": [BOB]} and gh.dispatched == 1


def test_a_dry_run_changes_nothing():
    gh = FakeGitHub()
    sha = gh.add_pr(1, ALICE, _files("my-tool"))
    gh.add_pr(2, BOB, ["README.md"])
    lines = []
    assert sweep(gh, dry_run=True, trigger_sha=sha, log=lines.append) == 0
    assert gh.merged == [] and gh.owners is None and gh.dispatched == 0 and gh.comments == {}
    assert any("would merge my-tool" in line for line in lines)


def test_a_tool_may_change_as_many_files_as_a_bundle_holds():
    gh = FakeGitHub()
    gh.add_pr(1, ALICE, [f"tools/my-tool/f{i}.py" for i in range(automerge.MAX_FILES - 1)]
              + ["tools/my-tool/plugin.json"])
    sweep(gh, log=lambda m: None)
    assert gh.merged == [1]


@pytest.mark.parametrize("count", [automerge.MAX_FILES + 1, automerge.MAX_FILES + 5])
def test_too_many_files(count):
    gh = FakeGitHub()
    sha = gh.add_pr(1, ALICE, [f"tools/my-tool/f{i}.py" for i in range(count)])
    sweep(gh, trigger_sha=sha, log=lambda m: None)
    assert gh.merged == [] and "more than a tool may hold" in gh.comments[1][0]
