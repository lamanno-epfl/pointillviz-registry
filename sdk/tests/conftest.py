"""Shared test data: the conformance fixture, written by the SDK's own writer.

The session copy is for tests that only read. A test that writes into a
dataset (the k-NN cache) or edits one of its files takes `fresh_fixture`, a
private copy, so no test can change what another one reads.
"""

import os

import pytest

from pointillviz_tool.fixture import write_fixture


@pytest.fixture(scope="session")
def fixture_info(tmp_path_factory):
    return write_fixture(str(tmp_path_factory.mktemp("fixture") / "fx"))


@pytest.fixture(scope="session")
def fixture_dir(fixture_info):
    return fixture_info["path"]


@pytest.fixture
def fresh_fixture(tmp_path):
    return write_fixture(str(tmp_path / "fx"))["path"]


def tree_state(root):
    """{relative path: (size, mtime_ns)} for every file under `root`."""
    out = {}
    for base, _dirs, names in os.walk(root):
        for name in names:
            full = os.path.join(base, name)
            st = os.stat(full)
            out[os.path.relpath(full, root)] = (st.st_size, st.st_mtime_ns)
    return out


@pytest.fixture
def snapshot():
    """`snapshot(root)` returns a callable that lists what changed under root since."""
    def take(root):
        before = tree_state(root)

        def changed():
            after = tree_state(root)
            return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        return changed
    return take
