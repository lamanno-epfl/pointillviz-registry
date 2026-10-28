""".pvtool bundles: a list of paths chosen by a stranger.

None of the three classic ways unpacking goes wrong (zip-slip, symlinks, zip
bombs) may survive, and `pack` must ship the tool and nothing it accumulated
while being worked on — above all not its installed pixi environment.
"""

import io
import json
import os
import zipfile

import pytest

from pointillviz_tool.bundle import (
    ALLOWED_SUFFIXES, MAX_BUNDLE_BYTES, MAX_FILES, MAX_PREVIEW_BYTES, MAX_UNCOMPRESSED_BYTES,
    BundleError, extract, files, inspect, pack,
)

MANIFEST = {"schema": 3, "id": "demo-tool", "name": "Demo", "version": "1.2.3",
            "author": "a", "description": "d", "outputs": ["summary"]}
GOOD = {"plugin.json": json.dumps(MANIFEST),
        "tool.py": "def run(ds, params):\n    return {'summary': 'ok'}\n",
        "README.md": "# demo\n"}


def _zip_of(entries, symlink=None, compress=zipfile.ZIP_STORED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
        if symlink:
            info = zipfile.ZipInfo(symlink[0])
            info.external_attr = (0xA000 | 0o777) << 16
            zf.writestr(info, symlink[1])
    return buf.getvalue()


def _tool_dir(root, tree):
    for name, body in tree.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(body)
    return str(root)


def test_the_limits():
    assert (MAX_BUNDLE_BYTES, MAX_UNCOMPRESSED_BYTES, MAX_FILES, MAX_PREVIEW_BYTES) == \
        (25 * 1024 * 1024, 100 * 1024 * 1024, 300, 512 * 1024)
    assert {".r", ".toml", ".lock"} <= ALLOWED_SUFFIXES
    assert all(s == s.lower() for s in ALLOWED_SUFFIXES)    # names are compared lowered


def test_an_r_entry_in_either_case_is_allowed(tmp_path):
    entries = dict(GOOD, **{"tool.R": "run <- function(ds, params) 1\n",
                            "helper.r": "x <- 1\n"})
    assert set(inspect(_zip_of(entries))["names"]) == set(entries)


@pytest.mark.parametrize("name", [
    ".pixi/envs/default/lib/python3.11/site-packages/numpy/__init__.py",
    ".hidden.py", "sub/.git/config.txt", "__pycache__/tool.py", "a/../.pixi/x.py"])
def test_what_pack_leaves_out_is_refused_on_the_way_in(tmp_path, name):
    with pytest.raises(BundleError, match="hidden or cache"):
        extract(_zip_of(dict(GOOD, **{name: "x = 1\n"})), str(tmp_path / "out"))
    assert not os.path.exists(tmp_path / "out")


def test_a_good_bundle_round_trips(tmp_path):
    data = _zip_of(GOOD)
    assert set(inspect(data)["names"]) == set(GOOD)
    dest = extract(data, str(tmp_path / "unpacked"))
    assert open(os.path.join(dest, "tool.py")).read() == GOOD["tool.py"]


def test_an_r_tool_with_its_environment_is_a_valid_bundle():
    tree = {**GOOD, "tool.R": "run <- function(ds, params) list(summary = 'ok')\n",
            "helpers.r": "x <- 1\n", "pixi.toml": "[workspace]\n", "pixi.lock": "version: 6\n"}
    assert set(inspect(_zip_of(tree))["names"]) == set(tree)


def test_zip_slip_is_refused_after_resolution(tmp_path):
    for name in ("../evil.py", "a/../../evil.py", "/etc/evil.py", "C:/evil.py"):
        with pytest.raises(BundleError):
            inspect(_zip_of({**GOOD, name: "x"}))
    with pytest.raises(BundleError):
        extract(_zip_of({**GOOD, "../evil.py": "x"}), str(tmp_path / "slip"))
    assert not os.path.exists(tmp_path / "evil.py")


def test_a_symlink_entry_is_refused_outright():
    with pytest.raises(BundleError, match="symlink"):
        inspect(_zip_of(GOOD, symlink=("link.py", "/etc/passwd")))


def test_a_zip_bomb_fails_on_the_declared_size():
    # DEFLATE, so the archive itself is small and the packed-size limit is not
    # what catches it -- otherwise this would pass without the declared-size
    # check existing at all. Streamed, so the test never holds 100 MB.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in GOOD.items():
            zf.writestr(name, body)
        with zf.open("big.txt", "w") as fh:
            block = b"0" * (1 << 20)
            for _ in range(MAX_UNCOMPRESSED_BYTES // len(block) + 1):
                fh.write(block)
    data = buf.getvalue()
    assert len(data) < 1_000_000, f"the archive is {len(data)} bytes; not a bomb"
    with pytest.raises(BundleError, match="expands"):
        inspect(data)


def test_card_art_over_its_limit_is_refused():
    # The limit PointillViz and the store show a preview under, checked where
    # every bundle is opened, so CI refuses what the app would.
    svg = '<svg xmlns="http://www.w3.org/2000/svg">' + " " * MAX_PREVIEW_BYTES + "</svg>"
    with pytest.raises(BundleError, match="preview.svg"):
        inspect(_zip_of({**GOOD, "preview.svg": svg}, compress=zipfile.ZIP_DEFLATED))
    inspect(_zip_of({**GOOD, "preview.svg": "<svg/>", "docs/big.svg": svg},
                    compress=zipfile.ZIP_DEFLATED))


def test_an_oversized_archive_is_refused_before_it_is_read():
    with pytest.raises(BundleError, match="limit"):
        inspect(b"\0" * (MAX_BUNDLE_BYTES + 1))


def test_a_nested_manifest_names_the_mistake():
    with pytest.raises(BundleError) as exc:
        inspect(_zip_of({f"my-tool/{k}": v for k, v in GOOD.items()}))
    assert "nested" in str(exc.value) and "my-tool" in str(exc.value), \
        "zipping the directory instead of its contents is THE common mistake"


def test_binaries_file_count_and_a_missing_manifest_are_refused():
    for bad in ({**GOOD, "lib.so": "x"}, {**GOOD, "env.tar.gz": "x"},
                {**GOOD, **{f"f{i}.txt": "x" for i in range(MAX_FILES)}},
                {"tool.py": "x"}):
        with pytest.raises(BundleError):
            inspect(_zip_of(bad))
    inspect(_zip_of({**GOOD, **{f"f{i}.txt": "x" for i in range(MAX_FILES - len(GOOD))}}))
    with pytest.raises(BundleError, match="zip"):
        inspect(b"not a zip at all")


def test_pack_ships_the_tool_and_none_of_its_residue(tmp_path):
    src = _tool_dir(tmp_path / "demo-tool", {
        **GOOD, "pixi.toml": "[workspace]\n", "pixi.lock": "version: 6\n",
        "lib/helper.py": "X = 1\n",
        "__pycache__/tool.cpython-311.pyc": "x", "lib/helper.pyc": "x",
        ".hidden": "x", ".git/config": "x", ".pixi/envs/default/bin/python": "x",
        ".pixi/envs/default/lib/libz.so": "x"})
    out = pack(src)
    assert out == str(tmp_path / "demo-tool-1.2.3.pvtool"), \
        "a bundle is named <id>-<version>.pvtool and lands next to the tool"
    names = inspect(open(out, "rb").read())["names"]
    assert sorted(names) == sorted([*GOOD, "pixi.toml", "pixi.lock", "lib/helper.py"])
    assert sorted(n for n, _ in files(src)) == sorted(names), \
        "files() is the one definition of what pack writes"
    assert pack(src, str(tmp_path / "elsewhere.pvtool")).endswith("elsewhere.pvtool")
