"""`.pvtool` bundles: pack a tool directory, and unpack one you did not write.

A bundle is an ordinary zip with `plugin.json` at its root. The interesting
half of this module is `extract`, because a zip is a list of paths chosen by
whoever built it, and every one of them is about to be joined onto a directory
on this machine. The three classic ways that goes wrong are all handled here
and none of them are hypothetical:

  * **Zip-slip.** An entry named `../../backend/serve.py` escapes the
    destination and overwrites the server. Every name is resolved and checked
    to still be inside the destination after resolution, which is the only
    check that holds for `a/../../b` as well as for a leading `..`.
  * **Symlinks.** A zip can carry a symlink; extract one pointing at `/etc` and
    the next entry writes through it. Python's own `ZipFile.extract` follows
    them. Symlink entries are refused outright.
  * **Zip bombs.** A 40 kB archive can declare 4 GB of contents. The DECLARED
    uncompressed size is summed before anything is written, and the write is
    counted as it goes, so a lying header fails at the same limit.

Absolute paths, Windows drive letters and device names are refused for the same
family of reasons.
"""

from __future__ import annotations

import io
import os
import posixpath
import zipfile

#: A tool is source files, a manifest, its pixi.toml and pixi.lock, a README
#: and a preview, plus whatever small data it ships. These leave room for
#: that and stop well short of "ship an environment": the environment is
#: built from pixi.lock at install time, never carried in the bundle.
MAX_BUNDLE_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_FILES = 300
#: preview.svg is card art, shown a few hundred pixels wide by PointillViz and
#: the store, so anything near this is a mistake rather than a picture.
MAX_PREVIEW_BYTES = 512 * 1024
#: What a bundle is allowed to contain. Not a security boundary — the entry
#: is arbitrary code — but it keeps a bundle to the shape the store page can
#: render and stops binaries riding along unexamined. Compared in lower case,
#: so `.r` also admits `tool.R`.
ALLOWED_SUFFIXES = {".py", ".r", ".json", ".md", ".txt", ".csv", ".tsv",
                    ".svg", ".png", ".yaml", ".yml", ".toml", ".lock", ".cfg",
                    ".html", ".js", ".css"}


class BundleError(ValueError):
    """A bundle the host will not open, and why in one sentence."""


def files(plugin_dir: str) -> list:
    """Every file `pack` puts in a bundle: [(name in the archive, path on disk)].

    The one definition of what is part of a tool, so the residue of working on
    a tool where it sits cannot ship: `__pycache__` (a .pyc compiled for this
    interpreter), and every dot-file and dot-directory — `.git`, and above all
    `.pixi`, the tool's installed environment.
    """
    out = []
    for root, dirs, names in os.walk(plugin_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for fn in sorted(names):
            if fn.endswith((".pyc", ".pyo")) or fn.startswith("."):
                continue
            full = os.path.join(root, fn)
            if os.path.islink(full):
                continue
            out.append((os.path.relpath(full, plugin_dir).replace(os.sep, "/"), full))
    return out


def pack(plugin_dir: str, out_path: str = None) -> str:
    """Zip a tool directory into `<id>-<version>.pvtool`. Returns the path.

    By default the bundle is written next to the tool directory, not inside
    it, so packing twice never packs the previous bundle.
    """
    from .manifest import load
    manifest = load(plugin_dir)
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(plugin_dir)),
            f"{manifest['id']}-{manifest['version']}.pvtool")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, full in files(plugin_dir):
            zf.write(full, rel)
    size = os.path.getsize(out_path)
    if size > MAX_BUNDLE_BYTES:
        os.remove(out_path)
        raise BundleError(f"The bundle is {size / 1e6:.1f} MB; the limit is "
                          f"{MAX_BUNDLE_BYTES / 1e6:.0f} MB.")
    return out_path


def inspect(data: bytes) -> dict:
    """Check a bundle's shape WITHOUT writing anything. Raises BundleError.

    Returns {names, total_uncompressed, has_manifest}. Split out from
    `extract` so the `bundle` gate and an install can both reject a bad
    archive before it touches a filesystem.
    """
    if len(data) > MAX_BUNDLE_BYTES:
        raise BundleError(f"The bundle is {len(data) / 1e6:.1f} MB; the limit is "
                          f"{MAX_BUNDLE_BYTES / 1e6:.0f} MB.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise BundleError("Not a readable zip archive.")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_FILES:
            raise BundleError(f"The bundle holds {len(infos)} files; the limit "
                              f"is {MAX_FILES}.")
        total = 0
        names = []
        for info in infos:
            _check_name(info.filename)
            # The high 16 bits of external_attr are the Unix mode; 0xA000 is
            # S_IFLNK. ZipFile.extract would happily create the symlink as a
            # regular file OR follow an existing one, so this is refused here.
            if (info.external_attr >> 16) & 0xF000 == 0xA000:
                raise BundleError(f"{info.filename!r} is a symlink; bundles may "
                                  "not contain symlinks.")
            if info.filename == "preview.svg" and info.file_size > MAX_PREVIEW_BYTES:
                raise BundleError(f"preview.svg is {info.file_size // 1024} kB; the limit "
                                  f"is {MAX_PREVIEW_BYTES // 1024} kB.")
            total += info.file_size
            names.append(info.filename)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise BundleError(f"The bundle expands to {total / 1e6:.1f} MB; the "
                              f"limit is {MAX_UNCOMPRESSED_BYTES / 1e6:.0f} MB.")
        if "plugin.json" not in names:
            nested = [n for n in names if n.endswith("/plugin.json")]
            hint = (f" It looks like the tool is nested inside "
                    f"{nested[0].rsplit('/', 1)[0]!r} — zip the CONTENTS of the "
                    "directory, not the directory.") if nested else ""
            raise BundleError("plugin.json must be at the root of the bundle." + hint)
    return {"names": names, "total_uncompressed": total, "has_manifest": True}


def extract(data: bytes, dest_dir: str) -> str:
    """Unpack a bundle into `dest_dir`, safely. Returns `dest_dir`."""
    info = inspect(data)
    os.makedirs(dest_dir, exist_ok=True)
    dest_real = os.path.realpath(dest_dir)
    written = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in info["names"]:
            target = os.path.realpath(os.path.join(dest_real, name))
            # After resolution, not before: `a/../../b` has no leading ".." and
            # would pass a naive prefix test on the raw name.
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise BundleError(f"{name!r} would be written outside the "
                                  "plugin directory.")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 16)
                    if not chunk:
                        break
                    written += len(chunk)
                    # The declared size was checked in inspect(); this catches a
                    # header that lied about it.
                    if written > MAX_UNCOMPRESSED_BYTES:
                        raise BundleError("The bundle expands past the "
                                          f"{MAX_UNCOMPRESSED_BYTES / 1e6:.0f} MB limit.")
                    out.write(chunk)
    return dest_dir


def _check_name(name: str) -> None:
    if not name or name.startswith("/") or name.startswith("\\"):
        raise BundleError(f"{name!r} is an absolute path.")
    if ":" in name.split("/")[0]:
        raise BundleError(f"{name!r} names a drive letter.")
    parts = posixpath.normpath(name).split("/")
    if ".." in parts:
        raise BundleError(f"{name!r} escapes the plugin directory.")
    # What `files` leaves out of a bundle is refused in one: a `.pixi/` here
    # would be unpacked where the host builds the tool's environment.
    if any(part.startswith(".") or part == "__pycache__" for part in parts):
        raise BundleError(f"{name!r} is a hidden or cache file; a bundle holds the tool's "
                          "own files only, as `pack` writes it.")
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise BundleError(
            f"{name!r} has an unsupported file type. Bundles may contain: "
            + ", ".join(sorted(ALLOWED_SUFFIXES)))
