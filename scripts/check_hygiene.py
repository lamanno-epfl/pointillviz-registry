#!/usr/bin/env python3
"""Keep private material out of this public repository.

    python3 scripts/check_hygiene.py

Reads every file git tracks or would add (`git ls-files --cached --others
--exclude-standard`), its path and its contents, and reports:

  * the names of private datasets;
  * private paths (/mnt/, home folders);
  * email addresses, in an author field or anywhere else;
  * links to the private app repository;
  * spatial-scale-spectrum anywhere but placeholders.json and index.json: it is
    listed as Coming soon, and has no code, README or artwork.

Prints one line per finding, as a GitHub annotation, and exits 1 if there is
any. This file holds the patterns, so it is the one file not checked; it keeps
the dataset names rot13-encoded, so that it does not spell them either.

Only the standard library and git are used, so CI runs this with the runner's
own python3.
"""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = "scripts/check_hygiene.py"
#: The private dataset names, rot13-encoded.
_WORDS = ("nyyra_", "vyrhz", "prphz", "pbyba ngynf")


def _names(encoded) -> re.Pattern:
    """Any of the decoded names, in any case, with any spacing between their words."""
    names = (codecs.decode(text, "rot13").split() for text in encoded)
    return re.compile("|".join(r"\s+".join(map(re.escape, words)) for words in names), re.I)


#: (what was found, pattern, the paths where it is allowed).
RULES = (
    ("the name of a private dataset", _names(_WORDS), ()),
    ("a private path",
     re.compile(r"/mnt/|/home/\w|/Users/\w|/scratch/|[A-Za-z]:\\Users\\"), ()),
    ("an email address",
     re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), ()),
    ("a link to the private app repository",
     re.compile(r"(?:github\.com|githubusercontent\.com)[/:]lamanno-epfl/pointillviz(?![\w-])",
                re.I), ()),
    ("spatial-scale-spectrum, which may appear only in placeholders.json and index.json",
     re.compile(r"spatial[-_ ]scale[-_ ]spectrum", re.I), ("placeholders.json", "index.json")),
)


def main() -> int:
    listed = subprocess.run(
        ["git", "-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=True).stdout
    paths = sorted({p for p in listed.split("\0") if p and p != SELF})
    found = 0
    for path in paths:
        for line, what, match in _findings(path):
            found += 1
            where = f"file={path},line={line}" if line else f"file={path}"
            print(f"::error {where}::{path}{f':{line}' if line else ''} holds {what}: {match[:60]}")
    if found:
        print(f"{found} finding{'s' if found > 1 else ''} in {len(paths)} files. This repository "
              "is public: remove them before committing.")
        return 1
    print(f"No private material in {len(paths)} files.")
    return 0


def _findings(path: str):
    """(line, what, match) for each rule a path or its contents break; line 0 is the path."""
    full = os.path.join(ROOT, path)
    if os.path.islink(full):
        text = os.readlink(full)             # the target's name, never what it points at
    elif os.path.isfile(full):
        with open(full, "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
    else:
        return                               # deleted, not yet committed
    for what, pattern, allowed in RULES:
        if path in allowed:
            continue
        hit = pattern.search(path)
        if hit:
            yield 0, what, hit.group(0)
        for match in pattern.finditer(text):
            yield text.count("\n", 0, match.start()) + 1, what, match.group(0)


if __name__ == "__main__":
    sys.exit(main())
