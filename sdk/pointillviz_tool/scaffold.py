"""Write a new tool that already works.

    python -m pointillviz_tool new my-tool [--r] [-o DIR]

The starter it writes is not a stub with `TODO` in `run()`: it computes
something real, paints it, and says one sentence about it, so `python -m
pointillviz_tool validate` passes on the output of this command with nothing
edited once `pixi lock` has written the lock file. That is the point: an
author's first run of the validator has to be a PASS, or they cannot tell
their own first mistake from a broken scaffold.

Two templates, one per language, each a complete schema-3 tool: plugin.json,
the entry file, pixi.toml and the README that becomes its store page.

  python  `tool.py`: per-cell transcript totals, painted as `cells`, with a
          minimum count below which a cell is left unpainted.
  r       `tool.R`: the most common cell type among each cell's nearest
          neighbours, painted as `labels`.

Each manifest has one numeric parameter, so the generated form is not empty
and the `edge` gate has corners to test, and declares exactly the outputs the
tool returns.
"""

from __future__ import annotations

import json
import os

from .bundle import files
from .manifest import validate

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
LANGUAGES = ("python", "r")


def scaffold(tool_id: str, parent_dir: str = ".", *, language: str = "python") -> str:
    """Write `<parent_dir>/<tool_id>/` from the template for `language`. Returns its path.

    Raises ManifestError for an id the manifest would refuse, and
    FileExistsError when the folder is already there.
    """
    if language not in LANGUAGES:
        raise ValueError(f"language must be 'python' or 'r', not {language!r}.")
    name = tool_id.replace("-", " ").replace("_", " ").title()
    source = os.path.join(TEMPLATES, language)
    texts = {}
    for rel, full in files(source):
        with open(full, "r", encoding="utf-8") as fh:
            texts[rel] = fh.read().replace("{{ID}}", tool_id).replace("{{NAME}}", name)
    # Validated before anything is written: a scaffold that emits a manifest
    # the host would refuse is worse than no scaffold at all, because the
    # author would go looking for the mistake in their own code.
    validate(json.loads(texts["plugin.json"]))

    out_dir = os.path.join(parent_dir, tool_id)
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists; pick another id, or remove it first.")
    os.makedirs(out_dir)
    for rel, text in texts.items():
        with open(os.path.join(out_dir, rel), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return out_dir
