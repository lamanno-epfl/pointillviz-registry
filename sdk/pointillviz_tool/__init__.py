"""The PointillViz tool SDK.

A tool is a folder with a `plugin.json` and an entry file defining
`run(ds, params)`. A Python tool needs only what this module exports:

    from pointillviz_tool import PluginError, html_view, download_file, image

    def run(ds, params):
        labels, codes = ds.cell_types()
        ...
        return {"summary": "...", "cells": {"values": per_cell_values}}

`ds` is a `Dataset`, `PluginError` is how a tool refuses an input in a
sentence, and `check_result` is the validator every return value goes
through. Nothing here imports scipy until a spatial query needs it.
"""

from .dataset import Dataset, PluginError
from .results import check_result, download_file, html_view, image

__version__ = "1.0.0"

__all__ = ["Dataset", "PluginError", "check_result", "download_file",
           "html_view", "image", "__version__"]
