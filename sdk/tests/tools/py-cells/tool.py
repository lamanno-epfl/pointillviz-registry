"""A clean Python tool: every gate passes on it."""

import numpy as np


def run(ds, params):
    labels, codes = ds.cell_types()
    values = np.where(codes >= 0, codes.astype(np.float64), np.nan)
    ds.log(f"{len(labels)} cell types at k={params['k']}")
    return {"summary": f"{int((codes >= 0).sum())} labelled cells, k={params['k']}.",
            "cells": {"values": values, "label": "Cell-type code"}}
