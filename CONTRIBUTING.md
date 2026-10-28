# Writing a PointillViz tool

This guide covers writing a community tool in Python or in R, checking it, and publishing it in this registry.

A tool is a folder. Its manifest, `plugin.json`, says what the tool needs from a dataset, which parameters the user chooses and what the tool returns; PointillViz builds the tool's form from it. Its entry file, `tool.py` or `tool.R`, defines one function, `run(ds, params)`. Its `pixi.toml` and `pixi.lock` describe the environment it runs in. PointillViz always runs a tool in a separate process, in that environment.

1. [Start](#1-start)
2. [A tool's folder](#2-a-tools-folder)
3. [plugin.json](#3-pluginjson)
4. [Python tools](#4-python-tools)
5. [R tools](#5-r-tools)
6. [Outputs](#6-outputs)
7. [The environment, and GPUs](#7-the-environment-and-gpus)
8. [The gates](#8-the-gates)
9. [The commands](#9-the-commands)
10. [Publishing](#10-publishing)
11. [Versions](#11-versions)

## 1. Start

You need:

- [pixi](https://pixi.sh), which builds every tool's environment and writes its lock file. CI uses pixi 0.83.0, so write the lock with that version or a later one that keeps its lock format;
- Python 3.10 or later with numpy and scipy, to run the SDK's commands;
- a clone of this repository. The SDK is `sdk/pointillviz_tool`: put `sdk` on `PYTHONPATH`, or install it with pip.

```bash
git clone https://github.com/lamanno-epfl/pointillviz-registry
cd pointillviz-registry
export PYTHONPATH="$PWD/sdk"        # or: pip install ./sdk
```

Write a starter tool, lock its environment and run the gates:

```bash
python -m pointillviz_tool new my-tool -o tools          # a Python tool
python -m pointillviz_tool new my-tool -o tools --r      # or a native R tool
cd tools/my-tool
pixi lock                                                # writes pixi.lock
python -m pointillviz_tool validate .                    # builds the environment, runs the gates
```

The starter already works: it passes every gate once `pixi lock` has written its lock file. The Python starter paints each cell by its total transcript count (`cells`); the R starter labels each cell with the most common cell type among its nearest neighbours (`labels`). Change `run()` and the manifest from there, and run `validate` after each change: its report says what to fix.

To try the tool on data, give `run` a dataset folder that PointillViz has converted (the folder that holds `meta.json`), or write the small synthetic dataset the gates use, somewhere outside the repository:

```bash
python -c "from pointillviz_tool.fixture import write_fixture; write_fixture('/tmp/pv-fixture')"
python -m pointillviz_tool run . --dataset /tmp/pv-fixture --params '{"min_counts": 50}'
```

To install it in PointillViz before it is published, pack it and use **Tools → Install from file**:

```bash
python -m pointillviz_tool pack .                        # writes ../my-tool-0.1.0.pvtool
```

## 2. A tool's folder

```
tools/<id>/
  plugin.json    the manifest
  tool.py        or tool.R: defines run(ds, params)
  pixi.toml      the tool's environment
  pixi.lock      written by `pixi lock`; commit it with pixi.toml
  README.md      the tool's store page
  preview.svg    optional card art for the store, at most 512 kB
  ...            optional helper modules (.py or .R) and small data files
```

- The folder is named after the tool's `id`.
- A tool travels as a bundle, a zip of its folder that `pack` writes (`<id>-<version>.pvtool`). A bundle holds only these file types: `.py .R .json .md .txt .csv .tsv .svg .png .yaml .yml .toml .lock .cfg .html .js .css`. It has at most 300 files, 25 MB packed and 100 MB unpacked, a `preview.svg` of at most 512 kB, and no symbolic links. Hidden files and folders, the `.pixi` environment among them, and `__pycache__` are left out.
- The environment never travels in the bundle: it is built from `pixi.lock` wherever the tool is installed.
- A tool runs with its own folder as the working directory, and Python finds the helper modules shipped beside `tool.py`. Read a data file relative to the working directory, or to `__file__` in Python. Give helpers names of their own: one named like a standard module (`select.py`, `json.py`) cannot be imported.
- `README.md` is the tool's page in the store, both in PointillViz and on store.pointillviz.com. Say what the tool computes, how to read each output, what each parameter does, and what its limits are; the starter's README shows the shape.

## 3. plugin.json

The Python starter's manifest:

```json
{
  "schema": 3,
  "id": "my-tool",
  "name": "My Tool",
  "version": "0.1.0",
  "author": "Your Name",
  "description": "Per-cell transcript totals: the starting point for a new tool.",
  "long_description": "Sums each cell's transcripts and paints them, leaving cells below a threshold unpainted. Replace the body of run() with the analysis you want; the manifest around it is already correct.",
  "icon": "functions",
  "tags": ["starter"],
  "entry": "tool.py",
  "outputs": ["summary", "cells"],
  "requires": {"genes": true, "spatial": "any", "min_cells": 1},
  "params": [
    {
      "name": "min_counts",
      "type": "int",
      "label": "Minimum counts",
      "default": 0,
      "min": 0,
      "max": 5000,
      "step": 1,
      "help": "Cells with fewer transcripts than this are left unpainted."
    }
  ]
}
```

| Field | Rule |
|---|---|
| `schema` | `3` |
| `id` | 3 to 50 characters: lowercase letters, digits and hyphens, starting with a letter and ending with a letter or a digit. It names the folder, the bundle and the release |
| `name` | the tool's name as people read it |
| `version` | `MAJOR.MINOR.PATCH` (see [Versions](#11-versions)) |
| `author` | a person's or a group's name, with no email address: this repository is public |
| `description` | one short sentence: the store card and the Tools tab show it |
| `long_description` | optional, longer Markdown |
| `entry` | `tool.py` or `tool.R` (default `tool.py`): its extension decides the language |
| `gpu` | `"optional"` or `"required"`, or left out for a tool that never uses a GPU (see [GPUs](#gpus)) |
| `outputs` | the output kinds every successful run returns (see [Outputs](#6-outputs)) |
| `optional_outputs` | kinds a run may return or leave out |
| `requires` | what a dataset needs for the tool to be offered on it |
| `params` | the parameters the user chooses |
| `icon` | a Material Symbols icon name, used when there is no `preview.svg` (default `extension`) |
| `tags` | up to 8 short words |
| `license`, `citation`, `homepage` | optional; cite the method the tool implements |

Schema 3 refuses `timeout_s`, `memory_mb` and `requires.scipy`: how long a tool takes and how much memory it uses are its author's to manage, and its dependencies belong in `pixi.toml`. Keys the schema does not know are dropped.

### requires

| Key | Meaning |
|---|---|
| `genes` | needs the expression matrix |
| `cell_types` | needs the cell-type annotation itself |
| `categoricals` | needs at least one categorical annotation: the right choice for a tool with a `categorical_field` parameter |
| `spatial` | `"any"`, `"2d"` or `"3d"` |
| `min_cells` | the number of cells below which the tool is not offered |

Declaring too little is a bug: the tool is offered, then fails. Declaring too much hides the tool from datasets it would work on.

### params

PointillViz draws one control per parameter, and checks and converts each value before `run()` sees it.

| `type` | Control | Value `run()` gets | Also |
|---|---|---|---|
| `int` | slider and number box | an integer, clamped to `[min, max]` | `min` and `max` required, `step` optional |
| `float` | slider and number box | a number, clamped to `[min, max]` | `min` and `max` required, `step` optional |
| `bool` | switch | true or false | |
| `text` | text box | a string | |
| `select` | menu | one of the option values | `options`: `[{"value": ..., "label": ...}]` |
| `gene` | gene search | a gene name | |
| `genes` | gene picker | a list of gene names | `max_items` (default 20); `preset` |
| `categorical_field` | the dataset's categorical columns | a column name | |
| `category_value` | the levels of a column | a level | `field_param`: the `categorical_field` parameter it follows |
| `continuous_field` | the dataset's continuous columns | a column name | |
| `json` | JSON editor | an object or a list, at most 64 KiB | |

Each parameter has a `name` (letters, digits and `_`, not starting with a digit, so that it works in Python and in R), a `label` and a one-line `help`.

- **Defaults.** `int`, `float`, `bool`, `text`, `select` and `json` parameters need a `default`, or `"optional": true`. The form opens on the defaults, and the gates run the tool on them. A `select` default must be one of its options, and a numeric default must lie in its own range.
- **Dataset parameters** (`gene`, `genes`, `categorical_field`, `category_value`, `continuous_field`) take no default: PointillViz fills them from the open dataset, and the gates from theirs. With `"optional": true`, `run()` gets `None` (`NULL` in R) when nothing is chosen.
- **Gene presets.** A `genes` parameter with `"preset": "top_markers"` opens on the `preset_n` genes (default `max_items`) detected in the most cells: the Marker Dot Plot's "Top markers" ranking, which `ds.top_detected_genes(n)` returns. The gates run on the same list, and the user edits it from there.
- **Sweeps.** One `int` or `float` parameter may set `"sweep": true`, with `sweep_steps` from 2 to 12 (default 8). PointillViz can then run the tool over a range of that parameter at once, and the user moves between the steps. A tool that returns per-cell outputs (`cells`, `vectors`, `labels`) cannot declare a sweep.

## 4. Python tools

```python
from pointillviz_tool import PluginError, html_view, download_file, image

def run(ds, params):
    ...
    return {"summary": "...", "cells": {"values": values, "label": "Score"}}
```

`run(ds, params)` is the one name PointillViz looks for. `params` is a dict of the declared parameters, already converted. The return value is a dict of [outputs](#6-outputs).

**Refusing.** Raise `PluginError(code, message)` to refuse an input in words the user reads: name the number that failed and the change that fixes it. Anything else that is raised is reported as a crash with its traceback, which is right for a bug and wrong for "no cell qualified".

### The ds handle

Every per-cell array has `ds.n_cells` entries, in dataset order, and NaN means "no value here". Carry it through (`np.nanmean`, `np.isfinite` masks) rather than filling it in.

```
ds.name, ds.title, ds.n_cells
ds.meta                 the dataset's meta.json (spatial.mode, spatial.units, ...)
ds.device               "cpu" or "cuda": where this run may compute
ds.subset               the indices of the cells the user selected, or None
ds.section_filter       the section the user isolated, or None

ds.xy                   (n, 2) float32 coordinates; NaN rows have no position
ds.coords               (n, 2) or (n, 3) float32: the coordinates to do geometry in
ds.z                    (n,) float32 for a 3D dataset, else None
ds.is_3d
ds.section              (n,) int32 section index
ds.sections             the dataset's sections

ds.gene_names           list of gene names
ds.has_gene(name)
ds.gene(name, normalize="cp10k")      (n,) float32; normalize is "raw", "cp10k" or "log1p"
ds.genes(names, normalize="cp10k")    (n, k) float32; column j is names[j]
ds.total_counts()       (n,) float64 transcripts per cell

ds.cell_types()         (levels, codes); codes is (n,) int32, -1 where unlabelled
ds.cell_type_colors     {level: "#rrggbb"}
ds.categorical_fields() [{name, label, levels, colors}]
ds.categorical(name=None)             (levels, codes) for one column; None is the cell-type column
ds.continuous_fields()  [{name, label, min, max}]
ds.continuous(name)     (n,) float32, NaN where absent

ds.knn(k=15)            (idx, dist), both (n, k); idx is -1 and dist inf where a slot cannot be filled
ds.kdtrees()            [{section, tree, rows}]: one scipy cKDTree per section

ds.log(message)         one line in the run's log
ds.progress(fraction, message=None)
```

- `ds.knn` and `ds.kdtrees` work per section, because sections sit side by side on one canvas and a single tree would give a cell neighbours from another specimen. Neighbour graphs are cached in the dataset's `plugin_cache/` folder, the one place a tool may write in a dataset.
- **Mask `-1`.** `codes[idx]` with `idx == -1` silently reads the last cell. Clamp with `np.where(idx >= 0, idx, 0)`, then mask.
- Honouring `ds.subset` is optional. A tool that does still returns full-length per-cell outputs, with NaN outside the selection.
- When `ds` refuses (an unknown gene, a dataset without cell types), it raises `PluginError`, and the user reads its sentence.

## 5. R tools

```r
run <- function(ds, params) {
  types <- ds$cell_types()          # list(levels = chr, codes = int), NA where unlabelled
  nn    <- ds$knn(15L)               # list(idx = int matrix, dist = dbl matrix)
  x     <- ds$genes(c("Cd3e", "Epcam"))
  ds$log("hello"); ds$progress(0.5, "half way")
  if (bad) pv_error("bad_input", "A sentence the user reads.")
  list(summary = "...", labels = list(values = niche, label = "Niche"))
}
```

`tool.R` defines `run <- function(ds, params)` and returns a named list of outputs, of the same kinds as a Python tool. `params` is a named list. PointillViz runs the tool with `Rscript` in its environment, which holds `r-base` and `r-jsonlite`. Add each R package the tool uses to `pixi.toml` as `r-<name>` from conda-forge (for example `r-mclust`), and load it with `library()`.

**Refusing.** `pv_error(code, message)` refuses an input in words the user reads, as `PluginError` does in Python. Any other error is reported as a crash, with R's call stack.

### The R ds

```
ds$n_cells, ds$name, ds$title, ds$meta, ds$device, ds$subset, ds$section_filter
ds$xy(), ds$coords(), ds$section()
ds$gene_names(), ds$has_gene(name)
ds$gene(name, normalize = "cp10k"), ds$genes(names, normalize = "cp10k")
ds$total_counts()
ds$categorical_fields(), ds$categorical(name = NULL), ds$cell_types()
ds$continuous_fields(), ds$continuous(name)
ds$knn(k = 15L)
ds$log(message), ds$progress(fraction, message = NULL)
```

They read the same data as the Python methods of the same names, in R's conventions: **everything that indexes something is 1-based, and NA means "none"**. That covers the neighbours of `ds$knn()` (`idx`, with NA, and an `Inf` distance, where a slot cannot be filled), the `codes` of `ds$categorical()` and `ds$cell_types()` (so `levels[codes]` works), `ds$section()`, `ds$section_filter` and `ds$subset`. Missing numbers stay `NaN`, which `is.na()` also catches. `ds$genes()` returns a matrix whose column names are the genes.

### What an R tool returns

- **`labels$values` is a factor or a character vector**, with one entry per cell and `NA` where a cell has no label. Category codes are never written by hand in R: the categories are the factor's levels, or the sorted distinct values of a character vector. `labels$categories` sets their order, and `labels$colors` follows the categories, by position or by name. Numeric `values` are refused.
- A `table` may be a data.frame, or have one as its `rows`.
- A vector of length one stays an array where an output needs one.
- `NA` is "no value"; an infinity is refused.

## 6. Outputs

`run()` returns a dict (a named list in R) whose keys are output kinds. Declare in `outputs` the kinds every successful run returns, and in `optional_outputs` those a run may leave out. A run returns at least one kind, and nothing undeclared.

| Kind | What PointillViz shows |
|---|---|
| `summary` | Markdown text: the findings of this run |
| `cells` | one number per cell, painted on the map |
| `labels` | one category per cell, painted on the map with a legend |
| `vectors` | arrows over the tissue |
| `table` | a table |
| `series` | lines against an x axis |
| `matrix` | a heatmap |
| `radar` | radial profiles |
| `chord` | a chord diagram |
| `images` | static figures, PNG or SVG |
| `files` | downloads |
| `views` | self-contained HTML, shown in an isolated frame |

```python
{
  "summary": "**Markdown.** What this run found, and the caveats on it.",

  "cells": {                         # exactly n_cells numbers; NaN is "no value"
    "values": values,
    "label": "Mixing entropy",
    "colormap": "viridis",           # viridis, plasma, inferno, magma, cividis or icefire
    "diverging": False,              # True for a signed statistic
    "domain": [0.0, 1.0],            # optional; otherwise the 1st to 99th percentile
    "units": "bits",
  },

  "labels": {                        # exactly n_cells codes, 0 to len(categories) - 1, or None
    "values": codes,
    "categories": ["N1", "N2", "N3"],                # 1 to 256 names of at most 80 characters
    "colors": ["#e6194b", "#3cb44b", "#ffe119"],     # optional, one per category
    "label": "Niche",
  },

  "vectors": {                       # anchors and components, in the dataset's coordinates
    "x": [...], "y": [...], "u": [...], "v": [...],
    "value": [...],                  # optional colour scalar; otherwise the magnitude
    "label": "Gradient", "units": "per 100 µm",
    "pitch": 55.4,                   # optional spacing between neighbouring anchors
  },

  "table": {"title": "...", "columns": ["Group", "Cells"], "rows": [["A", 10]],
            "sort_by": "Cells"},

  "series": {"title": "...", "x_label": "...", "y_label": "...",
             "lines": [{"label": "...", "x": [...], "y": [...],
                        "color": "#4477aa", "dashed": False, "band": False}],
             "rules": [0.0], "log_x": False},

  "matrix": {"title": "...", "rows": [...], "cols": [...], "values": [[...]],
             "diverging": False, "colormap": "viridis"},

  "radar": {"title": "...", "axes": ["A", "B", "C"], "domain": [0, 1],
            "layout": "multiples",   # or "overlay"
            "profiles": [{"label": "...", "values": [0.2, 0.8, 0.5]}], "note": "..."},

  "chord": {"title": "...", "directed": False,
            "nodes": [{"label": "A"}, {"label": "B"}],
            "links": [{"source": 0, "target": 1, "value": 120, "score": 1.2}], "note": "..."},

  "images": [image(figure, "UMAP")],
  "files": [download_file("niches.csv", csv_text, media_type="text/csv")],
  "views": [html_view(html, id="plot", title="Custom plot", height=480)],
}
```

- **One value per cell, in dataset order.** `cells.values` and `labels.values` have exactly `ds.n_cells` entries, not one per cell the tool scored.
- **NaN (None) is "no value".** The map dims those cells rather than painting them at the bottom of the colour ramp: "not measured" and "measured zero" are different statements. An infinity anywhere is an error.
- **`diverging: True` for a signed statistic.** It centres the colour range on zero and uses `icefire`, the one two-sided ramp.
- **`labels` in Python** holds integer codes into `categories`, counted from 0, or None for a cell with no label; a negative code is refused. Without `colors`, PointillViz picks a palette. An R tool passes a factor instead (see [R tools](#5-r-tools)).
- **`vectors`** are arrows at anchors that are not cells: `(u, v)` is the quantity at `(x, y)`, and PointillViz chooses the drawn length. A `band: True` series line fills the area to the line before it, which draws an envelope.
- **`radar`** has 3 to 100 axes and 1 to 100 profiles, every value inside `domain`. **`chord`** has 1 to 100 nodes and at most 2,000 links between node indices (from 0), with non-negative values; `score` is an optional signed number.
- **`images`**: 1 to 8, each with a title of at most 200 characters and exactly one of `png` (base64) or `svg` (text). `image(data, title)` builds one from PNG bytes, SVG text (a `str`) or a matplotlib Figure; matplotlib then belongs in the tool's `pixi.toml`. An SVG is shown as an image, never put into the page itself.
- **`files`**: 1 to 16 downloads, each `{name, media_type, base64}` with a plain file name. `download_file(name, data, media_type=...)` builds one from bytes or text.
- **`views`**: 1 to 8 self-contained HTML documents, each `{id, title, html, height}`, with a height from 160 to 1,600 pixels. A view runs in an isolated frame, with scripts but without the network: inline every asset. Use the native kinds for anything painted on the tissue.
- **Limits:** 5,000 table rows, 5,000 points per series line, 10,000 matrix cells, 20,000 arrows, 20,000 characters of summary and 200 log lines. Views, files and images share 8 MiB.
- **Refuse rather than paint nothing.** A map of NaN is a legal result, so nothing downstream can tell "the threshold excluded every cell" from "the tool is broken". When there is no answer, refuse with `PluginError` (`pv_error` in R), naming the number that failed and the change that fixes it. The gates fail a tool whose painted output is blank at its own defaults.
- **Keep the summary short:** the numbers of this run and the caveats on them, in a few lines. What the statistic is and how to read it belong in `README.md`. The summary's Markdown is a small subset: `**bold**`, `*italic*`, `` `code` ``, paragraphs separated by a blank line, and `-` lists.

## 7. The environment, and GPUs

`pixi.toml` describes the environment the tool runs in, and `pixi.lock`, which `pixi lock` writes, pins it exactly. The Python starter's:

```toml
[workspace]
name = "my-tool"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[dependencies]
python = ">=3.10"
numpy = "*"
scipy = "*"
```

- **Always declare** `python`, `numpy` and `scipy`, because the SDK runs inside the tool's environment. An R tool adds `r-base` and `r-jsonlite`.
- **Do not declare** the SDK: PointillViz provides it.
- **Channels:** `conda-forge`, and `bioconda` when needed. `[pypi-dependencies]` is allowed.
- **Platforms:** at least `linux-64`, where CI runs the gates. All four above are recommended, so that the tool installs on Linux, macOS and Windows.
- Run `pixi lock` after every change to `pixi.toml`, and commit both files. The gates check that the lock still matches.
- When someone installs the tool into PointillViz's own environment rather than its own, the tool's `[dependencies]`, `[pypi-dependencies]` and channels are added to PointillViz's, and PointillViz warns if they cannot be solved together, or if that would change or remove a package it already has. Loose requirements (`"*"`, `">=1.2"`) fit in more easily than exact pins. A requirement given as a path, a git repository or a URL can only be installed in the tool's own environment.

### GPUs

A tool that can use a GPU says so with one key in `plugin.json`:

- `"gpu": "optional"`: the tool runs on either device, and the user chooses CPU or GPU;
- `"gpu": "required"`: the tool is meant for a GPU. PointillViz selects the GPU by default, and says so when the computer has none.

Either way, the tool must also run on the CPU. `ds.device` is `"cpu"` or `"cuda"`: do the work there (`torch.device(ds.device)`, for example), and fall back to the CPU when CUDA is not available. On a CPU run, PointillViz also hides every GPU from the process. The gates run every tool on the CPU, `"required"` ones included, and CI has no GPU: that is how the CPU fallback is tested. Check the GPU path on your own computer with `validate --device cuda`.

GPU builds go in an optional pixi environment named `gpu`, which PointillViz uses for GPU runs; without one, GPU runs use `default`. From [examples/gpu-probe](examples/gpu-probe):

```toml
[dependencies]
python = ">=3.10"
numpy = "*"
scipy = "*"
pytorch = "*"                         # the CPU build, in `default`

[feature.gpu]
platforms = ["linux-64", "win-64"]

[feature.gpu.system-requirements]
cuda = "12"

[feature.gpu.dependencies]
pytorch = { version = "*", build = "cuda*" }

[environments]
gpu = ["gpu"]
```

Recent pixi versions call `[system-requirements]` deprecated; it still works. CI installs only the `default` environment.

## 8. The gates

`python -m pointillviz_tool validate` runs nine gates, in this order. PointillViz runs the same gates before it installs a tool, and CI runs them on every tool a pull request touches. Every run in the gates uses the CPU, and none has a time or memory limit: how long a tool takes and how much memory it uses are its author's to manage, and so is the reproducibility of its results.

| Gate | What it checks |
|---|---|
| `bundle` | the bundle's shape: `plugin.json` at the root, safe paths, no symbolic links or hidden files, the allowed file types, at most 300 files, 25 MB packed and 100 MB unpacked, and a `preview.svg` of at most 512 kB |
| `manifest` | `plugin.json` against the rules of [plugin.json](#3-pluginjson), with the declared defaults tried as the form would submit them. It also notes a module-level dict of defaults in `tool.py` that disagrees with the manifest |
| `environment` | `pixi.lock` exists and still matches `pixi.toml` (`pixi lock --check`); `pixi.toml` declares python, numpy and scipy (and r-base and r-jsonlite for an R tool); this computer's platform is among its platforms. It installs nothing |
| `entry` | the entry file loads in the tool's environment, in a separate process, and defines `run`, callable as `run(ds, params)`. For R, `Rscript` sources the file, and `run` must be a function of at least two arguments |
| `imports` | advisory, never fails: it lists, by file and line, any use of the network, other programs, file writing, dynamic code or environment variables, for whoever decides whether to install the tool |
| `fixture` | a run on the defaults, on the synthetic dataset below, completes: no crash, and no refusal of its own defaults. A declared sweep is run as well |
| `result` | that run returned every declared output and nothing undeclared, each in a valid shape, and each painted output (`cells`, `labels`, `vectors`, `series`, `matrix`, `radar`, `chord`, `images`) has something to show |
| `edge` | runs at the two corners of the parameters: every number at its minimum, booleans off and menus at their first option; then every number at its maximum, booleans on and menus at their last option. A refusal with `PluginError` (`pv_error`) is a pass; a crash or a malformed result is not |
| `no_mutation` | the dataset is unchanged after every run; only `plugin_cache/` may be written |

Every gate but `imports` is required. A tool without a `pixi.toml` runs in PointillViz's own environment and skips `environment`, but the registry accepts a tool only with its `pixi.toml` and `pixi.lock`.

The synthetic dataset is small and awkward on purpose: 600 cells in 2 sections, 8 cells with no coordinates, 6 cell types of which one has no cells, 12 unlabelled cells, 40 genes of which one is expressed nowhere, an extra categorical column with a missing level, and two continuous columns. A tool that has only met clean data fails here rather than on the first real slide. Dataset parameters are filled from it as the form would fill them, and two parameters of the same type get different values.

The report says what each gate found and what to change. `--json` prints it as JSON, and `validate` exits with 0 only when every required gate passes.

## 9. The commands

```
python -m pointillviz_tool new <id> [--r] [-o DIR]
python -m pointillviz_tool validate <dir|bundle> [--json] [--env own|app] [--device cpu|cuda]
python -m pointillviz_tool pack <dir> [-o OUT]
python -m pointillviz_tool run <dir> --dataset <dir> [--params JSON] [--env own|app] [--device cpu|cuda]
```

- `new` writes a working starter tool in `DIR/<id>/` (by default in the current folder): a Python tool, or a native R tool with `--r`.
- `validate` runs the gates on a tool folder or a `.pvtool` bundle. `--env own`, the default for a tool with a `pixi.toml`, builds the tool's environment from its lock file and runs the tool there; `--env app` runs it with the current Python, which is how CI runs it from inside the tool's environment. `--device` chooses the device of the gate runs (default `cpu`).
- `pack` writes `<id>-<version>.pvtool` beside the folder, or at `OUT`.
- `run` runs the tool on a converted dataset folder and prints PointillViz's reply as JSON. `--params` takes a JSON object; parameters it leaves out keep their defaults, and dataset parameters (a gene, a column) go in it too.

Each command exits with 0 when it did what was asked, 1 when it did not, and 2 for a usage error. Ctrl-C stops a running tool.

## 10. Publishing

The registry takes tools by pull request, and merges them automatically when their checks pass. Nobody reviews a tool before it is published: the gates make sure it works, not that it is safe.

1. Fork [lamanno-epfl/pointillviz-registry](https://github.com/lamanno-epfl/pointillviz-registry) and create a branch.
2. Put the tool in `tools/<id>/`, with its `pixi.toml` and `pixi.lock`.
3. Check it as CI will:
   ```bash
   python -m pointillviz_tool validate tools/<id>
   python3 scripts/check_hygiene.py
   ```
4. Open the pull request into `main`. CI runs:
   - the SDK's tests;
   - the hygiene check: the repository is public, so no email addresses (an `author` is a name) and no paths from your own computer;
   - the version check: a tool that is already released needs a higher version, one that no release has used (see [Versions](#11-versions));
   - the gates of your tool, in its `default` environment exactly as `pixi.lock` pins it, on the CPU. The report is printed in the job's log and attached to the run as `report-<id>`.
5. When every check passes, the pull request is merged automatically if:
   - it adds, changes or removes files in `tools/<id>/` and nowhere else: one tool per pull request;
   - `plugin.json` there gives the id `<id>`;
   - `<id>` is not reserved for a tool listed as Coming soon;
   - the tool is new, or you own it. Whoever first publishes a tool owns it, and only its owners can change or remove it; `owners.json` lists them by GitHub account. The registry's own tools are changed by its maintainers;
   - it is not a draft.

   A pull request that changes anything else waits for a maintainer, and a comment on it says why.
6. After the merge, the release workflow packs the tool, runs the gates on the bundle, releases it as `<id>-v<version>` with `<id>-<version>.pvtool` attached, and adds it to `index.json`. PointillViz's store lists it a few minutes later, and store.pointillviz.com within the hour.

To take a tool out of the store, its owner opens a pull request that deletes its folder. Its releases stay on GitHub, and its id stays its owner's.

A change to `sdk/`, `scripts/` or the workflows runs the gates of every tool again, and of the unlisted examples in `examples/`.

## 11. Versions

`version` is `MAJOR.MINOR.PATCH`. Only a tool's owners can publish a new version of it. A released bundle never changes, so any change to a released tool's folder (its code, manifest, README, preview or environment) needs a higher version, or the version check fails. Raise PATCH for a fix, MINOR for an addition such as a new parameter or output, and MAJOR for a change that breaks what people relied on, such as a removed parameter or a different statistic.

The index lists each tool's latest version, and earlier versions stay available as GitHub releases. A new version that fails the gates is not released, and the store keeps offering the version before it. The index describes a released version as it was released: a change to its folder without a new version is not published.
