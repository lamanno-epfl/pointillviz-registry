# PointillViz tool registry

The community tools of [PointillViz](https://pointillviz.com), and the small SDK they are written with.

A tool is an analysis that runs on the dataset you have open in PointillViz. It reads cells, genes and annotations, and answers with a map painted on the tissue, a table, a chart or a figure. PointillViz builds the tool's form from its manifest, so a tool has no interface code of its own. Tools are written in Python or in R, and each one brings its own [pixi](https://pixi.sh) environment.

Browse them at [store.pointillviz.com](https://store.pointillviz.com).

## Installing a tool

- **In PointillViz:** open **Tools → Store**, choose a tool and click **Install**.
- **From the website:** download the tool's `.pvtool` file from its page on [store.pointillviz.com](https://store.pointillviz.com), then in PointillViz use **Tools → Install from file**.

Either way, PointillViz asks where the tool should run: in an environment of its own (recommended), or in PointillViz's own environment. pixi builds that environment exactly as the tool's lock file pins it, and PointillViz runs the tool's checks again on your computer before the tool appears in the Tools tab.

A tool is arbitrary code that runs on your computer, with your permissions. The checks make sure it works and say what it reaches for (the network, other programs, files on disk); they are not a sandbox. Install tools you are willing to run.

## What is in this repository

| Path | What it holds |
|---|---|
| `tools/<id>/` | One folder per tool: `plugin.json`, `tool.py` or `tool.R`, `pixi.toml`, `pixi.lock`, `README.md` (its store page) and an optional `preview.svg` |
| `placeholders.json` | Tools announced as Coming soon |
| `owners.json` | Who owns each community tool, by GitHub account: the author who first published it. Written by the automation |
| `sdk/` | `pointillviz_tool`, the SDK: the dataset handle, the result contract, the checks (the gates), and the `new`, `validate`, `pack` and `run` commands |
| `examples/` | Examples that are not listed in the store |
| `scripts/` | What the automation runs: `changed_tools.py`, `check_hygiene.py`, `automerge.py`, `publish.py` and `build_index.py` |
| `index.json` | The store's index: written by the release workflow, never by hand |

## How tools are published

Nothing here is maintained by hand for each tool: a tool's own folder is the only place it is described.

1. An author opens a pull request that adds or changes a folder under `tools/`. [CONTRIBUTING.md](CONTRIBUTING.md) is the guide.
2. The pull request is checked: the SDK's tests, that nothing private is committed, that a changed tool has a new version, and the gates of every tool the change touches, in that tool's own environment.
3. When the checks pass, it is merged automatically, as long as it changes one tool's folder and nothing else, and that tool is new or its author's own. Nobody reviews it first. Anything else waits for a maintainer.
4. Every tool version that has no release yet is packed, gated again as the bundle itself, and released on GitHub as `<id>-v<version>`, with `<id>-<version>.pvtool` attached.
5. `index.json` is rebuilt from the tools' manifests, their bundles and their gate reports, and committed. PointillViz's store and store.pointillviz.com both read it.

A released bundle never changes. A new version is a new release; the index lists each tool's latest version, and the earlier ones stay available as GitHub releases.
