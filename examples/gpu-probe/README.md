# GPU Probe

Answers the question: **when I choose GPU or CPU for a tool, does the tool really run there?**

The probe reports whether PyTorch can see a GPU, names the GPU, and times the product of two random square matrices on the device the run was given. A run on the GPU that cannot see one is refused with an error rather than quietly moved to the CPU.

**Table.** The requested device, whether CUDA is available, the GPU's name, the PyTorch version and the CUDA version it was built for, the pixi environment the run used, and the time and throughput of one matrix product.

## How the device choice works

`plugin.json` says `"gpu": "optional"`, so PointillViz shows a CPU / GPU switch for this tool. Its `pixi.toml` declares two environments:

- `default`, used for CPU runs. It is solved without a CUDA driver, so it gets the CPU build of PyTorch. A CPU run also hides every GPU from the process.
- `gpu`, used for GPU runs. A feature adds a CUDA 12 system requirement and asks for a CUDA build of PyTorch.

A tool that has no `gpu` environment runs in `default` on either device. The tool reads the device as `ds.device` (`"cpu"` or `"cuda"`).

## Running it

This is an example, not a published tool. It is not listed in the store. To try it from the root of a checkout of the registry, with the SDK on `PYTHONPATH` and a Python that has numpy and scipy:

```
PYTHONPATH=sdk python -m pointillviz_tool run examples/gpu-probe --dataset <dataset folder> --device cuda
PYTHONPATH=sdk python -m pointillviz_tool run examples/gpu-probe --dataset <dataset folder> --device cpu
```

To install it in PointillViz, pack it (`PYTHONPATH=sdk python -m pointillviz_tool pack examples/gpu-probe`) and use **Install from file**. The `gpu` environment holds CUDA libraries, several gigabytes, which are downloaded the first time it is built.
