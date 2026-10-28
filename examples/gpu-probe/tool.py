"""Whether this tool can see a GPU, and how fast a matrix product runs on the device it was given.

An example rather than a published tool: it exists so that the device switch
can be tested end to end. On the GPU, PointillViz starts it in the tool's
`gpu` pixi environment, which has a CUDA build of PyTorch. On the CPU it uses
`default`, which has a CPU build, and every GPU is hidden from the process.
The table says which of those actually happened.
"""

import os
import time

import torch

from pointillviz_tool import PluginError

#: Products timed after the warm-up. The first product pays for kernel
#: selection and, on the GPU, for creating the CUDA context, so it is not one
#: of them.
REPEATS = 3


def run(ds, params):
    n = int(params["size"])
    seen = torch.cuda.is_available()
    if ds.device == "cuda" and not seen:
        # Refused rather than quietly run on the CPU: a GPU run that falls
        # back without saying so is exactly what this tool is here to catch.
        raise PluginError(
            "no_gpu",
            f"This run asked for the GPU, but PyTorch {torch.__version__} "
            f"(CUDA build: {torch.version.cuda or 'none'}) cannot see one.")
    device = torch.device(ds.device)
    gpu_name = torch.cuda.get_device_name(0) if seen else "none"

    ds.progress(0.2, f"multiplying two {n} x {n} matrices on {ds.device}")
    torch.manual_seed(0)
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    def finish():
        # GPU work is queued, not done, when the call returns.
        if device.type == "cuda":
            torch.cuda.synchronize()

    a @ b
    finish()
    started = time.perf_counter()
    for _ in range(REPEATS):
        a @ b
    finish()
    seconds = (time.perf_counter() - started) / REPEATS
    gflops = 2.0 * n ** 3 / seconds / 1e9

    workspace = os.environ.get("PIXI_WORKSPACE_NAME") or os.environ.get("PIXI_PROJECT_NAME")
    environment = os.environ.get("PIXI_ENVIRONMENT_NAME")
    where = f"{workspace} ({environment})" if workspace else "not a pixi environment"
    run_on = f"GPU run on {gpu_name}" if device.type == "cuda" else "CPU run"

    rows = [
        ["Requested device", ds.device],
        ["CUDA available", "yes" if seen else "no"],
        ["GPU", gpu_name],
        ["PyTorch", torch.__version__],
        ["PyTorch CUDA build", torch.version.cuda or "none (CPU build)"],
        ["pixi environment", where],
        ["Matrix size", f"{n} x {n}"],
        ["Time per product", f"{seconds * 1e3:.1f} ms"],
        ["Throughput", f"{gflops:,.0f} GFLOP/s"],
    ]
    summary = (f"**{run_on}:** a {n} x {n} matrix product took "
               f"{seconds * 1e3:.1f} ms ({gflops:,.0f} GFLOP/s). PyTorch "
               f"{'can' if seen else 'cannot'} see a GPU in this process.")
    return {
        "summary": summary,
        "table": {"title": "GPU probe", "columns": ["Check", "Value"], "rows": rows},
    }
