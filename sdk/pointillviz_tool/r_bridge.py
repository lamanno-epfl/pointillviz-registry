"""Native R tools: R runs the tool, this side serves it the dataset.

A native R tool is a `tool.R` that defines `run <- function(ds, params)` and
returns a named list of outputs, exactly like a Python tool's dict:

    run <- function(ds, params) {
      types <- ds$cell_types()          # list(levels = chr, codes = int, NA = unlabelled)
      nn    <- ds$knn(params$k)         # list(idx = int matrix, dist = dbl matrix)
      ds$log("hello"); ds$progress(0.5, "half")
      if (bad) pv_error("bad_input", "A sentence the user sees.")
      list(summary = "...", labels = list(values = factor(...), label = "Niche"))
    }

How a run goes
--------------
The runner has already built the Python `Dataset` as for any tool. `run`
here starts

    Rscript --vanilla <sdk>/pointillviz_tool/r/runner.R <to python> <from python> <tool dir> <entry>

in the same environment (under `pixi run`, Rscript is on PATH), and serves it
until it answers. R's half sources the tool, asks for the run's facts and
parameters, and calls `run(ds, params)` with an R `ds` whose methods are
requests to this process, so the data is read once, by the same code, for
both languages. One R process per call: a sweep starts R once per value,
while the dataset handle, and the neighbour graphs it has built, stay here.

The wire
--------
The protocol travels on two pipes of its own, which R opens by the /dev/fd
paths it is given. R's stdout is this process's stderr and its stdin is
empty. So whatever reaches stdout without going through runner.R (print, cat,
the output of system(), a child process, printf in compiled code) is a log
line: it can neither forge a result nor put R's calls and the replies out of
step, and nothing R starts can read a reply meant for R. Windows cannot hand
a child extra descriptors, so there the paths are "-" and the channel is R's
own stdout and stdin; runner.R keeps R's own output off it, but output written
to the descriptor itself (system(), a child process) can reach it.

R to Python, one JSON object per line:

    {"call": "<method>", "args": {...}}      a ds method
    {"t": "log", "m": ...}                  ds$log()
    {"t": "progress", "f": ..., "m": ...}   ds$progress()
    {"t": "result", "result": {...}}         finally, or
    {"t": "error", "code", "message", "traceback"?}

Python to R, one line per call:

    {"ok": true, "value": ...}                                  small values, inline
    {"ok": true, "file": ..., "dtype": "f8|f4|i4", "shape": [n, k]}
                                                                an array: raw
        little-endian bytes in a temp file, in R's column-major order, which R
        reads with readBin
    {"ok": true, "value": {...}, "arrays": {name: {file, dtype, shape}}}
                                                                a list holding arrays
    {"ok": false, "code", "message"}                            a refusal, which R
        raises as pv_error()

Indices and missing values
--------------------------
R counts from 1 and marks a missing value NA; the dataset counts from 0 and
marks a missing index -1. Everything handed to R that indexes something is
converted: k-NN neighbours (`ds$knn()$idx`), category codes (`codes` of
`ds$categorical()` and `ds$cell_types()`, so `levels[codes]` works), section
numbers (`ds$section()`, `ds$section_filter`) and the selection
(`ds$subset`). Each is 1-based with NA for "none". Missing numbers stay NaN,
which R's is.na() also catches, and a k-NN slot that could not be filled has
an NA index and an Inf distance.

What R returns
--------------
The same output kinds as Python, with three R conveniences:

  * `labels$values` is a factor or a character vector, with NA where a cell
    has no label. Category codes are never written by hand in R: the
    categories are the factor's levels, or the sorted distinct values of a
    character vector, or `labels$categories` when given, which sets their
    order. `labels$colors` follows the categories, by position or by name.
  * a `table` may be a data.frame, or have one as its `rows`.
  * vectors of length one stay arrays where the result needs an array (one
    category, one table column), which jsonlite would otherwise unbox.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading

import numpy as np

from .dataset import PluginError

#: The R half, shipped with this package.
RUNNER_R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r", "runner.R")
#: R's NA_integer_: the one int32 bit pattern readBin reads back as NA.
INT_NA = -2 ** 31
#: How long R may take to leave once the run is over.
_EXIT_GRACE_S = 5.0
#: How often a silent R is checked for having left, and how long the lines it
#: sent just before leaving may take to be read.
_POLL_S = 0.5

_NO_RSCRIPT = ("Rscript is not on PATH. A native R tool needs r-base and r-jsonlite "
               "in its environment: add them to the tool's pixi.toml.")


class RError(Exception):
    """An R run that ended without a result or a refusal.

    `code` is "crash" (an R error that was not pv_error, with R's own call
    stack as `traceback`), "no_result" (R exited without answering) or
    "no_subprocess" (R could not be started). Not a PluginError, because none
    of these is the tool refusing an input: a sweep stops on them, as it would
    on a crash in Python.
    """

    def __init__(self, code: str, message: str, traceback: str = ""):
        super().__init__(message)
        self.code = code
        self.traceback = traceback


def run(ds, params: dict, plugin_dir: str, entry: str):
    """Run the R tool `entry` once, serving its `ds` calls from `ds`.

    Returns whatever R's run() returned, as parsed JSON, for the runner to
    check like any other result. Raises PluginError for a refusal (pv_error,
    or a ds method refusing) and RError for anything else.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RError("no_subprocess", _NO_RSCRIPT)
    tmp = tempfile.mkdtemp(prefix="pv_r_")
    try:
        try:
            r = _R(rscript, plugin_dir, entry)
        except OSError as exc:
            raise RError("no_subprocess", f"Rscript could not be started: {exc}")
        try:
            return _serve(r, ds, params, tmp)
        finally:
            r.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def probe(plugin_dir: str, entry: str) -> dict:
    """Source the R entry file without running it, and say what `run` is.

    The validator's entry gate, for R: the same runner.R sources the file the
    same way a run does. Returns {probe, ok, ...}: on success has_run,
    callable, run_type, formals (argument names), dots and seconds; on failure
    phase ("rscript", "import" or "probe") and error.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"probe": True, "ok": False, "phase": "rscript", "error": _NO_RSCRIPT}
    try:
        r = _R(rscript, plugin_dir, entry, "--probe")
    except OSError as exc:
        return {"probe": True, "ok": False, "phase": "rscript",
                "error": f"Rscript could not be started: {exc}"}
    try:
        while (raw := r.next_line()) is not None:
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("probe"):
                return obj
    finally:
        r.close()
    return {"probe": True, "ok": False, "phase": "probe", "returncode": r.proc.returncode,
            "error": f"R exited (status {r.proc.returncode}) without answering."}


class _R:
    """One R process, and the private channel to it (see "The wire" above).

    The channel is read on a thread, so that R leaving is noticed even while
    something it started in the background still holds the channel open.
    stderr is inherited: R's own output goes straight to the parent's log,
    and nothing here has to drain it.
    """

    def __init__(self, rscript: str, plugin_dir: str, entry: str, *extra: str):
        if os.name == "nt":
            self.proc = subprocess.Popen(
                [rscript, "--vanilla", RUNNER_R, "-", "-", plugin_dir, entry, *extra],
                cwd=plugin_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            reader, self._writer = self.proc.stdout, self.proc.stdin
        else:
            reader_fd, r_out = os.pipe()
            r_in, writer_fd = os.pipe()
            try:
                self.proc = subprocess.Popen(
                    [rscript, "--vanilla", RUNNER_R, f"/dev/fd/{r_out}", f"/dev/fd/{r_in}",
                     plugin_dir, entry, *extra],
                    cwd=plugin_dir, stdin=subprocess.DEVNULL, stdout=2,
                    pass_fds=(r_out, r_in))
            except BaseException:
                os.close(reader_fd)
                os.close(writer_fd)
                raise
            finally:
                # R's ends: only R holds them now, so R leaving ends the stream.
                os.close(r_out)
                os.close(r_in)
            reader, self._writer = os.fdopen(reader_fd, "rb"), os.fdopen(writer_fd, "wb")
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(reader,), daemon=True).start()

    def _pump(self, reader) -> None:
        # It closes its own end of the channel: closing it from another thread
        # would wait for this thread's read, which something R left running
        # can keep from ever returning.
        try:
            for raw in reader:
                self._lines.put(raw)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)
            try:
                reader.close()
            except OSError:
                pass

    def next_line(self):
        """The next line R sent, or None once R has closed the channel or left.

        R leaving is watched as well as the channel closing: a process R
        started in the background (`system("... &")`) inherits the channel
        and can hold it open long after R has gone.
        """
        while True:
            try:
                return self._lines.get(timeout=_POLL_S)
            except queue.Empty:
                if self.proc.poll() is not None:
                    break
        try:
            return self._lines.get(timeout=_POLL_S)
        except queue.Empty:
            return None

    def send(self, obj) -> None:
        try:
            self._writer.write((json.dumps(obj, allow_nan=False) + "\n").encode("utf-8"))
            self._writer.flush()
        except OSError:
            pass                          # R has gone; its exit is reported by the caller

    def close(self) -> None:
        # Closing R's input also ends an R still waiting for a reply: its next
        # read finds the end of the stream.
        try:
            self._writer.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=_EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            # It answered, or the run is over without it, and it did not
            # leave: nothing it could still do would be read.
            self.proc.kill()
            self.proc.wait()


def _serve(r: _R, ds, params: dict, tmp: str):
    """Answer R's requests until it sends its terminal line."""
    while (raw := r.next_line()) is not None:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            msg = None
        if not isinstance(msg, dict):
            # Where the channel is R's own stdout (Windows): output written
            # straight to the descriptor, past R's sink.
            ds.log(line)
            continue
        if "call" in msg:
            r.send(_answer(ds, params, msg, tmp))
            continue
        kind = msg.get("t")
        if kind == "log":
            ds.log(str(msg.get("m", "")))
        elif kind == "progress":
            try:
                ds.progress(float(msg.get("f")), msg.get("m"))
            except (TypeError, ValueError):
                pass
        elif kind == "result":
            return msg.get("result")
        elif kind == "error":
            code = str(msg.get("code") or "crash")
            message = str(msg.get("message") or "The R tool reported an error.")
            if code == "crash":
                raise RError("crash", message, str(msg.get("traceback") or ""))
            raise PluginError(code, message)
        else:
            ds.log(line)
    try:
        status = r.proc.wait(timeout=_EXIT_GRACE_S)
    except subprocess.TimeoutExpired:
        raise RError("no_result", "R closed its output without returning a result.")
    raise RError("no_result", f"R exited (status {status}) without returning a result.")


# ===========================================================================
# The ds methods, as R asks for them
# ===========================================================================
def _answer(ds, params: dict, msg: dict, tmp: str) -> dict:
    method = msg.get("call")
    args = msg.get("args")
    args = args if isinstance(args, dict) else {}
    handler = _METHODS.get(method)
    if method == "init":
        return _init(ds, params, tmp)
    if handler is None:
        return {"ok": False, "code": "bad_call", "message": f"ds has no method {method!r}."}
    try:
        return handler(ds, tmp, **args)
    except PluginError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    except Exception as exc:                                    # noqa: BLE001
        # A bad argument from R (a list where a name belongs). R raises this
        # as an ordinary error, so it is reported as the bug it is, with the
        # R call that made it, and not as the tool refusing an input.
        return {"ok": False, "code": "bad_call",
                "message": f"ds${method}(): {type(exc).__name__}: {exc}"}


def _init(ds, params: dict, tmp: str) -> dict:
    """What R's `ds` holds as plain fields, and the parameters."""
    reply = _value({
        "n_cells": ds.n_cells, "name": ds.name, "title": ds.title, "device": ds.device,
        "section_filter": None if ds.section_filter is None else ds.section_filter + 1,
        "meta": ds.meta, "params": params,
    })
    if ds.subset is not None:
        reply["arrays"] = {"subset": _array(_index(ds.subset), tmp)}
    return reply


def _value(value) -> dict:
    return {"ok": True, "value": _jsonable(value)}


def _jsonable(value):
    """Plain JSON for R: numpy types unwrapped, NaN and infinities as null.

    jsonlite rejects the NaN token Python's json module writes by default,
    and a dataset's meta.json can hold one (the range of an empty column).
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _array(a, tmp: str) -> dict:
    """Write an array where R can readBin it, and describe it.

    Column-major, R's own order: a matrix's transpose in C order is its
    Fortran-order bytes, which `tofile` writes without a copy of the whole
    array. Floats keep their width (f4 or f8; R reads either as double) and
    integers travel as int32, R's integer.
    """
    a = np.asarray(a)
    dtype = ("f4" if a.dtype.itemsize <= 4 else "f8") if a.dtype.kind == "f" else "i4"
    fd, path = tempfile.mkstemp(dir=tmp, suffix=".bin")
    with os.fdopen(fd, "wb") as fh:
        a.astype("<" + dtype, copy=False).T.tofile(fh)
    return {"file": path, "dtype": dtype, "shape": list(a.shape)}


def _array_reply(a, tmp: str) -> dict:
    return dict({"ok": True}, **_array(a, tmp))


def _index(a) -> np.ndarray:
    """0-based indices, -1 for none, as R's 1-based int32 with NA for none."""
    a = np.asarray(a, dtype=np.int64)
    return np.where(a >= 0, a + 1, INT_NA).astype(np.int32)


def _categorical(found, tmp: str) -> dict:
    levels, codes = found
    return {"ok": True, "value": {"levels": list(levels)},
            "arrays": {"codes": _array(_index(codes), tmp)}}


def _knn(found, tmp: str) -> dict:
    idx, dist = found
    return {"ok": True, "value": {},
            "arrays": {"idx": _array(_index(idx), tmp), "dist": _array(dist, tmp)}}


def _names(names) -> list:
    return [str(names)] if isinstance(names, str) else [str(n) for n in names]


_METHODS = {
    "xy": lambda ds, tmp: _array_reply(ds.xy, tmp),
    "coords": lambda ds, tmp: _array_reply(ds.coords, tmp),
    "section": lambda ds, tmp: _array_reply(_index(ds.section), tmp),
    "gene_names": lambda ds, tmp: _value(ds.gene_names),
    "has_gene": lambda ds, tmp, name: _value(bool(ds.has_gene(str(name)))),
    "gene": lambda ds, tmp, name, normalize="cp10k":
        _array_reply(ds.gene(str(name), str(normalize)), tmp),
    "genes": lambda ds, tmp, names, normalize="cp10k":
        _array_reply(ds.genes(_names(names), str(normalize)), tmp),
    "total_counts": lambda ds, tmp: _array_reply(ds.total_counts(), tmp),
    "categorical_fields": lambda ds, tmp: _value(ds.categorical_fields()),
    "categorical": lambda ds, tmp, name=None:
        _categorical(ds.categorical(None if name is None else str(name)), tmp),
    "cell_types": lambda ds, tmp: _categorical(ds.cell_types(), tmp),
    "continuous_fields": lambda ds, tmp: _value(ds.continuous_fields()),
    "continuous": lambda ds, tmp, name: _array_reply(ds.continuous(str(name)), tmp),
    "knn": lambda ds, tmp, k=15: _knn(ds.knn(int(k)), tmp),
}
