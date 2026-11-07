"""
The packed layout of gene_counts.bin (and of every file in that layout: sparse
modalities `mod_<key>_counts.bin`, atlas shards), and its encoder and decoder.

A file holds one block per gene, back to back, in gene order. `<stem>_index.json`
marks the layout with `"layout": "packed"`, gives the file's `value_dtype` ("u16"
or "f32") and, per gene, its block's byte `offsets` and `lengths`, its `nnz` and
its two escape counts `n_esc_idx` and `n_esc_val` (a prefix is placed by them,
without reading the block's header first), and carries the file's `checksum`,
the hex SHA-1 of its bytes, which the browser's cache key is built from. A gene
with no non-zero entry has an empty block: nnz 0 and length 0, never one 0
without the other.

One block, little-endian, for `nnz` entries (cell index, value):

    u32  n_esc_idx                  how many gaps are escaped
    u32  n_esc_val                  how many values are escaped (0 for f32 data)
    u32  esc_idx[n_esc_idx]         the cell index of each escaped entry, in order
    u16  esc_val[n_esc_val]         the value of each escaped entry, in order
    u8   d[nnz]                     gap to the previous cell index, the previous
                                    being 0 before the first entry; 255 = escaped:
                                    the cell index is the next esc_idx
    values:
      u16 data:  u8  v[nnz]         the value; 255 = escaped: the next esc_val
      f32 data:  f32 v[nnz]         the value as stored, starting at the next
                                    4-byte boundary inside the block (zero padding)

Cell indices are written in the order given (ascending, as CSC writes them); a
gap that is negative or 255 or more is escaped, so ANY order round-trips. A u16
value of 255 or more is escaped, so every u16 value round-trips. The same layout
is what the browser's pathway store keeps in memory (frontend/js/pathwayGenes.js),
and frontend/js/genecodec.js decodes it in the browser. sdk/pointillviz_tool/
_genecodec.py is a copy of this module (the tool SDK runs without the backend);
tests/test_genecodec.py checks the copies are identical.

Decoding gives exactly the arrays of the cell indices and values that were
encoded: uint32 indices in their order, values in their stored dtype (uint16 or
float32). A prefix of the first `cut` entries (an atlas level of detail) decodes
from the header, the escapes and the first `cut` bytes of d and of v.
"""

from __future__ import annotations

import struct

import numpy as np

LAYOUT = "packed"
ESC = 255
HEADER_BYTES = 8
VALUE_DTYPES = ("u16", "f32")
_HEAD = struct.Struct("<II")


class GeneLayoutError(Exception):
    """A gene file not in the packed layout, or a block that does not decode. Not a
    ValueError on purpose: readers catch ValueError broadly, and this must not be
    swallowed into an empty gene."""


def check_index(gi: dict, what: str = "gene_index.json") -> None:
    """Refuse an index that is not in the packed layout, or that lacks the escape
    counts of every gene, with what to do about it. frontend/js/genecodec.js
    checkIndex makes the same checks and also requires the `checksum`, which only
    the browser uses."""
    if gi.get("layout") != LAYOUT:
        raise GeneLayoutError(
            f"{what} is not in the packed gene layout (layout={gi.get('layout')!r}). "
            "This dataset was converted before the layout changed: convert it again "
            "from its source .h5ad or SpatialData store.")
    if gi.get("value_dtype") not in VALUE_DTYPES:
        raise GeneLayoutError(f"{what}: unsupported value_dtype {gi.get('value_dtype')!r}")
    try:
        n = len(gi["nnz"])
        counted = len(gi["n_esc_idx"]) == n and len(gi["n_esc_val"]) == n
    except (KeyError, TypeError):
        counted = False
    if not counted:
        raise GeneLayoutError(
            f"{what} has no escape counts for every gene (n_esc_idx, n_esc_val): "
            "convert this dataset again from its source .h5ad or SpatialData store.")


def _values_at(d_at: int, nnz: int, value_dtype: str) -> int:
    end = d_at + nnz
    return (end + 3) & ~3 if value_dtype == "f32" else end


def block_layout(nnz: int, n_esc_idx: int, n_esc_val: int, value_dtype: str) -> dict:
    """Byte positions inside one block: where each part starts, and the block's length."""
    esc_val_at = HEADER_BYTES + 4 * n_esc_idx
    d_at = esc_val_at + 2 * n_esc_val
    v_at = _values_at(d_at, nnz, value_dtype)
    end = v_at + (4 * nnz if value_dtype == "f32" else nnz)
    return {"esc_idx_at": HEADER_BYTES, "esc_val_at": esc_val_at, "d_at": d_at, "v_at": v_at, "end": end}


def _exactly(values, dtype) -> np.ndarray:
    """`values` as `dtype`, refusing any value the cast would change: the writer
    decides the file's dtype and casts; the encoder never changes a value."""
    v = np.asarray(values)
    if v.dtype == dtype:
        return v
    out = v.astype(dtype)
    if not np.array_equal(out, v, equal_nan=np.issubdtype(dtype, np.floating)):
        raise GeneLayoutError(f"values of dtype {v.dtype} do not fit {np.dtype(dtype).name} exactly")
    return out


def encode_block(indices, values, value_dtype: str) -> tuple[bytes, int, int]:
    """(block bytes, n_esc_idx, n_esc_val) for one gene. `indices`: cell indices
    (any integer dtype, values 0..2**32-1, in the order to keep); `values`: the
    values, already in the file's value dtype (uint16 for "u16", float32 for "f32")."""
    if value_dtype not in VALUE_DTYPES:
        raise GeneLayoutError(f"unsupported value_dtype {value_dtype!r}")
    idx = np.asarray(indices)
    nnz = int(idx.shape[0])
    if nnz == 0:
        return b"", 0, 0
    idx64 = idx.astype(np.int64, copy=False)
    gaps = np.diff(idx64, prepend=0)
    gesc = (gaps < 0) | (gaps >= ESC)
    d = np.where(gesc, ESC, gaps).astype(np.uint8)
    esc_idx = idx64[gesc].astype("<u4")
    if value_dtype == "u16":
        vals = _exactly(values, np.uint16)
        vesc = vals >= ESC
        v = np.where(vesc, ESC, vals).astype(np.uint8)
        esc_val = vals[vesc].astype("<u2")
        body = d.tobytes() + v.tobytes()
    else:
        vals = _exactly(values, np.float32).astype("<f4", copy=False)
        esc_val = np.zeros(0, dtype="<u2")
        d_at = HEADER_BYTES + 4 * esc_idx.size
        pad = _values_at(d_at, nnz, "f32") - (d_at + nnz)
        body = d.tobytes() + b"\0" * pad + vals.tobytes()
    head = np.array([esc_idx.size, esc_val.size], dtype="<u4").tobytes()
    return head + esc_idx.tobytes() + esc_val.tobytes() + body, int(esc_idx.size), int(esc_val.size)


def decode_block(buf, nnz: int, value_dtype: str, cut: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(indices uint32, values uint16|float32) of one block, or of its first `cut`
    entries. `buf`: the block's bytes (bytes, bytearray, memoryview or a uint8
    array); for a prefix it may stop after the first `cut` values. A block whose
    length or escape counts do not add up raises GeneLayoutError."""
    # Written for a small per-call cost (an atlas reads one block per shard per
    # gene): the header by struct, the layout inline, u32 arithmetic throughout.
    if value_dtype == "u16":
        f32 = False
    elif value_dtype == "f32":
        f32 = True
    else:
        raise GeneLayoutError(f"unsupported value_dtype {value_dtype!r}")
    n = nnz if cut is None else max(0, min(int(cut), nnz))
    if n == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32 if f32 else np.uint16)
    size = len(buf) if not isinstance(buf, memoryview) else buf.nbytes
    if size < HEADER_BYTES:
        raise GeneLayoutError(f"block of {size} bytes has no header")
    ne, nv = _HEAD.unpack_from(buf, 0)
    if ne > nnz or nv > nnz or (f32 and nv):
        raise GeneLayoutError(f"block header ({ne}, {nv}) does not fit {nnz} entries")
    esc_val_at = HEADER_BYTES + 4 * ne
    d_at = esc_val_at + 2 * nv
    if f32:
        v_at = (d_at + nnz + 3) & ~3
        end = v_at + 4 * nnz
        vneed = v_at + 4 * n
    else:
        v_at = d_at + nnz
        end = v_at + nnz
        vneed = v_at + n
    if (n == nnz and size != end) or size < vneed:
        raise GeneLayoutError(f"block of {size} bytes, expected {end} for {nnz} entries")
    d = np.frombuffer(buf, np.uint8, n, d_at)
    g = d.astype(np.uint32)
    if ne:
        pos = (d == ESC).nonzero()[0]
        k = pos.size
        if k > ne or (n == nnz and k != ne):
            raise GeneLayoutError(f"{k} escaped gaps against {ne} in the header")
        if k:
            esc_idx = np.frombuffer(buf, "<u4", k, HEADER_BYTES)
            g[pos] = 0
            # The run before escape j is [p_(j-1), p_j) ([0, p_0) for the first);
            # it starts from the previous escape's cell index (0 before the first).
            # The true gap there is esc_idx[j] - (start + run), in u32 arithmetic:
            # wrapping exactly as the cumsum below does.
            bounds = np.empty(k + 1, dtype=np.intp)
            bounds[0] = 0
            bounds[1:] = pos
            run = np.add.reduceat(g, bounds)[:k]
            start = np.empty(k, dtype=np.uint32)
            start[0] = 0
            start[1:] = esc_idx[:-1]
            g[pos] = esc_idx - (start + run)
    elif d.max(initial=0) == ESC:
        raise GeneLayoutError(f"escaped gaps against {ne} in the header")
    g.cumsum(dtype=np.uint32, out=g)
    if f32:
        return g, np.frombuffer(buf, "<f4", n, v_at).astype(np.float32)
    v8 = np.frombuffer(buf, np.uint8, n, v_at)
    vals = v8.astype(np.uint16)
    if nv:
        vpos = (v8 == ESC).nonzero()[0]
        m = vpos.size
        if m > nv or (n == nnz and m != nv):
            raise GeneLayoutError(f"{m} escaped values against {nv} in the header")
        if m:
            vals[vpos] = np.frombuffer(buf, "<u2", m, esc_val_at)
    elif v8.max(initial=0) == ESC:
        raise GeneLayoutError(f"escaped values against {nv} in the header")
    return g, vals


def read_gene(fh, gi: dict, gene_idx: int, cut: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(indices uint32, values uint16|float32) of one gene of an open file in the
    packed layout, or its first `cut` entries. Reads only the bytes it needs: the
    whole block, or for a prefix one read from the block's start to its `cut`-th
    value, placed by the index's escape counts (check_index requires them)."""
    nnz = int(gi["nnz"][gene_idx])
    length = int(gi["lengths"][gene_idx])
    vdt = np.uint16 if gi["value_dtype"] == "u16" else np.float32
    if (nnz == 0) != (length == 0):
        raise GeneLayoutError(f"gene {gene_idx}: {nnz} entries with a {length}-byte block")
    n = nnz if cut is None else max(0, min(int(cut), nnz))
    if n == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=vdt)
    off = int(gi["offsets"][gene_idx])
    fh.seek(off)
    if n == nnz:
        buf = fh.read(length)
        if len(buf) != length:
            raise GeneLayoutError(f"gene {gene_idx}: short read ({len(buf)} of {length} bytes)")
        return decode_block(buf, nnz, gi["value_dtype"])
    try:
        ne, nv = int(gi["n_esc_idx"][gene_idx]), int(gi["n_esc_val"][gene_idx])
    except (KeyError, IndexError, TypeError):
        raise GeneLayoutError(f"gene {gene_idx}: the index has no escape counts to place a prefix by") from None
    if not (0 <= ne <= nnz and 0 <= nv <= nnz):
        raise GeneLayoutError(f"gene {gene_idx}: escape counts ({ne}, {nv}) do not fit {nnz} entries")
    lay = block_layout(nnz, ne, nv, gi["value_dtype"])
    if lay["end"] != length:
        raise GeneLayoutError(f"gene {gene_idx}: escape counts imply {lay['end']} bytes, the index says {length}")
    want = lay["v_at"] + n * (4 if gi["value_dtype"] == "f32" else 1)
    buf = fh.read(want)
    if len(buf) != want:
        raise GeneLayoutError(f"gene {gene_idx}: short read of a {n}-entry prefix")
    if _HEAD.unpack_from(buf, 0) != (ne, nv):
        raise GeneLayoutError(f"gene {gene_idx}: block header does not match the index's escape counts")
    return decode_block(buf, nnz, gi["value_dtype"], cut=n)


# Blocks of at least this many entries are decoded one at a time (decode_block is
# then a few ns per entry); smaller ones together, in vectorised batches, where a
# per-block call would cost far more in fixed overhead than in decoding.
BULK_MAX_NNZ = 4096
BULK_BATCH_ENTRIES = 1 << 21


def _gather_u32(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Little-endian u32 at each byte position `pos` of `arr`."""
    return (arr[pos].astype(np.uint32) | (arr[pos + 1].astype(np.uint32) << 8)
            | (arr[pos + 2].astype(np.uint32) << 16) | (arr[pos + 3].astype(np.uint32) << 24))


def _decode_batch(arr, offs, n, ne, nv, v_at, value_dtype):
    """Concatenated (indices uint32, values) of small blocks: one pass for all of
    them, the same arithmetic as decode_block (gaps summed from 0 in each block,
    an escape restarting from its stored cell index)."""
    nb = int(n.size)
    starts = np.zeros(nb, dtype=np.int64)
    np.cumsum(n[:-1], out=starts[1:])
    total = int(starts[-1] + n[-1])
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, n)
    d = arr[np.repeat(offs + HEADER_BYTES + 4 * ne + 2 * nv, n) + within]
    g = d.astype(np.uint32)
    esc = np.flatnonzero(d == ESC)
    blk = np.repeat(np.arange(nb), n)
    if not np.array_equal(np.bincount(blk[esc], minlength=nb), ne):
        raise GeneLayoutError("a block's escaped gaps do not match its header")
    E = int(esc.size)
    if E:
        estarts = np.zeros(nb, dtype=np.int64)
        np.cumsum(ne[:-1], out=estarts[1:])
        ev = _gather_u32(arr, np.repeat(offs + HEADER_BYTES, ne) + 4 * (np.arange(E, dtype=np.int64) - np.repeat(estarts, ne)))
        g[esc] = 0
    c = np.cumsum(g, dtype=np.uint32)
    # Anchors: each block's first entry (unless escaped) restarts the sum from 0,
    # each escape from its stored index. base = anchor value - running sum there,
    # carried to every entry up to the next anchor; idx = running sum + base, all
    # in u32 arithmetic (wrapping), as decode_block's cumsum does.
    first = starts[d[starts] != ESC]
    pos = np.concatenate((first, esc))
    base = np.concatenate(((d[first].astype(np.uint32) - c[first]), (ev - c[esc]) if E else np.zeros(0, np.uint32)))
    order = np.argsort(pos, kind="stable")
    pos, base = pos[order], base[order]
    adj = np.zeros(total, dtype=np.uint32)
    adj[pos] = np.diff(base, prepend=np.uint32(0))
    idx = c + np.cumsum(adj, dtype=np.uint32)
    if value_dtype == "u16":
        v8 = arr[np.repeat(v_at, n) + within]
        vals = v8.astype(np.uint16)
        vpos = np.flatnonzero(v8 == ESC)
        if not np.array_equal(np.bincount(blk[vpos], minlength=nb), nv):
            raise GeneLayoutError("a block's escaped values do not match its header")
        V = int(vpos.size)
        if V:
            vstarts = np.zeros(nb, dtype=np.int64)
            np.cumsum(nv[:-1], out=vstarts[1:])
            vp = np.repeat(offs + HEADER_BYTES + 4 * ne, nv) + 2 * (np.arange(V, dtype=np.int64) - np.repeat(vstarts, nv))
            vals[vpos] = arr[vp].astype(np.uint16) | (arr[vp + 1].astype(np.uint16) << 8)
    else:
        vals = _gather_u32(arr, np.repeat(v_at, n) + 4 * within).view(np.float32)
    return idx, vals


def decode_blocks(buf, gi: dict, genes=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Many genes of one file at once: (indptr int64, indices uint32, values
    uint16|float32), the genes' entries concatenated in the order of `genes`
    (default: every gene), gene j's at indptr[j]:indptr[j+1]. `buf`: the whole
    file (bytes, a uint8 array or memmap). Exactly decode_block's arrays, with the
    same checks, without its per-call cost on many small blocks."""
    vd = gi["value_dtype"]
    if vd not in VALUE_DTYPES:
        raise GeneLayoutError(f"unsupported value_dtype {vd!r}")
    arr = buf if isinstance(buf, np.ndarray) else np.frombuffer(buf, dtype=np.uint8)
    sel = np.arange(len(gi["nnz"]), dtype=np.int64) if genes is None else np.asarray(list(genes), dtype=np.int64)
    nnz = np.asarray(gi["nnz"], dtype=np.int64)[sel]
    lengths = np.asarray(gi["lengths"], dtype=np.int64)[sel]
    offs = np.asarray(gi["offsets"], dtype=np.int64)[sel]
    if np.any((nnz == 0) != (lengths == 0)):
        g = int(sel[np.flatnonzero((nnz == 0) != (lengths == 0))[0]])
        raise GeneLayoutError(f"gene {g}: {gi['nnz'][g]} entries with a {gi['lengths'][g]}-byte block")
    if np.any(offs + lengths > arr.size):
        raise GeneLayoutError("a block runs past the end of the file")
    ne = np.asarray(gi["n_esc_idx"], dtype=np.int64)[sel]
    nv = np.asarray(gi["n_esc_val"], dtype=np.int64)[sel]
    rel_d = HEADER_BYTES + 4 * ne + 2 * nv
    rel_v = ((rel_d + nnz + 3) & ~3) if vd == "f32" else rel_d + nnz
    end = rel_v + (4 * nnz if vd == "f32" else nnz)
    live = nnz > 0
    if np.any(live & ((ne > nnz) | (nv > nnz) | (end != lengths))) or (vd == "f32" and np.any(nv)):
        g = int(sel[np.flatnonzero(live & ((ne > nnz) | (nv > nnz) | (end != lengths)))[0]]) if np.any(live & ((ne > nnz) | (nv > nnz) | (end != lengths))) else -1
        raise GeneLayoutError(f"gene {g}: the index's escape counts or length do not fit its block")
    hpos = offs[live]
    if hpos.size and not (np.array_equal(_gather_u32(arr, hpos), ne[live].astype(np.uint32))
                          and np.array_equal(_gather_u32(arr, hpos + 4), nv[live].astype(np.uint32))):
        raise GeneLayoutError("a block header does not match the index's escape counts")
    indptr = np.zeros(sel.size + 1, dtype=np.int64)
    np.cumsum(nnz, out=indptr[1:])
    total = int(indptr[-1])
    indices = np.empty(total, dtype=np.uint32)
    values = np.empty(total, dtype=np.uint16 if vd == "u16" else np.float32)
    big = np.flatnonzero(nnz >= BULK_MAX_NNZ)
    for j in big:
        o, L = int(offs[j]), int(lengths[j])
        i, v = decode_block(arr[o:o + L], int(nnz[j]), vd)
        indices[indptr[j]:indptr[j + 1]] = i
        values[indptr[j]:indptr[j + 1]] = v
    small = np.flatnonzero(live & (nnz < BULK_MAX_NNZ))
    k = 0
    while k < small.size:
        # a batch of up to BULK_BATCH_ENTRIES entries (at least one block)
        cum = np.cumsum(nnz[small[k:]])
        m = max(1, int(np.searchsorted(cum, BULK_BATCH_ENTRIES, side="right")))
        part = small[k:k + m]
        idx, vals = _decode_batch(arr, offs[part], nnz[part], ne[part], nv[part], offs[part] + rel_v[part], vd)
        # scatter each block's run to its place in gene order
        dest = np.repeat(indptr[part], nnz[part]) + (np.arange(idx.size, dtype=np.int64)
                                                     - np.repeat(np.concatenate(([0], np.cumsum(nnz[part])[:-1])), nnz[part]))
        indices[dest] = idx
        values[dest] = vals
        k += m
    return indptr, indices, values
