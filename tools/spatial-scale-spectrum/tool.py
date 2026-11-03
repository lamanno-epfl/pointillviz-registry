"""Spatial scale spectrum: at what LENGTH SCALE is each population organised?

What this computes
------------------
For a population c (a level of any categorical annotation), write z_j = 1 if
cell j is of type c, p = N_c / N, and u_j = exp(-i k . x_j). The mean-subtracted
Fourier amplitude of the composition field is

    T_c(k) = sum_j (z_j - p) u_j = sum_{j in c} u_j  -  p * sum_{all j} u_j

and the tool reports the structure factor

    S_c(k) = |T_c(k)|^2 / [ p (1-p) (N/(N-1)) (N - S_all(k)) ],
                                        S_all(k) = |sum_j u_j|^2 / N

against wavelength lambda = 2 pi / |k|. S = 1 is no structure at that scale,
S > 1 is excess power (patchiness), S < 1 is suppressed fluctuation -- cells of
that type laid out more evenly than a coin toss over the same positions would
lay them.

Why the normaliser is that, and not |FFT|^2 / N_c
------------------------------------------------
The textbook estimator |sum_{j in c} u_j|^2 / N_c is dominated at small |k| by
the WINDOW: the sum has mean rho_c * FT(1_W)(k), which is O(N_c) below
2 pi / L, so a naive FFT of a binned indicator field reports the Fourier
transform of the tissue outline and calls it biology. Every standard repair --
a Bartlett taper, a support mask, truncating the k range -- is a partial one,
and each of them costs the null its exact expectation.

The estimator above conditions on the cell POSITIONS and permutes only the
LABELS. Under a uniformly random permutation of the labels over the same
positions, exactly and with no asymptotics,

    E[T_c(k)] = 0,   E|T_c(k)|^2 = p(1-p) (N/(N-1)) (N - S_all(k))

so E[S_c(k)] = 1 at every k, for any window shape, any tissue outline, any
density gradient and any k grid. The window transform enters the numerator
(through the p * sum_all term) and the denominator (through S_all) and cancels.

A random labelling of the observed positions -- no structure by construction,
the same ragged outline -- therefore reads S = 1 at every shell and every
abundance, while the naive |FFT|^2 / N_c of the same binned field grows in
proportion to N_c, because what it measures is the section boundary. That is
why this tool never tapers, never masks and never truncates the k range on
window grounds: each of those breaks E[S] = 1 and the outline comes back.

What it therefore measures is the structure factor of the COMPOSITION field --
where type c sits given where cells sit at all -- which is the biologically
answerable question and the only one with an exact null on a ragged outline.

Binning
-------
Rounding positions onto a grid of pitch h is an independent jitter of width h,
so S_binned = |phi|^2 S_true + (1 - |phi|^2) with phi(k) = sinc(kx h/2)
sinc(ky h/2), inverted per mode. Two rules keep the answer independent of the
grid:

* The LADDER is pinned to the point pattern (lam_lo = 1.5 / sqrt(rho)), never
  to the grid. A shell floor tied to the pitch would move the whole geomspace
  when `grid` changed, so `S long` -- the mean over the top quarter of the
  shells -- would integrate a different band at each resolution.
* A mode is reported only with MIN_PIX_LAM = 4 pixels per wavelength
  (|phi|^2 >= 0.81). Cells sharing a bin have IDENTICAL rounding displacements
  rather than independent ones, so the jitter model degrades near Nyquist
  faster than |phi|^2 admits, and a looser floor lets biased shells through.

What a coarse grid costs is therefore REACH, not correctness: it resolves fewer
short wavelengths, and where it cannot resolve the packing band the `S short`
column is dropped rather than refilled with a longer band under the same
heading.

The real-space half
-------------------
Enabled by default, and it is a CHECK, not a second opinion dressed as one: it
answers the same question with disjoint machinery, so a disagreement is
informative. Square windows of side R are laid over the section; the count of
type c in each window has variance growing like Var[N_c(R)] ~ R^alpha, with
alpha = 2 the area law a Poisson pattern obeys, above 2 clustering, below 2
suppressed large-scale fluctuation. Every count is four reads of a summed-area
table, so the cost is independent of R.

Two details carry it:

* A window is admitted only when EVERY pixel of it lies inside the morpho-
  logical closing of the point set. A window poking into a lumen or off the
  section is mostly empty, its variance collapses toward its mean, and the
  exponent it produces is the outline's, not the tissue's.
* The ladder stops where fewer than MIN_INDEP DISJOINT windows fit -- not
  overlapping ones. Overlapping windows are cheap and far more numerous, but
  they are correlated and do not give the same slope; a standard error quoted
  off correlated samples would be a fiction.

The null here is analytic. Given the total count n in a window, N_c is
Hypergeometric(N, n_c, n) under the same label permutation, so

    Var_null[N_c(R)] = E_W[n(N-n)/(N-1)] p(1-p) + p^2 Var_W[n]

which is built from window counts of the total population that are already in
hand. Exact in expectation, free, and deterministic. It matters because the
tissue's own crowding is scale-dependent: alpha_null is not 2.

The real-space half never aborts the run. Where the tissue admits no window
ladder -- a lasso too small, a section too fragmented -- its two columns are
dropped and the summary says so, because the spectrum is the answer and this
is the check on it.

Reading the columns
-------------------
`S short` and `Patch scale` lead the table because they are what neighbour
counts cannot see: `S short` below the null band is a population laid out more
evenly than chance, and `Patch scale` is the wavelength at which a population
is organised. `S long`, `Var / Var_null` and `alpha` corroborate the
clustering that neighbourhood statistics already report.

Cost and determinism
--------------------
One histogram plus one rfft2 per population per section, batched eight at a
time so the mode stack never materialises; the null costs n_perms passes over
at most NULL_TYPE_CAP populations, since its output is one pooled per-shell
standard deviation and 16 populations already oversample it. The permutation
seed is SHA-1 of the parameters, never `random.seed()` on the clock and never
Python's salted `hash()`, so two runs of the same form are byte-identical --
which is what a figure in a paper needs.
"""

from __future__ import annotations

import gc
import hashlib
import os

import numpy as np

from pointillviz_tool import PluginError

# --------------------------------------------------------------------------
# Reciprocal-space half
# --------------------------------------------------------------------------
N_SHELLS = 32        # log-spaced wavelength shells
MIN_MODES = 8        # a shell thinner than this is not reported
MIN_PIX_LAM = 4.0    # pixels per wavelength a mode needs to be reported
TYPE_CHUNK = 8       # populations per batched FFT call, to cap peak memory
MAX_LINES = 7        # curves drawn over the null band

#: The null's product is ONE pooled standard deviation per shell, and 16
#: populations x n_perms permutations already oversample it by an order of
#: magnitude. Without this cap the null is n_perms x n_populations FFTs: at the
#: form's most expensive settings, on an annotation with many levels, that is
#: well over a thousand transforms for a quantity that has converged after the
#: first hundred.
NULL_TYPE_CAP = 16

# --------------------------------------------------------------------------
# Real-space half
# --------------------------------------------------------------------------
GRID_MAX = 1024      # pixels on the long axis of the window grid
NN_SAMPLE = 40000    # points sampled per section for the median NN spacing
CLOSE_NN = 4.0       # support closing scale, in median-NN units
MIN_PIXELS = 3       # a window must be at least this many pixels across
MIN_WINDOWS = 24     # an R with fewer admissible windows is dropped
MIN_INDEP = 60       # ... and the ladder stops where fewer DISJOINT ones fit
MIN_SCALES = 4       # fewer usable window sizes than this -> no exponent
R_MIN_NN = 2.0       # smallest window, in median-NN units
R_MAX_FRAC = 0.25    # largest window, as a fraction of the shortest extent
N_SCALES = 12        # window sizes requested along the ladder

MIN_USABLE = 50

_BAND_COLOR = "#9aa5b1"

#: Keys are the declared parameters; the manifest owns the values the host
#: passes. This exists so a direct call from a notebook gets the same run.
DEFAULTS = {"field": None, "focus": None, "min_cells": 200, "grid": "1024",
            "real_space": True, "n_perms": 3}


# ===========================================================================
# Entry point
# ===========================================================================
def run(ds, params):
    if ds.is_3d:
        raise PluginError(
            "not_2d",
            "This tool bins to a 2D grid and lays out 2D windows; on a 3D "
            "dataset both the spectrum and the area law it is read against are "
            "the wrong ones. Use it on a 2D section.")

    field = (params.get("field") or "").strip() or None
    labels, codes = ds.categorical(field)
    grid = int(params["grid"])
    min_cells = int(params["min_cells"])
    n_perms = int(params["n_perms"])
    real_space = bool(params["real_space"])

    xy = np.asarray(ds.coords, dtype=np.float64)[:, :2]
    sect = ds.section
    usable = np.isfinite(xy).all(axis=1) & (codes >= 0)
    if ds.section_filter is not None:
        usable &= (sect == ds.section_filter)
    if ds.subset is not None:
        keep = np.zeros(ds.n_cells, dtype=bool)
        keep[ds.subset] = True
        usable &= keep
    n_usable = int(usable.sum())
    if n_usable < MIN_USABLE:
        raise PluginError(
            "too_few_cells",
            f"Only {n_usable} cells have both coordinates and a label here; a "
            f"structure factor needs at least {MIN_USABLE}.")

    counts_all = np.bincount(codes[usable], minlength=len(labels))
    if int((counts_all > 0).sum()) < 2:
        # Named for what happened. Without this the run dies later in
        # _accumulate's `nc == N` skip, leaving every shell empty, and the user
        # is told the section is too small or the grid too coarse -- neither of
        # which is true, and neither of which they can act on.
        raise PluginError(
            "one_level",
            "Every labelled cell here belongs to the same level of this "
            "annotation. A composition field is one population measured "
            "against the others, so there has to be more than one: pick a "
            "column with at least two populations in it.")
    tids = np.flatnonzero(counts_all >= min_cells)
    relaxed = False
    if tids.size == 0:
        # Refusing at a threshold nobody meets would leave the map blank at the
        # tool's own maximum corner. Score the largest population instead and
        # say so, which is a smaller sample, not a different statistic.
        tids = np.array([int(np.argmax(counts_all))])
        relaxed = True
        if counts_all[tids[0]] < 20:
            raise PluginError(
                "too_few_cells",
                f"The largest level of this annotation has {int(counts_all[tids[0]])} "
                "cells with coordinates. Nothing here is a population.")
    counts = counts_all[tids].astype(np.float64)
    n_types = tids.size

    secs = _sections(ds, xy, sect, usable, grid)
    edges, lam_mid = _shell_edges(secs)
    n_bins = edges.size - 1
    modes = np.zeros(n_bins, dtype=np.int64)
    for s in secs:
        s["keep"], s["sh"], s["ph"] = _shell_modes(grid, s["h"], edges)
        modes += np.bincount(s["sh"], minlength=n_bins)

    ds.progress(0.10, f"structure factor of {n_types} population(s) "
                      f"on a {grid}x{grid} grid")
    num = np.zeros((n_types, n_bins))
    den = np.zeros((n_types, n_bins))
    for s in secs:
        _accumulate(s, codes[s["rows"]], tids, len(labels), grid, num, den)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    ok_shell = (modes >= MIN_MODES) & np.isfinite(S).any(axis=0)
    if not ok_shell.any():
        pitch = min(s["h"] for s in secs)
        raise PluginError(
            "no_shells",
            f"No wavelength shell holds {MIN_MODES} Fourier modes to average "
            f"over. A {grid}x{grid} grid puts {pitch:.1f} units between bins "
            f"and reports nothing under {MIN_PIX_LAM * pitch:.0f}, against a "
            f"reportable range of {edges[0]:.0f}-{edges[-1]:.0f} units: either "
            "raise the binning resolution, or the section is too small for a "
            "spectrum.")
    S[:, ~ok_shell] = np.nan

    ds.progress(0.40, f"null band from {n_perms} label permutations")
    sd = _null_sd(ds, secs, codes, usable, tids, len(labels), grid, n_bins,
                  ok_shell, modes, n_perms,
                  (field, grid, min_cells, n_perms, len(labels)))

    idx_ok = np.flatnonzero(ok_shell)
    # Both bands are slices of the LADDER, not of whatever survived the
    # resolution cut, and the ladder is pinned to the point pattern -- so both
    # name the same wavelengths at every grid. Only the shortest shells: the
    # packing dip lives within about two mean cell spacings and a wider band
    # averages it straight out.
    tail = max(3, n_bins // 4)
    long_bins = np.arange(n_bins - tail, n_bins)[ok_shell[n_bins - tail:]]
    short_bins = np.arange(min(3, n_bins))
    have_short = bool(ok_shell[short_bins].all())
    S_long, sd_long = _pooled(num, den, sd, long_bins)
    if have_short:
        S_short, sd_short = _pooled(num, den, sd, short_bins)
    else:
        # A grid too coarse to resolve the short band drops the column. It does
        # not report a longer band under the same heading: on a coarse grid the
        # shortest reportable shell can lie many cell spacings out, a scale
        # where clustering dominates, and "S short" there would be the opposite
        # of the packing statistic the name promises.
        S_short = np.full(n_types, np.nan)
        sd_short = float("nan")

    alpha = _spectral_exponent(S, idx_ok, lam_mid)
    lam_half, band_bin = _patch_scale(S, S_long, sd_long, idx_ok, lam_mid)

    focus_j = _focus(labels, tids, counts, params.get("focus"))
    fb = int(band_bin[focus_j])
    focus_flat = not np.isfinite(lam_half[focus_j])
    if focus_flat:
        # Nothing rose above the null, so there is no patch scale to band-pass
        # at; take the middle of the range and call the map noise in words
        # rather than hand back a picture of the longest, thinnest shell.
        fb = int(idx_ok[len(idx_ok) // 2])
    ds.progress(0.75, f"band-passing {labels[tids[focus_j]]}")
    values = _band_map(ds, secs, codes, tids[focus_j], grid, edges[fb], edges[fb + 1])

    rs = None
    if real_space:
        ds.progress(0.85, "real-space variance ladder")
        rs = _real_space(ds, xy, sect, codes, usable, tids)

    # Both halves are done and everything they built is dead, while what comes
    # next is a per-cell payload the host has to serialise. Free the binned
    # sections and the coordinate copy first, so they do not add to the peak.
    for _s in secs:
        _s.clear()
    del secs, xy
    gc.collect()

    ds.progress(0.95, "assembling")
    return _report(ds, labels, tids, counts, lam_mid, ok_shell, S, sd, S_short,
                   S_long, lam_half, alpha, rs, focus_j, focus_flat, edges, fb,
                   values, n_usable, grid, n_perms, field, relaxed, min_cells,
                   short_bins, sd_short, modes, have_short)


# ===========================================================================
# Reciprocal space
# ===========================================================================
def _sections(ds, xy, sect, usable, grid):
    """One binned grid per section, plus the wavelength range it can report."""
    out = []
    for s in np.unique(sect[usable]):
        rows = np.flatnonzero(usable & (sect == s))
        if rows.size < 20:
            continue
        X = xy[rows]
        x0, y0 = float(X[:, 0].min()), float(X[:, 1].min())
        lx = float(X[:, 0].max()) - x0
        ly = float(X[:, 1].max()) - y0
        # One square grid over the longer side: an anisotropic pitch would give
        # kx and ky different binning corrections and the radial average would
        # mix them.
        side = max(lx, ly, 1e-9) * (1.0 + 1e-6)
        h = side / grid
        ix = np.minimum(((X[:, 0] - x0) / h).astype(np.int64), grid - 1)
        iy = np.minimum(((X[:, 1] - y0) / h).astype(np.int64), grid - 1)
        rho = rows.size / max(lx * ly, 1e-9)
        out.append({
            "rows": rows, "flat": iy * grid + ix, "h": h,
            # Pinned to the POINT PATTERN, never to the grid. A floor of
            # 3h would move the whole geomspace when `grid` changed, so `S long`
            # would integrate a different band at each resolution, with no
            # binning error involved at all. Coarseness REMOVES shells
            # (_shell_modes); it does not relocate them.
            "lam_lo": 1.5 / np.sqrt(max(rho, 1e-12)),
            "lam_hi": min(lx, ly) / 3.0,
        })
    if not out:
        raise PluginError("no_section",
                          "No section has 20 cells with both coordinates and a "
                          "label; there is nothing to take a spectrum of.")
    return out


def _shell_edges(secs):
    lam_lo = min(s["lam_lo"] for s in secs)
    lam_hi = max(s["lam_hi"] for s in secs)
    if not lam_hi > lam_lo * 1.5:
        raise PluginError(
            "no_scale_range",
            f"The reportable wavelength range is {lam_lo:.0f}-{lam_hi:.0f} units, "
            "under half an octave: these cells are too sparse for the size of "
            "the section to have any range of scale between them.")
    edges = np.geomspace(lam_lo, lam_hi, N_SHELLS + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def _kgrid(grid, h):
    fy = np.fft.fftfreq(grid, d=h)
    fx = np.fft.rfftfreq(grid, d=h)
    kmag = np.sqrt((2.0 * np.pi * fy[:, None]) ** 2 + (2.0 * np.pi * fx[None, :]) ** 2)
    # np.sinc(z) = sin(pi z) / (pi z), and the jitter transfer we want is
    # sin(k h / 2) / (k h / 2) = sinc(f h).
    phi = np.sinc(fy[:, None] * h) * np.sinc(fx[None, :] * h)
    return kmag, (phi * phi).astype(np.float32)


def _shell_modes(grid, h, edges):
    """(kept-mode mask, shell index of each kept mode, phi^2 of each)."""
    kmag, phi2 = _kgrid(grid, h)
    with np.errstate(divide="ignore"):
        lam = np.where(kmag > 0, 2.0 * np.pi / np.maximum(kmag, 1e-30), np.inf)
    shell = np.digitize(lam.ravel(), edges) - 1
    # Resolution floor in pixels per wavelength, not in phi^2. The sinc
    # inversion is exact for an isolated point, but cells that share a bin have
    # IDENTICAL rounding displacements rather than independent ones, so the
    # model degrades near Nyquist by more than phi^2 admits, and a looser floor
    # (phi^2 >= 0.40, lambda >= 1.6 h) lets shells through that read low.
    # lambda >= 4 h, the same thing as phi^2 >= 0.81, keeps every grid in the
    # dropdown in agreement with a much finer one.
    keep = ((shell >= 0) & (shell < edges.size - 1)
            & (lam.ravel() >= MIN_PIX_LAM * h))
    return keep, shell[keep].astype(np.int64), phi2.ravel()[keep]


def _workers():
    """Threads for scipy.fft: OMP_NUM_THREADS when the host sets it, else eight.

    scipy.fft's workers=-1 means os.cpu_count(), one thread per core on a
    many-core host.
    """
    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "8")))
    except ValueError:
        return 8


def _rfft2(a):
    try:
        import scipy.fft as sfft
        return sfft.rfft2(a, workers=_workers())
    except ImportError:
        return np.fft.rfft2(a)


def _accumulate(sec, codes_here, tids, n_levels, grid, num, den):
    """Pool per-mode S into the wavelength shells, for every population."""
    flat = sec["flat"]
    keep, sh, ph = sec["keep"], sec["sh"], sec["ph"]
    n_bins = den.shape[1]
    N = int(flat.size)

    order = np.argsort(codes_here, kind="stable")
    fs = flat[order]
    starts = np.searchsorted(codes_here[order], np.arange(n_levels + 1))

    allf = np.bincount(flat, minlength=grid * grid).reshape(grid, grid).astype(np.float32)
    FA = _rfft2(allf).ravel()[keep]
    del allf
    Sall = (FA.real.astype(np.float64) ** 2 + FA.imag.astype(np.float64) ** 2) / N

    for s0 in range(0, len(tids), TYPE_CHUNK):
        sub = tids[s0:s0 + TYPE_CHUNK]
        # One chunk at a time, so the full (n_types, grid, grid) stack never
        # exists at once.
        blk = np.empty((len(sub), grid, grid), dtype=np.float32)
        for j, c in enumerate(sub):
            blk[j] = np.bincount(fs[starts[c]:starts[c + 1]],
                                 minlength=grid * grid).reshape(grid, grid)
        FC = _rfft2(blk).reshape(len(sub), -1)[:, keep]
        del blk
        for j, c in enumerate(sub):
            # p is this SECTION's fraction, not the pooled one: the identity
            # E[S] = 1 is a statement about permuting labels within the set of
            # positions being transformed, and a population that is 20% of one
            # section and 2% of another has no single p.
            nc = int(starts[c + 1] - starts[c])
            if nc == 0 or nc == N:
                continue
            p = nc / N
            t = FC[j] - p * FA
            power = t.real.astype(np.float64) ** 2 + t.imag.astype(np.float64) ** 2
            d = p * (1.0 - p) * (N / (N - 1.0)) * np.maximum(N - Sall, 1e-12)
            # Invert the binning transfer per mode, not per shell: phi^2 varies
            # by a factor of two inside one shell near the grid Nyquist.
            strue = (power / d - 1.0 + ph) / ph
            num[s0 + j] += np.bincount(sh, weights=strue, minlength=n_bins)
            den[s0 + j] += np.bincount(sh, minlength=n_bins)


def _null_sd(ds, secs, codes, usable, tids, n_levels, grid, n_bins, ok_shell,
             modes, n_perms, seed_key):
    """Per-shell standard deviation of S under label permutation.

    Pooled over permutations AND populations: under the null every population
    has the same E[S] = 1 and the spread is a property of how many modes the
    shell holds, so pooling is what makes three permutations enough.
    """
    pool_ids = tids
    if pool_ids.size > NULL_TYPE_CAP:
        take = np.unique(np.linspace(0, pool_ids.size - 1, NULL_TYPE_CAP)
                         .round().astype(int))
        pool_ids = pool_ids[take]
    rng = np.random.default_rng(_seed(seed_key))
    rows_pool = np.flatnonzero(usable)
    draws = np.full((n_perms, pool_ids.size, n_bins), np.nan)
    for r in range(n_perms):
        perm = codes.copy()
        perm[rows_pool] = codes[rows_pool][rng.permutation(rows_pool.size)]
        n2 = np.zeros((pool_ids.size, n_bins))
        d2 = np.zeros((pool_ids.size, n_bins))
        for s in secs:
            _accumulate(s, perm[s["rows"]], pool_ids, n_levels, grid, n2, d2)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[r] = np.where(d2 > 0, n2 / np.maximum(d2, 1e-30), np.nan)
        ds.progress(0.40 + 0.30 * (r + 1) / n_perms,
                    f"null permutation {r + 1} of {n_perms}")
    draws[:, :, ~ok_shell] = np.nan

    flat = draws.reshape(-1, n_bins)
    sd = np.full(n_bins, np.nan)
    for b in np.flatnonzero(np.isfinite(flat).sum(axis=0) >= 2):
        v = flat[:, b]
        sd[b] = np.std(v[np.isfinite(v)], ddof=1)
    # Where the permutations gave nothing usable, fall back to the mode-count
    # scaling they would have measured: S is an average of |T|^2 over the
    # shell's modes, so its null spread goes as 1/sqrt(modes).
    return np.where(np.isfinite(sd) & (sd > 0), sd,
                    1.0 / np.sqrt(np.maximum(modes, 1)))


def _pooled(num, den, sd, bins):
    """S over a band of shells, weighted by the modes each shell contributed."""
    d = np.nansum(den[:, bins], axis=1)
    # NaN where nothing was measured, never 0/1e-30 = 0.0. A population that
    # cleared min_cells on its total but lives only in sections _sections()
    # dropped contributes no modes, and 0.0 is the most extreme value this
    # statistic can take -- it would be reported as the most evenly spaced
    # population in the section and named in the summary's lead sentence.
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(d > 0, np.nansum(num[:, bins], axis=1) / np.maximum(d, 1e-30),
                     np.nan)
    e = np.sqrt(np.nansum(sd[bins] ** 2)) / max(len(bins), 1)
    return v, max(float(e), 1e-9)


def _spectral_exponent(S, idx_ok, lam_mid):
    """Slope of log S against log k over the long-wavelength half."""
    # The upper half of the LADDER intersected with what is reportable, not
    # the upper half of what is reportable: the second moves with `grid`.
    half = idx_ok[idx_ok >= lam_mid.size // 2]
    out = np.full(S.shape[0], np.nan)
    if len(half) < 4:
        return out
    lk = np.log(2.0 * np.pi / lam_mid[half])
    for j in range(S.shape[0]):
        y = S[j, half]
        m = np.isfinite(y) & (y > 1e-6)
        if int(m.sum()) >= 4:
            out[j] = np.polyfit(lk[m], np.log(y[m]), 1)[0]
    return out


def _patch_scale(S, S_long, sd_long, idx_ok, lam_mid):
    """Wavelength at which S has climbed half way to its long-lambda plateau.

    A half-crossing rather than the argmax of S. Most real populations have a
    spectrum that rises and then flattens rather than peaking, so the argmax is
    whichever shell noise put on top -- often the LONGEST reportable shell,
    which reports the size of the section rather than a scale of anything in
    it. The half-crossing finds the knee, and only for populations whose
    plateau clears the null band by 3 sd; everything else gets NaN and no
    patch scale.
    """
    n_types = S.shape[0]
    lam_half = np.full(n_types, np.nan)
    band_bin = np.full(n_types, int(idx_ok[-1]), dtype=int)
    for j in range(n_types):
        if not S_long[j] > 1.0 + 3.0 * sd_long:
            continue
        target = 1.0 + 0.5 * (S_long[j] - 1.0)
        y = S[j, idx_ok]
        above = np.flatnonzero(np.isfinite(y) & (y >= target))
        if above.size:
            b = int(idx_ok[above[0]])
            lam_half[j] = lam_mid[b]
            band_bin[j] = b
    return lam_half, band_bin


def _focus(labels, tids, counts, wanted):
    if wanted:
        for j, c in enumerate(tids):
            if labels[c] == wanted:
                return j
    return int(np.argmax(counts))


def _band_map(ds, secs, codes, code, grid, lam_lo, lam_hi):
    """Per-cell composition excess of one population, band-passed to one shell."""
    try:
        import scipy.fft as sfft
        irfft2 = lambda a: sfft.irfft2(a, workers=_workers())
    except ImportError:
        irfft2 = np.fft.irfft2

    values = np.full(ds.n_cells, np.nan, dtype=np.float64)
    for s in secs:
        flat = s["flat"]
        sel = codes[s["rows"]] == code
        n_here = int(sel.sum())
        N = int(flat.size)
        if n_here == 0 or n_here == N:
            continue
        allf = np.bincount(flat, minlength=grid * grid).reshape(grid, grid).astype(np.float32)
        cf = np.bincount(flat[sel], minlength=grid * grid).reshape(grid, grid).astype(np.float32)
        Tk = _rfft2(cf) - (n_here / N) * _rfft2(allf)
        kmag, _ = _kgrid(grid, s["h"])
        with np.errstate(divide="ignore"):
            lam = np.where(kmag > 0, 2.0 * np.pi / np.maximum(kmag, 1e-30), np.inf)
        g = irfft2(np.where((lam >= lam_lo) & (lam < lam_hi), Tk, 0))
        # Counts-per-bin excess -> excess FRACTION, so the map is comparable
        # between sections of different density. The occupied-bin census is the
        # denominator because the empty half of a bounding box is not tissue.
        coarse = max(8, grid // 8)
        cg = (np.minimum((flat // grid) * coarse // grid, coarse - 1) * coarse
              + np.minimum((flat % grid) * coarse // grid, coarse - 1))
        occupied = int((np.bincount(cg, minlength=coarse * coarse) > 0).sum())
        rho_bin = N / max(occupied * (grid // coarse) ** 2, 1)
        values[s["rows"]] = g.ravel()[flat] / rho_bin
    return _quantize(values)


def _quantize(v):
    """Round the painted map to five significant digits of its own range.

    The host serialises this array with json.dumps, and a full-precision float64
    repr costs 20-odd characters per cell for a number the viewer turns into a
    colour. Five significant digits is finer than any colour ramp resolves and
    costs eight characters.
    """
    m = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.0
    if not m > 0:
        return v
    return np.round(v, int(max(0, min(15, 4 - np.floor(np.log10(m))))))


# ===========================================================================
# Real space: the support-restricted variance ladder
# ===========================================================================
def _sat(counts):
    """Summed-area table, S[y, x] = sum of counts[:y, :x]."""
    out = np.zeros((counts.shape[0] + 1, counts.shape[1] + 1), dtype=counts.dtype)
    np.cumsum(np.cumsum(counts, axis=0, dtype=counts.dtype), axis=1,
              dtype=counts.dtype, out=out[1:, 1:])
    return out


def _box_corner(S, m, s):
    """Counts in every m x m window whose top-left pixel is on a stride-s lattice."""
    H, W = S.shape[0] - 1, S.shape[1] - 1
    return (S[m:H + 1:s, m:W + 1:s] - S[0:H - m + 1:s, m:W + 1:s]
            - S[m:H + 1:s, 0:W - m + 1:s] + S[0:H - m + 1:s, 0:W - m + 1:s])


def _box_centred(S, b):
    """Counts in the b x b box centred on every pixel, edges clamped."""
    H, W = S.shape[0] - 1, S.shape[1] - 1
    r = b // 2
    ys, ye = np.clip(np.arange(H) - r, 0, H), np.clip(np.arange(H) + r + 1, 0, H)
    xs, xe = np.clip(np.arange(W) - r, 0, W), np.clip(np.arange(W) + r + 1, 0, W)
    return (S[np.ix_(ye, xe)] - S[np.ix_(ys, xe)]
            - S[np.ix_(ye, xs)] + S[np.ix_(ys, xs)])


def _ols(x, y):
    """slope, r^2, stderr(slope) of a straight-line fit."""
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + slope * (x - xm))
    sst = float(((y - ym) ** 2).sum())
    r2 = float(1.0 - float((resid ** 2).sum()) / sst) if sst > 0 else np.nan
    se = float(np.sqrt(float((resid ** 2).sum()) / (n - 2) / sxx)) if n > 2 else np.nan
    return slope, r2, se


def _real_space(ds, xy, sect, codes, usable, tids):
    """Var[N_c(R)] ~ R^alpha over windows lying wholly inside the tissue.

    Returns None with a reason where the tissue admits no ladder. It is a check
    on the spectrum, so it declines rather than aborting the run.
    """
    med_nn = _median_nn(ds)
    short_side = np.inf
    for s in np.unique(sect[usable]):
        P = xy[usable & (sect == s)]
        if P.shape[0] >= 2:
            short_side = min(short_side, float(min(np.ptp(P[:, 0]), np.ptp(P[:, 1]))))
    if not np.isfinite(short_side) or short_side <= 0:
        return {"reason": "the cells have no spatial extent to place windows in"}

    r_min = R_MIN_NN * med_nn
    r_cap = R_MAX_FRAC * short_side
    if r_cap < 2.0 * r_min:
        return {"reason": f"windows would run {r_min:.0f}-{r_cap:.0f} units, "
                          "under one octave of scale"}

    # The pixel is set by the cell spacing alone, never by the window range: the
    # support mask is built at this resolution, so letting the smallest window
    # move the pixel would silently move the closing scale, hence the support,
    # hence the largest window that fits.
    pix = max(med_nn / 2.0, 1e-9)
    grids = []
    for s in np.unique(sect[usable]):
        rows = np.flatnonzero(usable & (sect == s))
        if rows.size < MIN_USABLE:
            continue
        P = xy[rows]
        x0, y0 = float(P[:, 0].min()), float(P[:, 1].min())
        x1, y1 = float(P[:, 0].max()), float(P[:, 1].max())
        p_ = max(pix, max(x1 - x0, y1 - y0) / GRID_MAX)
        W = int(np.floor((x1 - x0) / p_)) + 1
        H = int(np.floor((y1 - y0) / p_)) + 1
        if W < 4 or H < 4:
            continue
        flat = (np.minimum(((P[:, 1] - y0) / p_).astype(np.int64), H - 1) * W
                + np.minimum(((P[:, 0] - x0) / p_).astype(np.int64), W - 1))
        occ = np.bincount(flat, minlength=H * W).reshape(H, W).astype(np.int32)
        sat_all = _sat(occ)
        # ceil, not round: the closing box is odd-sized, and one pixel less of
        # it shrinks the support enough to cut the largest admissible window
        # sharply.
        b = int(max(3, np.ceil(CLOSE_NN * med_nn / p_)))
        b += (b + 1) % 2
        dil = _box_centred(sat_all, b) >= 1
        support = _box_centred(_sat(dil.astype(np.int32)), b) == b * b
        grids.append({"p": p_, "H": H, "W": W, "flat": flat,
                      "codes": codes[rows], "sat_all": sat_all,
                      "sat_sup": _sat(support.astype(np.int32)),
                      "N": int(rows.size),
                      "area": float(support.sum()) * p_ * p_,
                      "bbox": max((x1 - x0) * (y1 - y0), 1e-9)})
    if not grids:
        return {"reason": "no section has enough positioned cells to place "
                          "windows on"}

    def n_disjoint(R):
        n = 0
        for g in grids:
            m = int(round(R / g["p"]))
            if 2 <= m <= min(g["H"], g["W"]):
                n += int((_box_corner(g["sat_sup"], m, m) == m * m).sum())
        return n

    lo, hi = r_min, r_cap
    if n_disjoint(hi) >= MIN_INDEP:
        r_max = hi
    elif n_disjoint(lo) < MIN_INDEP:
        return {"reason": f"not even a {r_min:.0f}-unit window fits wholly "
                          f"inside the tissue at {MIN_INDEP} non-overlapping "
                          "positions"}
    else:
        for _ in range(24):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if n_disjoint(mid) >= MIN_INDEP else (lo, mid)
        r_max = lo
    if r_max < 1.6 * r_min:
        return {"reason": f"the largest window fitting inside the tissue is "
                          f"{r_max:.0f} units against a smallest of {r_min:.0f}, "
                          "under an octave"}

    # Lay the ladder out in pixels and deduplicate, so two requested sizes that
    # round to the same window are not fitted as two independent points.
    p_ref = max(g["p"] for g in grids)
    m_lo = max(MIN_PIXELS, int(round(r_min / p_ref)))
    m_hi = max(m_lo + 1, int(round(r_max / p_ref)))
    m_lad = np.unique(np.round(np.geomspace(m_lo, m_hi, N_SCALES)).astype(int))
    if m_lad.size < MIN_SCALES:
        return {"reason": f"{r_min:.0f}-{r_max:.0f} units is only {m_lad.size} "
                          "distinct window size(s) at this resolution"}
    r_want = m_lad * p_ref
    n_s = r_want.size

    sum_w = np.zeros(n_s)
    obs_e = np.zeros((tids.size, n_s))
    obs_v = np.zeros((tids.size, n_s))
    nul_v = np.zeros((tids.size, n_s))
    r_actual = np.full(n_s, np.nan)
    support_area = 0.0
    bbox_area = 0.0

    for g in grids:
        support_area += g["area"]
        bbox_area += g["bbox"]
        N_tot = g["N"]
        shells = []
        for j, Rj in enumerate(r_want):
            m = int(round(Rj / g["p"]))
            if m < 2 or m > min(g["H"], g["W"]):
                shells.append(None)
                continue
            st = max(1, m // 2)
            adm = _box_corner(g["sat_sup"], m, st) == m * m
            n_ok = int(adm.sum())
            if n_ok < MIN_WINDOWS:
                shells.append(None)
                continue
            tot = _box_corner(g["sat_all"], m, st)[adm].astype(np.float64)
            shells.append({"m": m, "s": st, "adm": adm, "n": n_ok,
                           "e": float(tot.mean()), "v": float(tot.var()),
                           "h": float((tot * (N_tot - tot) / max(N_tot - 1, 1)).mean())})
            R_here = m * g["p"]
            r_actual[j] = R_here if not np.isfinite(r_actual[j]) else \
                0.5 * (r_actual[j] + R_here)
            sum_w[j] += n_ok
        for ci, code in enumerate(tids):
            sel = g["codes"] == code
            n_c = int(sel.sum())
            if n_c == 0:
                continue
            sat_c = _sat(np.bincount(g["flat"][sel], minlength=g["H"] * g["W"])
                         .reshape(g["H"], g["W"]).astype(np.int32))
            p = n_c / N_tot
            for j, sh in enumerate(shells):
                if sh is None:
                    continue
                v = _box_corner(sat_c, sh["m"], sh["s"])[sh["adm"]].astype(np.float64)
                obs_e[ci, j] += sh["n"] * float(v.mean())
                obs_v[ci, j] += sh["n"] * float(v.var())
                # Hypergeometric given the window's total count: the tissue's
                # own crowding is scale-dependent, so alpha_null is not 2.
                nul_v[ci, j] += sh["n"] * (p * (1.0 - p) * sh["h"] + p * p * sh["v"])

    good = sum_w > 0
    if int(good.sum()) < MIN_SCALES:
        return {"reason": f"only {int(good.sum())} window size(s) had "
                          f"{MIN_WINDOWS} windows entirely inside the tissue"}
    R = r_actual[good]
    logR = np.log10(R)
    v_obs = obs_v[:, good] / sum_w[good]
    v_nul = nul_v[:, good] / sum_w[good]

    alpha = np.full(tids.size, np.nan)
    alpha_null = np.full(tids.size, np.nan)
    excess = np.full(tids.size, np.nan)
    r2 = np.full(tids.size, np.nan)
    for ci in range(tids.size):
        ok = np.isfinite(v_obs[ci]) & (v_obs[ci] > 0)
        if int(ok.sum()) < MIN_SCALES:
            continue
        alpha[ci], r2[ci], _se = _ols(logR[ok], np.log10(v_obs[ci][ok]))
        okn = ok & (v_nul[ci] > 0)
        if int(okn.sum()) >= 2:
            alpha_null[ci] = _ols(logR[okn], np.log10(v_nul[ci][okn]))[0]
        # The RATIO, not the excess variance: wherever Var_obs / Var_null is
        # flat in R -- most unstructured populations -- the difference is
        # proportional to the null and its own exponent collapses onto a smooth
        # function of abundance.
        excess[ci] = float(np.exp(np.mean(np.log(np.maximum(
            v_obs[ci][ok] / np.maximum(v_nul[ci][ok], 1e-12), 1e-12)))))
    if not np.isfinite(alpha).any():
        return {"reason": "no population produced a non-zero window variance at "
                          f"{MIN_SCALES} or more window sizes"}
    return {"alpha": alpha, "alpha_null": alpha_null, "excess": excess,
            "r2": r2, "r_lo": float(R[0]), "r_hi": float(R[-1]),
            "n_sizes": int(good.sum()),
            "windows": (float(sum_w[good].min()), float(sum_w[good].max())),
            "support_frac": support_area / bbox_area, "reason": None}


def _median_nn(ds):
    """Median nearest-neighbour spacing, from a strided sample per section.

    A median needs a sample, not a census: a deterministic stride of at most
    NN_SAMPLE points gives the median of every point at a fraction of the
    cost.
    The tree is over every positioned cell rather than the selected ones on
    purpose -- the spacing being estimated is the tissue's, and a lasso does
    not move the cells apart.
    """
    nn = []
    for part in ds.kdtrees():
        pts = part["tree"].data
        step = max(1, pts.shape[0] // NN_SAMPLE)
        d, _ = part["tree"].query(pts[::step], k=2)
        nn.append(np.atleast_2d(d)[:, 1])
    nn = np.concatenate(nn)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 1.0


# ===========================================================================
# Reporting
# ===========================================================================
def _report(ds, labels, tids, counts, lam_mid, ok_shell, S, sd, S_short, S_long,
            lam_half, alpha, rs, focus_j, focus_flat, edges, fb, values,
            n_usable, grid, n_perms, field, relaxed, min_cells, short_bins,
            sd_short, modes, have_short):
    names = [labels[c] for c in tids]
    x_lam = lam_mid[ok_shell]
    has_rs = rs is not None and rs.get("reason") is None

    columns = ["Population", "Cells"]
    if have_short:
        columns.append("S short")
    columns += ["Patch scale", "S long", "Spectral exponent"]
    if has_rs:
        columns += ["Var / Var_null", "alpha"]
    rows = []
    for j in range(tids.size):
        row = [names[j], int(counts[j])]
        if have_short:
            row.append(_num(S_short[j], 3))
        row += [_num(lam_half[j], 0), _num(S_long[j], 3), _num(alpha[j], 2)]
        if has_rs:
            row += [_num(rs["excess"][j], 2), _num(rs["alpha"][j], 2)]
        rows.append(row)
    # Ascending, and no sort_by: the viewer opens a named column DESCENDING,
    # and the end of S(short) worth seeing first is the low one -- a population
    # laid out more evenly than chance is the thing no shipped tool can see.
    key = S_short if have_short else -np.where(np.isfinite(S_long), S_long, -np.inf)
    order = sorted(range(tids.size),
                   key=lambda j: (not np.isfinite(key[j]),
                                  key[j] if np.isfinite(key[j]) else 0.0,
                                  names[j]))
    rows = [rows[j] for j in order]

    lines = [
        {"label": "null 95% low", "x": x_lam,
         "y": np.maximum(1.0 - 2.0 * sd[ok_shell], 0.0),
         "color": _BAND_COLOR, "dashed": True},
        {"label": f"null 95% high ({n_perms} label permutations)", "x": x_lam,
         "y": 1.0 + 2.0 * sd[ok_shell], "color": _BAND_COLOR,
         "dashed": True, "band": True},
    ]
    colors = _colors(ds, field)
    # Spread the drawn curves over the whole range of the statistic: the most
    # extreme populations are often near-duplicate subsets of one lineage, and
    # a plot of only those says nothing.
    rank = [j for j in np.argsort(-np.where(np.isfinite(S_long), S_long, -1.0))
            if int(j) != focus_j]
    if len(rank) > MAX_LINES - 1:
        take = np.unique(np.linspace(0, len(rank) - 1, MAX_LINES - 1).round().astype(int))
        rank = [rank[int(i)] for i in take]
    for j in [focus_j] + [int(v) for v in rank]:
        lines.append({"label": names[j], "x": x_lam, "y": S[j][ok_shell],
                      "color": colors.get(names[j])})

    return {
        "summary": _summary(names, counts, S_short, S_long, lam_half, alpha, rs,
                            focus_j, focus_flat, edges, fb, x_lam,
                            n_usable, grid, n_perms, field, relaxed, min_cells,
                            short_bins, sd_short, has_rs, ok_shell, have_short),
        "cells": {
            "values": values,
            "label": f"{names[focus_j]} excess @ {edges[fb]:.0f}-{edges[fb + 1]:.0f} u",
            "diverging": True, "units": "fraction of cells",
        },
        "series": {"title": "Structure factor S(lambda)",
                   "x_label": "wavelength lambda = 2 pi / k (dataset units)",
                   "y_label": "S", "lines": lines, "rules": [1.0], "log_x": True},
        "table": {"title": "Scale spectrum by population",
                  "columns": columns, "rows": rows},
    }


def _summary(names, counts, S_short, S_long, lam_half, alpha, rs, focus_j,
             focus_flat, edges, fb, x_lam, n_usable, grid, n_perms,
             field, relaxed, min_cells, short_bins, sd_short, has_rs,
             ok_shell, have_short):
    n = len(names)
    lam_s0, lam_s1 = edges[short_bins[0]], edges[short_bins[-1] + 1]
    # Below the null band, not merely below 1: on real tissue many populations
    # sit under 1 at the shortest shells by sampling alone.
    even = ([int(j) for j in np.argsort(S_short)
             if np.isfinite(S_short[j]) and S_short[j] < 1.0 - 3.0 * sd_short]
            if have_short else [])
    lines = [
        f"**{n} population(s)**, {n_usable:,} labelled cells, wavelengths "
        f"{x_lam.min():.0f}-{x_lam.max():.0f} units on a {grid}x{grid} grid.",
        "",
    ]
    if not have_short:
        lines.append(
            f"**No S short at this grid** — the packing band is "
            f"{lam_s0:.0f}-{lam_s1:.0f} units and a {grid}x{grid} grid resolves "
            f"nothing below {x_lam.min():.0f}. The column is dropped rather "
            f"than refilled with a longer band; raise the binning resolution "
            f"to read it.")
    elif even:
        top = ", ".join(f"{names[j]} {S_short[j]:.2f}" for j in even[:4])
        lines.append(
            f"**Laid out more evenly than chance at {lam_s0:.0f}-{lam_s1:.0f} "
            f"units** — {len(even)} of {n} below the null band (S short < "
            f"{1.0 - 3.0 * sd_short:.2f}): {top}.")
    else:
        j0 = int(np.nanargmin(S_short))
        lines.append(
            f"**Nothing is sub-random at {lam_s0:.0f}-{lam_s1:.0f} units** — the "
            f"lowest S short is {names[j0]} at {S_short[j0]:.2f}, against a null "
            f"band of +/-{3.0 * sd_short:.2f}.")
    scaled = [j for j in range(n) if np.isfinite(lam_half[j])]
    if scaled:
        scaled.sort(key=lambda j: lam_half[j])
        lines.append(
            f"**Patch scale** resolved for {len(scaled)} of {n}: from "
            f"{names[scaled[0]]} at {lam_half[scaled[0]]:.0f} units to "
            f"{names[scaled[-1]]} at {lam_half[scaled[-1]]:.0f}.")
    else:
        lines.append(f"**No population has a resolvable patch scale** — none "
                     f"clears the null band by 3 sd at long wavelength.")
    j_long = int(np.nanargmax(np.where(np.isfinite(S_long), S_long, -np.inf)))
    lines.append(
        f"Context — S long spans {np.nanmin(S_long):.2f} to "
        f"{np.nanmax(S_long):.1f} (highest {names[j_long]}); it tracks the "
        "self-neighbour fraction an adjacency tool already reports.")
    if has_rs:
        rho = _spearman(np.log(np.maximum(S_long, 1e-6)),
                        np.log(np.maximum(rs["excess"], 1e-6)))
        lines.append(
            f"Real-space check over {rs['n_sizes']} window sizes "
            f"{rs['r_lo']:.0f}-{rs['r_hi']:.0f} units (support "
            f"{100 * rs['support_frac']:.0f}% of the bounding box): Var/Var_null "
            f"ranks with S long at Spearman "
            + ("n/a" if not np.isfinite(rho) else f"{rho:+.2f}")
            + f", alpha spans {np.nanmin(rs['alpha']):.2f}-{np.nanmax(rs['alpha']):.2f} "
              "against 2 for the area law.")
    elif rs is not None:
        lines.append(f"Real-space check unavailable: {rs['reason']}.")
    fl = names[focus_j]
    lines += ["", f"Map: **{fl}** composition excess band-passed to "
                  f"{edges[fb]:.0f}-{edges[fb + 1]:.0f} units"
                  + (", the middle of the range — its S never leaves the null "
                     "band, so the map is noise." if focus_flat
                     else f", its patch scale (S long {S_long[focus_j]:.2f})."),
              f"S cannot fall much below p/(1-p) = "
              f"{counts[focus_j] / max(n_usable - counts[focus_j], 1):.3f} for "
              f"{fl}: at that abundance the rest of the field is its own noise."]
    if relaxed:
        lines.append(f"No population reached {min_cells} cells, so only the "
                     f"largest ({int(counts[0]):,}) was scored.")
    thin = int(np.sum(ok_shell) )
    if thin < 8:
        lines.append(f"Only {thin} wavelength shell(s) hold {MIN_MODES} Fourier "
                     "modes; the exponent is fitted over very little range.")
    if field:
        lines.append(f"Grouping column `{field}`.")
    return "\n".join(lines)


def _colors(ds, field):
    for entry in ds.categorical_fields():
        if field is None or entry["name"] == field or entry["label"] == field:
            return entry.get("colors") or {}
    return {}


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 4:
        return np.nan
    ra, rb = _ranks(a[m]), _ranks(b[m])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def _ranks(v):
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size, dtype=np.float64)
    r[order] = np.arange(v.size, dtype=np.float64)
    # Average ties, so two populations with identical S do not get an arbitrary
    # order that would move the correlation between runs on different platforms.
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r, minlength=uniq.size)
    return (sums / cnt)[inv]


def _num(v, nd):
    return None if not np.isfinite(v) else round(float(v), nd)


def _seed(key):
    """A seed that is a function of the parameters and of nothing else.

    SHA-1 rather than hash(): PYTHONHASHSEED randomises string hashing per
    process, so hash() would give a different null band on every run, for a
    reason nobody would find.
    """
    return int.from_bytes(
        hashlib.sha1("|".join(str(k) for k in key).encode("utf-8")).digest()[:8],
        "big")
