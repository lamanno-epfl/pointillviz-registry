"""Spatial vector field — the pure field, gradient and significance numerics.

Everything the Vector Field tool needs that is NOT the plugin contract: grid
geometry, binning, the smoothed scalar field, its spatial gradient, the
significance statistic and the arrow lattice. Every function here is pure and
takes plain arrays, so the numerical behaviour can be driven directly from a
test without a dataset, a manifest or a subprocess — which is what
`tests/vector_field_numerics_test.py` does.

SCIENTIFIC NOTES

 * THE FIELD IS A LOCAL MEAN, NEVER A DENSITY. Numerator and denominator are
   convolved with the SAME separable Gaussian and divided (Knutsson normalized
   convolution). A region with twice the cells and the same expression yields
   the same value.

 * THE GRADIENT DIFFERENTIATES THE ACCUMULATORS, NOT THE FINISHED FIELD.
   d/dx of A/C is NOT (dA/dx)/C. The quotient rule

       dF/dx = (A_x - F*C_x) / (C * cw)

   is used instead, with A_x obtained from a derivative-of-Gaussian kernel.
   Two properties follow, and both are the reason this module does not simply
   central-difference a finished field array:

     1. BOUNDARY IMMUNITY. A zero-filled field array holds a finite 0.0 outside
        the support mask that is indistinguishable from genuine zero
        expression, and any stencil crossing that edge reads real-against-zero
        and manufactures a maximum-magnitude arrow pointing out of the tissue.
        The quotient rule is an identity on the RESIDUAL (alpha_i - F), whose
        local mean is zero by construction, so a tissue edge with uniform
        expression yields 0 no matter how sharply cell density falls off. A
        density edge is not an expression gradient.

     2. DENSITY IMMUNITY. Differencing the smoothed mean directly carries a
        -F * dln(C)/dx term wherever cellularity varies, which is a large part
        of any real section. The quotient rule cancels it exactly.

 * THE ARROWS ARE GATED. Gaussian smoothing manufactures a beautiful, entirely
   convincing vector field out of pure noise: a permuted gene (all spatial
   structure destroyed) still produces arrows whose neighbours agree closely,
   and a gene present in 0.1% of cells produces a MORE directionally coherent
   field than a real, broadly expressed one — it is a few hundred point masses
   convolved with a kernel, and it looks exactly like biology. Every global
   "does this look coherent" summary fails to discriminate. So a site is
   reported only if it clears a Rayleigh significance gate (z >= 3 with at
   least 30 effective cells under the kernel). Under the null that admits
   1.11% of sites; measured, real signal survives at an order of magnitude
   more.

PRECISION. Every grid array here is float64. The browser implementation this
was ported from used Float32Array, because it fed the same buffers to the GPU;
nothing here does, and a 512x512 grid is two megabytes per array either way.
The global-mean centring in `accumulate` is kept regardless — it is an exact
identity on the gradient, so it can only help, and it is what keeps the field
honest on embedding channels that reach 6.2e7.
"""

from __future__ import annotations

import numpy as np

# Kernel radius cap. A derivative-of-Gaussian is MORE truncation-sensitive than
# a Gaussian because its mass sits near +/-sigma, so this is generous.
#
# 9 is not arbitrary: it is 3 x MAX_SIGMA, so the widest smoothing this module
# offers still gets a full three-sigma kernel. Past that point the operator
# stops being a Gaussian pair and drifts — measured lattice curl, normalized,
# as the truncation tightens:
#
#     sigma 2.00  r=8 (3.00 sigma)   max 3.5e-2
#     sigma 3.00  r=8 (2.67 sigma)   max 4.7e-2
#     sigma 6.00  r=8 (1.33 sigma)   max 4.2e-1   <- 12x worse
#
# MAX_SIGMA and MAX_RADIUS must be changed together, and the manifest's sigma
# ceiling is MAX_SIGMA for exactly this reason.
MAX_RADIUS = 9
MAX_SIGMA = 3.0

# Minimum fraction of the smoothing kernel that must sit on supported grid
# cells before a gradient is reported.
#
# CALIBRATED, NOT CHOSEN. Measured on an exact linear ramp, sweeping the
# threshold and sigma over 0.4..3, reporting the bias in the recovered slope
# and how much of the supported grid survives:
#
#     threshold   worst    mean     kept
#       0.0       52.0%    3.77%    90.3%     <- no window-truncation gate
#       0.9       22.8%    1.28%    79.5%
#       0.99       5.8%    0.17%    68.3%     <- the knee
#       0.999      0.0%    0.00%    60.8%
#
# 0.99 is the knee: it removes essentially all of the bias while keeping eight
# points more of the grid than the exact-erosion setting. Without any gate the
# outermost arrows understate the true slope by up to 52%, silently and only at
# the tissue edge.
MIN_FILL = 0.99

# Support threshold: the smoothed cell count under the kernel below which a
# grid site is not part of the tissue at all.
DEFAULT_SUPPORT = 0.5


def radius_for(sigma: float) -> int:
    """Kernel half-width for a sigma, in grid cells. Always in [1, MAX_RADIUS]."""
    return int(max(1, min(MAX_RADIUS, np.ceil(3.0 * float(sigma)))))


# ===========================================================================
# Kernels. All three are float64 arrays of length 2r+1, indexed by t+r.
# ===========================================================================
def gauss_kernel(sigma: float, r: int) -> np.ndarray:
    """Normalized 1D Gaussian taps, sum(k) == 1."""
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(t * t) / (2.0 * sigma * sigma))
    return k / k.sum()


def dog_kernel(sigma: float, r: int) -> np.ndarray:
    """Derivative-of-Gaussian taps for a CORRELATION, with sum(k)==0, sum(t*k)==1.

    The sign convention matters and is the single easiest thing to get exactly
    backwards. With the correlation `out[i] = sum_t src[i+t] * k[t+r]`, a
    POSITIVE response to an increasing ramp requires taps that are positive for
    t > 0. Sum-normalizing instead (dividing by sum(k), which is 0) or using
    the convolution convention (t -> -t) both yield a field whose every arrow
    points exactly backwards, and no magnitude, curl or boundary test detects
    it.

    Renormalizing by the first moment rather than by an analytic constant is
    what makes the operator EXACT on a linear ramp at every sigma and every
    truncation radius, rather than merely asymptotically right.
    """
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = t * np.exp(-(t * t) / (2.0 * sigma * sigma))
    # m1 > 0 always (every term t*t*exp(..) >= 0, and t=+/-1 contribute), so
    # this is never a division by zero for any r >= 1.
    return k / float((t * k).sum())


def sq_kernel(kg: np.ndarray) -> np.ndarray:
    """Squared Gaussian taps for the effective-sample-size pass.

    NOT renormalized — sum(k^2) is exactly the quantity N_eff needs.
    """
    return kg * kg


# ===========================================================================
# Separable correlation with ZERO padding.
#
# Zero padding, not replication. Absent grid cells genuinely hold no
# observations; replicating the edge value fabricates cells outside the bbox,
# inflates the smoothed count at the border and flattens the normal derivative
# there. Mass must LEAVE at the border, never be created.
#
# The two axes are separate functions rather than one helper applied twice
# because the quotient rule needs to mix kernels PER AXIS (a Gaussian on one
# axis and a derivative on the other). A single-kernel helper applied to both
# axes with a derivative kernel yields d2/dxdy, which is identically zero on a
# ramp — a silent, total failure.
# ===========================================================================
def _correlate(src: np.ndarray, k: np.ndarray, r: int, axis: int) -> np.ndarray:
    n = src.shape[axis]
    out = np.zeros_like(src, dtype=np.float64)
    for t in range(-r, r + 1):
        w = k[t + r]
        if w == 0.0:
            continue
        if t >= 0:
            if t >= n:
                continue
            dst_sl, src_sl = slice(0, n - t), slice(t, n)
        else:
            if -t >= n:
                continue
            dst_sl, src_sl = slice(-t, n), slice(0, n + t)
        idx_dst = [slice(None)] * src.ndim
        idx_src = [slice(None)] * src.ndim
        idx_dst[axis] = dst_sl
        idx_src[axis] = src_sl
        out[tuple(idx_dst)] += src[tuple(idx_src)] * w
    return out


def horiz(src: np.ndarray, k: np.ndarray, r: int) -> np.ndarray:
    """Correlate along x. `src` is (GY, GX)."""
    return _correlate(np.asarray(src, dtype=np.float64), k, r, axis=1)


def vert(src: np.ndarray, k: np.ndarray, r: int) -> np.ndarray:
    """Correlate along y. `src` is (GY, GX)."""
    return _correlate(np.asarray(src, dtype=np.float64), k, r, axis=0)


# ===========================================================================
# Grid geometry.
#
# Square cells by construction: the extent is derived from GX*cell_size, not
# from the bbox, so cw == ch exactly. When max_g binds on either axis the cell
# size is ENLARGED (never the axis clamped independently), which is what keeps
# the cells square on a section of any aspect ratio. Independently clamped
# axes give cw != ch, and that rotates every arrow.
# ===========================================================================
class Grid:
    __slots__ = ("GX", "GY", "NC", "cw", "ch", "min_x", "min_y",
                 "max_x", "max_y", "cell_size", "clamped")

    def __init__(self, GX, GY, cell, min_x, min_y, clamped):
        self.GX, self.GY, self.NC = int(GX), int(GY), int(GX) * int(GY)
        self.cw = self.ch = self.cell_size = float(cell)
        self.min_x, self.min_y = float(min_x), float(min_y)
        self.max_x = self.min_x + self.GX * self.cell_size
        self.max_y = self.min_y + self.GY * self.cell_size
        self.clamped = bool(clamped)

    def __repr__(self):
        return (f"Grid({self.GX}x{self.GY}, cell={self.cell_size:g}, "
                f"clamped={self.clamped})")


def make_grid(min_x, min_y, max_x, max_y, cell_size, max_g=512) -> Grid:
    max_g = max(2, int(max_g))
    w = max(float(max_x) - float(min_x), 1e-9)
    h = max(float(max_y) - float(min_y), 1e-9)
    a = max(float(cell_size), 1e-9)
    # Enlarge the cell until BOTH axes fit inside max_g. Solved directly rather
    # than iterated so a pathological aspect ratio cannot loop.
    need = max(w / max_g, h / max_g)
    if a < need:
        a = need
    GX = max(2, min(max_g, int(np.ceil(w / a))))
    GY = max(2, min(max_g, int(np.ceil(h / a))))
    return Grid(GX, GY, a, min_x, min_y, a > float(cell_size) * (1 + 1e-12))


def assign(x, y, g: Grid) -> np.ndarray:
    """(n,) int64 grid-cell index per point; -1 outside the grid or non-finite.

    np.floor, NOT a cast. A cast truncates TOWARD ZERO, so a point just left of
    min_x gives a quotient in (-1, 0], truncates to 0 and passes a `gx < 0`
    test — silently binned into the first column instead of being rejected.
    Same trap on the low y edge. Reachable the moment the grid is built from
    anything narrower than the full data bbox, and it folds out-of-view cells
    onto exactly the border column where boundary artefacts are read.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    gx = np.full(x.shape, -1, dtype=np.int64)
    gy = np.full(y.shape, -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        gx[ok] = np.floor((x[ok] - g.min_x) / g.cw).astype(np.int64)
        gy[ok] = np.floor((y[ok] - g.min_y) / g.ch).astype(np.int64)
    inside = ok & (gx >= 0) & (gy >= 0) & (gx < g.GX) & (gy < g.GY)
    out = np.full(x.shape, -1, dtype=np.int64)
    out[inside] = gy[inside] * g.GX + gx[inside]
    return out


class Accum:
    """Per-grid-cell sums, raw and centred. Shaped (GY, GX)."""
    __slots__ = ("a", "c", "a2", "ac", "a2c", "m0", "used")

    def __init__(self, a, c, a2, ac, a2c, m0, used):
        self.a, self.c, self.a2 = a, c, a2
        self.ac, self.a2c = ac, a2c
        self.m0, self.used = float(m0), int(used)


def accumulate(cell_of, alpha, g: Grid, valid=None, eligible=None) -> Accum:
    """Accumulate sum(alpha), count and sum(alpha^2) per grid cell.

    A cell contributes only if it is ELIGIBLE, VALID for this feature, and
    carries a finite value. Note the asymmetry that makes this honest: a zero
    is a real observation and enters BOTH arrays, while a non-finite value is a
    missing measurement and enters NEITHER. Collapsing missing data to a
    measured zero would fabricate a composition step wherever missingness is
    spatially clustered — which, in a gene panel, it always is.

    The global mean is removed HERE, once, into dedicated buffers. With
    a' = a - m0*c the smoothed quantities become A' = A - m0*C and F' = F - m0,
    so

        A'_x - F'*C_x  =  (A_x - m0*C_x) - (F - m0)*C_x  =  A_x - F*C_x

    identically: the reported gradient is unchanged, but the large common
    offset is gone before any convolution runs. It lives here rather than in
    `build_field` because it MUTATES nothing the caller owns and because
    build_field is called repeatedly over one accumulation — centring there
    would re-centre already-centred arrays on the second call, compute m0 ~ 0,
    and return every field VALUE short by the feature's global mean while the
    gradients still looked perfect.

    The second moment is re-centred by the same identity,
    a2' = a2 - 2*m0*a + m0^2*c, so the local residual variance stays a variance
    about the new origin.
    """
    cell_of = np.asarray(cell_of, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float64)
    keep = (cell_of >= 0) & np.isfinite(alpha)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    if eligible is not None:
        keep &= np.asarray(eligible, dtype=bool)
    k = cell_of[keep]
    v = alpha[keep]
    NC = g.NC
    a = np.bincount(k, weights=v, minlength=NC).astype(np.float64)
    c = np.bincount(k, minlength=NC).astype(np.float64)
    a2 = np.bincount(k, weights=v * v, minlength=NC).astype(np.float64)
    shape = (g.GY, g.GX)
    a = a.reshape(shape)
    c = c.reshape(shape)
    a2 = a2.reshape(shape)
    tot_c = c.sum()
    m0 = float(a.sum() / tot_c) if tot_c > 0 else 0.0
    ac = a - m0 * c
    a2c = a2 - 2.0 * m0 * a + m0 * m0 * c
    return Accum(a, c, a2, ac, a2c, m0, int(keep.sum()))


class Field:
    """The smoothed field, its gradient, the gate, and the counters.

    Every array is (GY, GX) float64 except `mask` and `valid`, which are bool.

      F      the smoothed local mean, in the CALLER's units (m0 added back)
      gx,gy  the spatial gradient, per data unit
      mag    hypot(gx, gy)
      neff   effective cells under the kernel
      z      mag / SE, Rayleigh-distributed under the null
      fill   fraction of the kernel sitting on supported cells
      mask   the support test passed: this site is inside the tissue
      valid  mask AND interior AND fill AND the significance gate: an arrow
             may be drawn here

    n_supported / n_valid / n_gated / n_edge count `mask`, the MEASURED set
    (mask + four supported neighbours + fill), the population of `valid`, and
    supported sites rejected for being too close to an edge to measure.
    """
    __slots__ = ("F", "gx", "gy", "mag", "neff", "z", "fill", "mask", "valid",
                 "n_supported", "n_valid", "n_gated", "n_edge",
                 "radius", "sigma_p", "min_fill", "m0")

    def __init__(self, **kw):
        for key, val in kw.items():
            setattr(self, key, val)


def build_field(g: Grid, acc: Accum, sigma: float, *, support=DEFAULT_SUPPORT,
                min_neff=30.0, z_min=3.0, min_fill=MIN_FILL) -> Field:
    """The field, its gradient, and the significance statistic.

    Fourteen separable sub-passes, sharing horizontals wherever two outputs
    read the same intermediate.

    The `valid` mask is deliberately stricter than `mask`:
      - the four von-Neumann neighbours must also be supported, so no arrow is
        ever emitted from a stencil that straddles the tissue edge;
      - the outermost grid ring is excluded outright;
      - the smoothing window must be at least `min_fill` covered by supported
        cells, or the local mean is one-sided and its slope is biased low;
      - N_eff and z must clear their thresholds.
    """
    sigma = float(sigma)
    if not (sigma > 0):
        raise ValueError("sigma must be positive; a zero or negative sigma "
                         "makes exp(-t^2/0) NaN and every downstream array "
                         "with it.")
    r = radius_for(sigma)
    kg = gauss_kernel(sigma, r)
    kd = dog_kernel(sigma, r)
    kq = sq_kernel(kg)
    m0 = acc.m0

    # ---- horizontals ------------------------------------------------------
    Hga = horiz(acc.ac, kg, r)
    Hda = horiz(acc.ac, kd, r)
    Hgc = horiz(acc.c, kg, r)
    Hdc = horiz(acc.c, kd, r)

    # ---- verticals: numerator and denominator + their derivatives ---------
    A = vert(Hga, kg, r)          # A   = G(x)G(y) * a
    Ax = vert(Hda, kg, r)         # A_x = D(x)G(y) * a
    Ay = vert(Hga, kd, r)         # A_y = G(x)D(y) * a
    C = vert(Hgc, kg, r)          # C   = G(x)G(y) * c
    Cx = vert(Hdc, kg, r)
    Cy = vert(Hgc, kd, r)

    # ---- verticals: the significance pair ---------------------------------
    # C2 = (G^2 * c) gives N_eff = C^2 / C2, the effective number of cells
    # under the kernel. A2 gives the LOCAL residual scale, which is what makes
    # the gate discriminate; a single global tau is materially weaker.
    C2 = vert(horiz(acc.c, kq, r), kq, r)
    A2 = vert(horiz(acc.a2c, kg, r), kg, r)

    # ---- normalized convolution + support mask ----------------------------
    mask = C >= support
    F = np.zeros_like(C)
    np.divide(A, C, out=F, where=mask)
    F[mask] += m0

    # ---- fill fraction: how much of the kernel sits on real data ----------
    #
    # The quotient rule is exact only while the smoothing window lies entirely
    # inside the supported region. Where the window is truncated — at every
    # tissue edge — F becomes a ONE-SIDED local mean, its centroid shifts
    # inward, and the estimated slope is biased LOW. Smoothing the binary mask
    # with the SAME kernel measures exactly that truncation: fill == 1 means
    # the window is entirely on supported cells, which is precisely the
    # condition under which the operator is exact. A soft threshold rather
    # than a hard box erosion by `radius` because it degrades gracefully
    # around concave boundaries and costs the same two passes.
    fill = vert(horiz(mask.astype(np.float64), kg, r), kg, r)

    # ---- gradient, N_eff, z ----------------------------------------------
    # sigma_p is the smoothing length in DATA units, which is what the standard
    # error of a derivative is expressed per.
    sigma_px, sigma_py = sigma * g.cw, sigma * g.ch
    gx = np.zeros_like(C)
    gy = np.zeros_like(C)
    mag = np.zeros_like(C)
    neff = np.zeros_like(C)
    z = np.zeros_like(C)

    # Interior only: the outermost ring can never satisfy the neighbour rule
    # and is excluded outright.
    inner = np.zeros_like(mask)
    inner[1:-1, 1:-1] = True
    # Boundary rule: this cell AND its four stencil neighbours must be
    # supported. Without it roughly one arrow in ten is a full-scale vector
    # pointing out of the tissue, and they form a continuous ring around every
    # specimen.
    nbr = np.zeros_like(mask)
    nbr[1:-1, 1:-1] = (mask[1:-1, :-2] & mask[1:-1, 2:] &
                       mask[:-2, 1:-1] & mask[2:, 1:-1])
    edge_ok = inner & mask & nbr
    measured = edge_ok & (fill >= min_fill)
    # Not supported at all = outside the tissue; that is not a loss and must
    # not be reported as one. Only a SUPPORTED site rejected for being too near
    # an edge counts against the usable field.
    n_edge = int((inner & mask & ~measured).sum())

    if measured.any():
        c_m = C[measured]
        # The CENTRED field: F holds F' + m0 for the caller's benefit, but the
        # quotient rule is stated about the centred origin, and re-adding m0
        # here would reintroduce exactly the term the centring removed.
        f_m = F[measured] - m0
        ux = (Ax[measured] - f_m * Cx[measured]) / c_m / g.cw
        uy = (Ay[measured] - f_m * Cy[measured]) / c_m / g.ch
        m_m = np.hypot(ux, uy)
        gx[measured] = ux
        gy[measured] = uy
        mag[measured] = m_m

        c2_m = C2[measured]
        ne = np.where(c2_m > 0, (c_m * c_m) / np.where(c2_m > 0, c2_m, 1.0), 0.0)
        neff[measured] = ne
        # Local residual variance of (alpha - F), floored at zero: for a
        # perfectly uniform neighbourhood the two terms cancel to a small
        # negative number.
        var_local = np.maximum(A2[measured] / c_m - f_m * f_m, 0.0)
        tau = np.sqrt(var_local)
        # SE of a Gaussian-derivative estimate of the mean's slope. The sqrt(2)
        # is the two-sided derivative; sigma_p converts per-kernel to per-data
        # unit. Under H0 the magnitude of a 2-vector of independent N(0,SE)
        # components is Rayleigh(1) in z, so P(z > 3) = exp(-4.5) = 1.11% per
        # independent kernel.
        good = (ne > 0) & (tau > 0)
        se = np.full(ne.shape, np.inf)
        denom = 0.5 * (sigma_px + sigma_py) * np.sqrt(2.0 * np.where(good, ne, 1.0))
        se[good] = tau[good] / denom[good]
        zz = np.zeros(ne.shape)
        usable = good & np.isfinite(se) & (se > 0)
        zz[usable] = m_m[usable] / se[usable]
        z[measured] = zz

    valid = np.zeros_like(mask)
    valid[measured] = (neff[measured] >= min_neff) & (z[measured] >= z_min)

    return Field(
        F=F, gx=gx, gy=gy, mag=mag, neff=neff, z=z, fill=fill,
        mask=mask, valid=valid,
        n_supported=int(mask.sum()), n_valid=int(measured.sum()),
        n_gated=int(valid.sum()), n_edge=n_edge,
        radius=r, sigma_p=0.5 * (sigma_px + sigma_py), min_fill=float(min_fill),
        m0=m0,
    )


def percentile(field: Field, p: float) -> float:
    """Percentile of |grad| over the GATED sites. 0 when nothing is gated.

    The reference length scale the arrow lengths are normalized against. Taken
    over the gated sites only, because a percentile that included the sites
    nobody will see would set the scale from arrows that are not drawn.
    """
    m = field.mag[field.valid]
    if m.size == 0:
        return 0.0
    return float(np.percentile(m, 100.0 * float(p)))


def stride_for(n_gated: int, target: int) -> int:
    """Power-of-two stride giving about `target` arrows over the whole field.

    A power of two, and ANCHORED AT 0 in `lattice_sites`, so the lattices nest:
    stride-4 sites {0,4,8,...} contain every stride-8 site {0,8,16,...}, and
    thinning the field only removes arrows rather than relocating all of them.
    A centred anchor is the natural-looking choice and it is wrong — stride-4
    sites {2,6,10,...} and stride-8 sites {4,12,20,...} are entirely disjoint.
    """
    s = 1
    target = max(1, int(target))
    while s < 64 and n_gated / float(s * s) > target:
        s *= 2
    return s


def lattice_sites(g: Grid, field: Field, stride: int, max_arrows: int):
    """(rows, cols) of the gated sites on the stride lattice, capped in order.

    Row-major order, so the cap truncates from the bottom of the section rather
    than sampling — a truncated field is visibly partial, which is the honest
    failure. `stride_for` is what keeps the cap from being reached in practice.
    """
    stride = max(1, int(stride))
    sub = field.valid[::stride, ::stride]
    rows, cols = np.nonzero(sub)
    if rows.size > max_arrows:
        rows, cols = rows[:max_arrows], cols[:max_arrows]
    return rows * stride, cols * stride


def site_centres(g: Grid, rows, cols):
    """Data-space centre of each grid site. The arrow's anchor.

    Centre, not corner: the anchor marks where the measurement was taken.
    """
    x = g.min_x + (np.asarray(cols, dtype=np.float64) + 0.5) * g.cw
    y = g.min_y + (np.asarray(rows, dtype=np.float64) + 0.5) * g.ch
    return x, y


def cell_at(g: Grid, field: Field, x: float, y: float) -> int:
    """Flat grid index under a data-space point, or -1 outside / unsupported."""
    gx = np.floor((float(x) - g.min_x) / g.cw)
    gy = np.floor((float(y) - g.min_y) / g.ch)
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return -1
    gx, gy = int(gx), int(gy)
    if gx < 0 or gy < 0 or gx >= g.GX or gy >= g.GY:
        return -1
    return gy * g.GX + gx if field.mask[gy, gx] else -1


def spacing(x, y, g_bbox, n_probe=None) -> float:
    """Median inter-cell spacing, estimated from occupied-area over cell count.

    Bin onto a sqrt(n) x sqrt(n) lattice, count occupied bins, and take the
    square root of occupied area per cell. Robust to the shape of the tissue in
    a way a nearest-neighbour distance is not — it costs one pass and does not
    need a tree — and it is a LENGTH, which is what the grid cell size has to
    be derived from if the smoothing scale is to mean the same thing on a
    dataset with a hundred cells per square unit and on one with a hundredth.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n <= 0:
        return 1.0
    min_x, min_y, max_x, max_y = g_bbox
    G = int(max(16, min(1024, round(np.sqrt(n)))))
    w = max(max_x - min_x, 1e-9)
    h = max(max_y - min_y, 1e-9)
    cw, ch = w / G, h / G
    gx = np.clip(((x[ok] - min_x) / cw).astype(np.int64), 0, G - 1)
    gy = np.clip(((y[ok] - min_y) / ch).astype(np.int64), 0, G - 1)
    occupied = np.unique(gy * G + gx).size
    return float(np.sqrt((occupied * cw * ch) / n)) or 1.0
