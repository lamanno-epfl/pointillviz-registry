"""Conditional contact-odds fit -- the optimised reference implementation.

Estimator: Besag pseudo-likelihood.  The conditional law of one cell given all
the others is a softmax over its neighbour-type counts, so maximising
sum_i log P(z_i | neighbours of i) is a multinomial logistic regression whose
design matrix N (n x K, neighbour-type counts) is FIXED.  Convex, exact
gradient, one build of N for the whole fit.

What was measured on the 135,232-cell / 73-type colon section, and what each
step bought (everything against a float64 reference solved on an A100 to
max|g|/n = 2.3e-8):

  float32 arithmetic         71 ms/eval against 145 ms in float64, with the
                             gradient agreeing to a cosine of 1.0 and a relative
                             sup-error of 3e-8.  The two reductions that CAN
                             lose digits -- the log-partition sum and the
                             gradient accumulation over row blocks -- stay in
                             float64, which is free.  The price is a floor: the
                             line search dies around max|g|/n = 1.7e-5, still
                             6x tighter than the tolerance that matters.
  row blocking               92 -> 78 ms/eval single-threaded.  The elementwise
                             passes (row max, exp, row sum, divide, one-hot
                             subtract) cost 51 ms against 31 ms for the two
                             GEMMs, so the win is cache residency, not flops.
  symmetric parametrisation  K(K+1)/2 = 2,701 free entries, not 5,329.
  column centring            An EXACT reparametrisation (see PL).  426 -> 374
                             iterations to max|g|/n <= 1e-4, and at a fixed
                             tolerance a much better answer: the largest error
                             over well-observed pairs falls 0.27 -> 0.018.
  diagonal preconditioner    TRIED AND DROPPED.  Rescaling by the closed-form
                             Hessian diagonal at the marginal frequencies
                             (`hess_diag`, still here behind precondition=True)
                             made it WORSE -- 635 iterations / 47 s against
                             374 / 26 s, at the same tolerance and the same
                             accuracy.  Recorded because it is the obvious thing
                             to reach for.
  step-size stopping         TRIED AND DROPPED.  See `fit`.

Gauge.  Adding a constant to every entry of J shifts cell i's logits by
c * deg(i), which the softmax removes; likewise a constant added to h.  Those
two directions are unidentifiable and are pinned by the ridge, which picks the
minimum-norm representative.  Compare fitted matrices only after centring.

Regularisation.  The ridge is also the conditioning knob: the Hessian diagonal
spans 9.4e5 at lam=1 and 9.5e3 at lam=100, and the iteration count tracks the
square root of that (2,445 float64 iterations to machine precision at lam=1;
311 at lam=100).  Of the 5,329 pairs, 3,631 see fewer than 100 contacts and 14
of the 73 types have fewer than 100 cells: those entries are set by the ridge,
not by the data, and they are also the directions the optimiser spends most of
its iterations on.

WHAT THIS DOES NOT ESTABLISH.  The fit is fast and it is correct as an
estimator, and the fitted J still does not describe the colon section: started
AT the observed labelling, Gibbs sampling under the fitted (J, h) walks away
from the data within ~25 sweeps (neighbour agreement 0.32 -> 0.95, largest type
share 0.11 -> 0.30), at every ridge tried.  That is why the tool that
imports this module reports the fit as conditional log-odds and offers no
generative reading of it: README section 2.
"""
import numpy as np

# ------------------------------------------------------------------ graph --
def undirected_edges(idx):
    n, k = idx.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idx.ravel().astype(np.int64)
    ok = (dst >= 0) & (dst != src)
    src, dst = src[ok], dst[ok]
    key = np.unique(np.minimum(src, dst) * np.int64(n) + np.maximum(src, dst))
    return np.stack([key // n, key % n], axis=1)


def neighbour_counts(edges, codes, K, n):
    i, j = edges[:, 0], edges[:, 1]
    ci, cj = codes[i], codes[j]
    m1, m2 = cj >= 0, ci >= 0
    flat = np.concatenate([i[m1] * K + cj[m1], j[m2] * K + ci[m2]])
    return np.bincount(flat, minlength=n * K).astype(np.float32).reshape(n, K)


# -------------------------------------------------------------- objective --
class PL:
    """Pseudo-likelihood in (J, h').

    `center` subtracts the column means of N.  This is an EXACT
    reparametrisation, not an approximation: the shifted term

        sum_b J_ab (N_ib - mbar_b)  =  sum_b J_ab N_ib  -  (J mbar)_a

    differs from the original logit by a quantity that depends only on the
    class a, so it is absorbed exactly by h' = h + J mbar.  J itself is
    untouched.  What it buys is conditioning: the raw N has column means
    proportional to type abundance, so N^T N is dominated by a rank-one mean
    outer product and the Hessian's largest eigenvalue is set by it.  Recover
    the field with `h = h_prime - J @ mbar`.
    """

    def __init__(self, N, codes, K, lam=1.0, dtype=np.float32, chunk=16384,
                 center=True):
        keep = codes >= 0
        self.N = np.ascontiguousarray(N[keep], dtype=dtype)
        self.mbar = self.N.mean(axis=0).astype(np.float64) if center \
            else np.zeros(K)
        if center:
            self.N -= self.mbar.astype(dtype)
        self.y = codes[keep].astype(np.int64)
        self.n = self.y.size
        self.K = K
        self.lam = float(lam)
        self.chunk = int(chunk)
        iu = np.triu_indices(K)
        self.iu = iu
        self.m = iu[0].size
        self.is_diag = iu[0] == iu[1]
        self.w = np.where(self.is_diag, 1.0, 2.0)
        self.nfev = 0
        self.dtype = dtype

    def unpack(self, theta):
        K = self.K
        t = theta[:self.m]
        J = np.zeros((K, K), dtype=self.dtype)
        J[self.iu[0], self.iu[1]] = t
        J[self.iu[1], self.iu[0]] = t
        return J, theta[self.m:].astype(self.dtype)

    def __call__(self, theta):
        self.nfev += 1
        J, h = self.unpack(theta)
        K, ch = self.K, self.chunk
        G = np.zeros((K, K), np.float64)
        gh = np.zeros(K, np.float64)
        nll = 0.0
        for s in range(0, self.n, ch):
            e = min(s + ch, self.n)
            Nc, yc = self.N[s:e], self.y[s:e]
            L = Nc @ J
            L += h
            r = np.arange(e - s)
            tl = L[r, yc].astype(np.float64)
            mx = L.max(axis=1, keepdims=True)
            L -= mx
            np.exp(L, out=L)
            Z = L.sum(axis=1, keepdims=True)
            nll += float(np.log(Z[:, 0].astype(np.float64)).sum()
                         + mx[:, 0].astype(np.float64).sum() - tl.sum())
            L /= Z
            L[r, yc] -= 1.0
            G += (Nc.T @ L).astype(np.float64)
            gh += L.sum(axis=0, dtype=np.float64)
        Gs = G + G.T
        gJ = Gs[self.iu]
        d = self.is_diag
        gJ[d] = G[self.iu[0][d], self.iu[1][d]]
        t, hh = theta[:self.m], theta[self.m:]
        nll += 0.5 * self.lam * float((self.w * t * t).sum() + (hh * hh).sum())
        gJ += self.lam * self.w * t
        gh += self.lam * hh
        return nll, np.concatenate([gJ, gh])


def hess_diag(N, codes, K, lam, center=True):
    """Closed-form diagonal of the Hessian at the marginal frequencies."""
    keep = codes >= 0
    Nk = np.ascontiguousarray(N[keep], np.float32)
    if center:
        Nk = Nk - Nk.mean(axis=0)
    y = codes[keep]
    n = y.size
    pi = np.bincount(y, minlength=K).astype(np.float64) / n
    S = (Nk.T @ Nk).astype(np.float64)
    q = pi * (1.0 - pi)
    Sd = np.diag(S).copy()
    a, b = np.triu_indices(K)
    hd = q[a] * Sd[b] + q[b] * Sd[a] - 2.0 * pi[a] * pi[b] * S[a, b]
    d = a == b
    hd[d] = 4.0 * q[a[d]] * Sd[a[d]]
    hd = np.maximum(hd, 0.0) + lam * np.where(d, 1.0, 2.0)
    return np.concatenate([hd, n * q + lam])


class _Stop(Exception):
    pass


def fit(N, codes, K, lam=1.0, gtol_rel=1e-4, jtol=None, maxiter=3000,
        maxcor=30, chunk=16384, dtype=np.float32, precondition=False,
        center=True, progress=None, trace=False):
    """Fit J, h.  Returns (J, h, info).

    Stopping.  The criterion is `max |dNLL/dtheta| <= gtol_rel * n`.  The
    gradient with respect to J_ab is an excess adjacency count -- how many more
    a|b contacts the model predicts than were observed -- so gtol_rel is read as
    "per cell", and it is the only test here that measures OPTIMALITY.

    A step-size test (`max |J_t - J_{t-1}| < jtol`) was tried first and dropped:
    a small step means either convergence or a crawl, and the two configurations
    that tripped it soonest were the ones crawling.  At jtol = 1e-3 the
    unpreconditioned fit stopped after 377 iterations with max|g|/n = 1.0e-3,
    while the preconditioned one ran 1,352 iterations to reach 2.9e-5: the
    "faster" configuration was simply further from the optimum.  jtol survives
    only as an optional early exit.

    Why 1e-4.  Against the float64 reference (max|g|/n = 2.3e-8), stopping at
    max|g|/n <= 1e-4 leaves the 1,698 pairs with at least 100 observed contacts
    correct to 0.018 in the worst entry and correlated 0.99992 with the
    reference, on a matrix whose spread over those pairs is sd 0.297.  A decade
    looser (1e-3) is not enough: 0.34 in the worst entry, correlation 0.958.
    The 3,631 sparse pairs are still moving at 1e-4 (worst entry 0.62) and would
    need roughly ten times the iterations to settle -- they are determined by
    the ridge rather than by the data, and waiting for them buys nothing.
    """
    from scipy.optimize import minimize
    obj = PL(N, codes, K, lam=lam, dtype=dtype, chunk=chunk, center=center)
    D = 1.0 / np.sqrt(hess_diag(N, codes, K, lam, center=center)) if precondition \
        else np.ones(obj.m + K)
    state = {"g": None, "theta": None}

    def fun(u):
        f, g = obj(u * D)
        state["g"], state["theta"] = g, u * D
        return f, g * D

    hist = []
    prev = {"J": np.zeros(obj.m), "hits": 0, "it": 0}

    def cb(intermediate_result):
        u = intermediate_result.x
        t = u * D
        dJ = np.abs(t[:obj.m] - prev["J"]).max()
        prev["J"] = t[:obj.m].copy()
        prev["it"] += 1
        gm = np.abs(state["g"]).max() if state["g"] is not None else np.inf
        if trace:
            hist.append((prev["it"], float(intermediate_result.fun), float(dJ), float(gm)))
        if progress is not None and prev["it"] % 10 == 0:
            progress(min(0.99, prev["it"] / 300.0), f"iteration {prev['it']}")
        if gm <= gtol_rel * obj.n:
            raise _Stop
        if jtol is not None and dJ < jtol:
            prev["hits"] += 1
            if prev["hits"] >= 2:
                raise _Stop
        else:
            prev["hits"] = 0

    try:
        res = minimize(fun, np.zeros(obj.m + K), jac=True, method="L-BFGS-B",
                       callback=cb,
                       options=dict(maxiter=maxiter, maxcor=maxcor,
                                    maxfun=4 * maxiter, ftol=1e-16,
                                    gtol=0.0))
        theta, nit, msg = res.x * D, int(res.nit), str(res.message)
    except _Stop:
        theta, nit, msg = state["theta"], prev["it"], "CONVERGENCE: max|g| <= gtol_rel * n"
    J, h = obj.unpack(theta)
    h = np.asarray(h, np.float64) - np.asarray(J, np.float64) @ obj.mbar
    g = state["g"]
    info = dict(nit=nit, nfev=obj.nfev, message=msg,
                gmax=float(np.abs(g).max()), gmax_per_cell=float(np.abs(g).max() / obj.n),
                gtol_met=bool(np.abs(g).max() <= gtol_rel * obj.n),
                n_used=obj.n, n_params=obj.m + K, hist=hist)
    return np.asarray(J, np.float64), h, info
