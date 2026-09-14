import math
import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from trig_moment_integrals import eval_S, eval_C


def get_linear_interp_coeffs(x, fx):
    N = len(x)
    alphas = np.zeros((2, N - 1))
    for i in range(N - 1):
        alphas[1][i] = (fx[i + 1] - fx[i]) / (x[i + 1] - x[i])
        alphas[0][i] = fx[i] - x[i] * alphas[1][i]
    return alphas


# ---------------------------------------------------------------------
# Stage 1: precompute the n-stacked, interval-differenced S_n(kx),
# C_n(kx) arrays. Depends only on x, freqs, and the term cap m_max --
# NOT on l or f. Because the inverse-x expansion is truncated at
# m_max terms, this n-range is now FIXED regardless of l: n = -m and
# n = 1-m for m in [0, m_max] always lands in n in [-m_max, 1], no
# matter how large l gets. NOT part of the FFT-comparison hot path, so
# free to use multiple threads: eval_S/eval_C are numba-compiled and
# release the GIL, so a plain ThreadPoolExecutor genuinely parallelizes
# the per-n table builds.
# ---------------------------------------------------------------------

def choose_m_max(l_max, u, v, x_min, k_min, tol=1e-7):
    """
    Choose an m_max for the Rayleigh sin/cos expansion cutoff from the
    actual problem being solved, instead of a fixed guess.

    The individual terms u[l,m]*(kx)^{-m}, v[l,m]*(kx)^{-m} are largest
    at the SMALLEST argument the problem will actually evaluate,
    kx_min = k_min * x_min (since x_min, k_min are the limits of
    integration/frequency range). Using l_max's row of u, v (the
    coefficients that grow fastest with l, so the worst case across all
    requested l), m_max is set to the largest m at which either term's
    magnitude at kx_min still exceeds `tol` -- i.e. every dropped term
    is, at worst, this small at the smallest argument the run will see.

    This is a rough/heuristic check, as requested: term magnitudes are
    not guaranteed to decrease monotonically in m (especially once
    kx_min is not large compared to l -- see the earlier truncation
    failure at l=10 with a fixed m_max=10), so this does not replace
    checking accuracy against a reference for a given problem. It is a
    cheap scalar computation (no cost added to the hot per-f path);
    the only added cost is summing over however many terms this decides
    are actually needed, which may end up larger OR smaller than a
    fixed guess, whichever the problem actually requires.
    """
    kx_min = k_min * x_min
    if kx_min <= 0:
        # can't safely truncate at all if the smallest argument is 0
        return l_max + 1

    m_range = np.arange(0, l_max + 2)
    u_row = np.abs(u[l_max, :l_max + 2].astype(np.float64))
    v_row = np.abs(v[l_max, :l_max + 2].astype(np.float64))
    with np.errstate(over='ignore'):
        terms = np.maximum(u_row, v_row) * kx_min ** (-m_range.astype(np.float64))

    above = np.where(terms > tol)[0]
    if len(above) == 0:
        return 1  # even m=0 is already below tol; keep at least one term
    return int(above.max())


def precompute_dS_dC(x, freqs, m_max=10, n_workers=None):
    """
    Precompute S_n(kx), C_n(kx) (interval-differenced) for every n in
    [-m_max, 1] -- the full set needed by the truncated Rayleigh-formula
    expansion (see `hankel_transform_multi_l`) for ANY l, since terms
    with inverse-x power beyond m_max are dropped regardless of l.

    Returns
    -------
    n_values : (m_max+2,) int array, n_values[i] - n_values[0] == i
    dS, dC   : (m_max+2, Nx-1, Nk) arrays of S_n(k x_{j+1}) - S_n(k x_j)
               (resp. C_n) for each n, interval, and frequency.
    """
    arg_array = np.outer(x, freqs)
    n_min, n_max = -m_max, 1
    n_values = np.arange(n_min, n_max + 1)
    Nx, Nk = arg_array.shape
    dS = np.empty((len(n_values), Nx - 1, Nk))
    dC = np.empty((len(n_values), Nx - 1, Nk))

    def _compute_one(i):
        n = int(n_values[i])
        if n == -1:
            # Si(0) = 0 exactly, but the library's fused Si/Ci kernel
            # returns NaN at an argument of exactly 0 (0 * Ci(0) = 0 *
            # -inf internally, even though Ci's coefficient is 0 here).
            with np.errstate(divide='ignore', invalid='ignore'):
                Sn = eval_S(n, arg_array)
            Sn = np.where(arg_array == 0.0, 0.0, Sn)
        else:
            Sn = eval_S(n, arg_array)
        Cn = eval_C(n, arg_array)
        dS[i] = np.diff(Sn, axis=0)
        dC[i] = np.diff(Cn, axis=0)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_compute_one, range(len(n_values))))

    return n_values, dS, dC


# ---------------------------------------------------------------------
# Stage 2a: fold the u[l,m]/v[l,m]-weighted combination into a per-l
# effective (term0, term1) array ONCE, ahead of time -- rather than
# redoing it inside the hot per-f kernel on every call. This is what
# `hankel_transform_multi_l` now consumes directly. Depends on l (and
# m_max) but NOT on f, so it is computed once per (x, freqs, l_values)
# and reused across every fx. Not part of the FFT-comparison hot path,
# so this plain-numpy loop is fine as-is.
#
# Tradeoff versus not doing this (see `hankel_transform_multi_l_bounded`
# below): this array's memory is O(n_l * Nx * Nk), growing with however
# many l values you request, in exchange for a hot path with no m-loop
# at all. If you need very many/large l values and memory is the
# binding constraint, use `hankel_transform_multi_l_bounded` instead,
# which keeps memory fixed at O(m_max * Nx * Nk) by paying the
# reweighting cost on every call instead.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Small-kx fix: replace the ill-conditioned Rayleigh sin/cos sum with
# the analytic small-argument series for j_l wherever it is not
# trustworthy. Both helpers are plain numpy/python (not numba) -- this
# only ever runs inside `precompute_effective_dS`, which is precompute,
# not the hot per-f path, so there's no speed requirement here.
#
# Investigation notes (found via a real failure at l=23, M=30): two
# successively-refined, and successively more subtle, bugs were fixed
# to get here.
#
# (1) An early threshold based only on the small-z series' OWN
#     truncation error badly underestimated how far the Rayleigh
#     method's cancellation problem extends for larger l -- validated
#     against l=13 alone, it left a ~7-unit-wide dead zone in z for
#     l=23 where NEITHER method was accurate.
#
# (2) Fixing (1) by basing the threshold on the Rayleigh method's own
#     cancellation risk (`_rayleigh_cancellation_log_ratio`, a point-
#     value diagnostic) still wasn't enough: dS_eff0/dS_eff1 are
#     *differences* of that (already only approximately accurate)
#     quantity over a narrow interval, and each point's own error does
#     NOT shrink as the interval narrows, while the signal being
#     differenced does -- so the diagnostic needed an additional,
#     interval-width-dependent margin (`_small_z_mask`) on top of the
#     point-value check, or narrow intervals near the crossover could
#     still be wrong by orders of magnitude despite passing the
#     point-value check comfortably.
#
# Residual caveat, found during calibration and not fully resolved:
# even with margin=30, a residual error on the order of 1e-4 (for
# l=23; likely worse for even larger l) can remain in a narrow z range
# right at the crossover, and does NOT shrink with finer x sampling --
# accuracy stopped improving monotonically with margin in the
# calibration data, so this appears to be close to an intrinsic float64
# precision limit of narrow-interval Rayleigh differencing at high l,
# not something this margin alone fully removes. If this floor matters
# for l this large in your application, treat it as a real limitation
# to check against a reference (e.g. hankel_eq33.py's Eq(33)-based
# method) rather than something further margin tuning will clear up.
# ---------------------------------------------------------------------

def _log_double_factorial_odd(l):
    """log((2l+1)!!), via (2l+1)!! = (2l+1)!/(2^l l!) and lgamma, to
    avoid ever forming the (huge, for large l) raw double factorial."""
    return math.lgamma(2 * l + 2) - l * math.log(2) - math.lgamma(l + 1)


def _rayleigh_cancellation_log_ratio(l, z, u, v):
    """
    log(largest individual Rayleigh term magnitude) minus log(expected
    true magnitude of j_l(z), ~z^l/(2l+1)!!), at scalar or array z. A
    large positive value means the raw sin/cos Rayleigh sum needs that
    many natural-log units (divide by log(10) for decimal digits) of
    cancellation to reach the true answer -- i.e. how untrustworthy
    Rayleigh is at this z, given float64's ~36 natural-log-unit budget.

    NOTE: this measures cancellation risk for a single POINT evaluation
    of j_l(z). It is NOT sufficient on its own to judge whether the
    *differenced* quantity dS_eff0/dS_eff1 = stuff(z1) - stuff(z0) over
    a narrow interval is trustworthy -- see `_small_z_mask` for why an
    additional, interval-width-dependent margin is needed on top of
    this.
    """
    z = np.asarray(z, dtype=np.float64)
    m_range = np.arange(0, l + 2, dtype=np.float64)
    u_row = np.abs(u[l, :l + 2].astype(np.float64))
    v_row = np.abs(v[l, :l + 2].astype(np.float64))
    with np.errstate(over='ignore', divide='ignore'):
        terms = np.maximum(u_row, v_row)[:, None] * z[None, :] ** (-m_range[:, None])
        worst = np.max(terms, axis=0)
        log_true_scale = l * np.log(z) - _log_double_factorial_odd(l)
        return np.log(worst) - log_true_scale


_LOG_MACHINE_EPS = float(np.log(np.finfo(np.float64).eps))  # ~ -36.04


def _small_z_mask(l, z0, z1, u, v, margin=30.0):
    """
    True where the Rayleigh sum should NOT be trusted for the
    *differenced* quantity dS_eff0/dS_eff1 = stuff(z1) - stuff(z0), and
    the small-z series should be used instead.

    Important subtlety this fixes (found by investigating a real
    failure at l=23): `_rayleigh_cancellation_log_ratio` only measures
    cancellation risk for a single point value of j_l(z). But dS_eff0/1
    are *differences* of that (already only approximately accurate)
    quantity over a narrow interval Delta_z = z1 - z0. Each point's own
    error is roughly eps * (largest term magnitude), and this error does
    NOT shrink as the interval narrows -- while the true signal being
    differenced DOES shrink (roughly proportional to Delta_z). So the
    narrower the interval, the more margin is needed beyond what the
    point-value diagnostic alone suggests: trustworthy requires

        log_ratio(z) < log(Delta_z) - log(machine_eps) - margin

    margin=30 was calibrated by direct comparison against
    scipy.integrate.quad for the specific dS_eff0/dS_eff1 formulas (not
    just point values of j_l) at l=23 -- see module test notes. Some
    residual error (a few percent) can remain right at this boundary
    even so; this is a real, apparently intrinsic limit of the narrow-
    interval Rayleigh differencing (not something a larger margin alone
    removes -- accuracy stopped improving monotonically with margin in
    the calibration data). Widening the small-z-series region further
    is the mitigation, not raising this margin indefinitely.
    """
    dz = np.abs(z1 - z0)
    with np.errstate(divide='ignore'):
        log_dz = np.log(dz)
    log_ratio = _rayleigh_cancellation_log_ratio(l, np.minimum(z0, z1).ravel(), u, v)
    log_ratio = log_ratio.reshape(z0.shape)
    return log_ratio > (log_dz - _LOG_MACHINE_EPS - margin)


def _small_z_effective(l, z0, z1, n_terms=20):
    """
    dS_eff0, dS_eff1 (matching the (a0/k)*dS_eff0 + (a1/k^2)*dS_eff1
    convention) from the n_terms-term small-argument series for j_l
    (DLMF 10.53.1):

        j_l(z) = z^l * sum_k a_k z^(2k),   a_0 = 1/(2l+1)!!
                 a_k = a_{k-1} * (-1) / (2k*(2l+2k+1))

    integrated exactly, term by term, as a polynomial -- in z-space
    directly (z0=k*x_i, z1=k*x_{i+1}), not x and k separately, so
    there's no overflow risk even when x is large but k is
    proportionally small. a_k is built via the running ratio above
    (never forming a raw factorial), and z itself is bounded/moderate
    in the regime this is used, so summing more terms doesn't introduce
    new overflow -- validated empirically up to n_terms~20 closing the
    gap cleanly even at l=23 (see module notes).
    """
    a_k = math.exp(-_log_double_factorial_odd(l))  # a_0
    dS_eff0 = np.zeros_like(z0, dtype=np.float64)
    dS_eff1 = np.zeros_like(z0, dtype=np.float64)
    for k in range(n_terms):
        p0 = l + 2 * k + 1
        p1 = l + 2 * k + 2
        dS_eff0 += a_k / p0 * (z1 ** p0 - z0 ** p0)
        dS_eff1 += a_k / p1 * (z1 ** p1 - z0 ** p1)
        a_k *= -1.0 / (2 * (k + 1) * (2 * l + 2 * (k + 1) + 1))
    return dS_eff0, dS_eff1


def precompute_effective_dS(l_values, u, v, n_values, dS, dC, m_max=10,
                             x=None, freqs=None, small_z_terms=40,
                             small_z_margin=30.0):
    """
    Combine the n-stacked dS/dC (from `precompute_dS_dC`) into per-l
    effective (term0, term1) arrays using the Rayleigh sin/cos
    coefficient tables u[l,m], v[l,m]:

        dS_eff0[l] = sum_m u[l,m]*dS[n=-m]  + v[l,m]*dC[n=-m]
        dS_eff1[l] = sum_m u[l,m]*dS[n=1-m] + v[l,m]*dC[n=1-m]

    so that (a0/k)*dS_eff0[l] + (a1/k^2)*dS_eff1[l], summed over
    intervals, is int g(x) j_l(kx) dx for piecewise-linear g -- see
    `hankel_transform_multi_l`'s docstring for the derivation. Feed the
    result straight to `hankel_transform_multi_l`.

    Small-kx fix: the Rayleigh sum above suffers catastrophic
    cancellation when kx is small relative to l (individual u[l,m],
    v[l,m] terms reach huge magnitude while the true j_l(kx) is tiny).
    If x and freqs are provided, intervals where kx falls below a
    per-l threshold (`_small_z_threshold`) instead use the analytic
    2-term small-argument series for j_l (DLMF 10.53.1), integrated
    exactly as a polynomial -- no sin/cos, no cancellation, since it's
    computed directly in z=kx space (bounded/small by construction, so
    no overflow risk even when x is large but k is proportionally
    small). This is a rough, theory-based threshold (see
    `_small_z_threshold`), not a per-problem-calibrated one -- same
    spirit as `choose_m_max`. Pass x, freqs to enable it; omitting them
    reproduces the previous (uncorrected) behavior exactly.

    Returns l_values, dS_eff0, dS_eff1, each of shape
    (len(l_values), Nx-1, Nk).
    """
    n_min = int(n_values[0])
    l_values = np.atleast_1d(np.asarray(l_values, dtype=int))
    Nint, Nk = dS.shape[1], dS.shape[2]

    dS_eff0 = np.zeros((len(l_values), Nint, Nk))
    dS_eff1 = np.zeros((len(l_values), Nint, Nk))

    if x is not None and freqs is not None:
        x0 = np.asarray(x)[:-1][:, None]   # (Nint, 1)
        x1 = np.asarray(x)[1:][:, None]    # (Nint, 1)
        k = np.asarray(freqs)[None, :]     # (1, Nk)
        z0 = k * x0                        # (Nint, Nk), broadcasts
        z1 = k * x1

    for li, l in enumerate(l_values):
        for m in range(min(l + 2, m_max + 1)):
            um, vm = u[l, m], v[l, m]
            if um == 0 and vm == 0:
                continue
            idx0 = (-m) - n_min
            idx1 = (1 - m) - n_min
            if um != 0:
                dS_eff0[li] += um * dS[idx0]
                dS_eff1[li] += um * dS[idx1]
            if vm != 0:
                dS_eff0[li] += vm * dC[idx0]
                dS_eff1[li] += vm * dC[idx1]

        if x is not None and freqs is not None and l >= 4:
            # l=0..3 have exact few-term closed forms (see interp_jl's
            # own treatment) with no cancellation risk at any z -- the
            # small-z series is only an approximation, so applying it
            # there would trade an exact result for a less accurate
            # one. Only l>=4, where genuine cancellation risk exists,
            # is eligible for the substitution.
            mask = _small_z_mask(l, z0, z1, u, v, margin=small_z_margin)
            if np.any(mask):
                sz0, sz1 = _small_z_effective(l, z0, z1, n_terms=small_z_terms)
                dS_eff0[li] = np.where(mask, sz0, dS_eff0[li])
                dS_eff1[li] = np.where(mask, sz1, dS_eff1[li])

    return l_values, dS_eff0, dS_eff1


# Hot per-f kernel: no m-loop, no u/v, no n_min bookkeeping -- just the
# two-term weighted sum per (l, interval, frequency), matching the
# hankel_eq33.py kernel exactly (both precomputed-effective-array
# approaches reduce to the identical operation once the l-weighting is
# already done). Parallelized over l (numba.prange): each l writes only
# to its own out[li,:] and reads only its own dS_eff0[li]/dS_eff1[li],
# so this is a clean, race-free split across threads. Parallelizing
# over l rather than the frequency axis j is deliberate: j is the
# contiguous, fastest-varying axis of dS_eff0/dS_eff1 (see the much
# earlier loop-order benchmark -- ~17x difference from this alone), so
# each thread doing a full i-outer/j-inner sweep for its own l values
# keeps that cache-friendly access pattern intact; parallelizing over j
# instead would force strided per-thread memory access. Scales best
# when the number of l values is comparable to or larger than the
# available core count -- with very few l's, there's limited work to
# split across threads regardless of how it's parallelized here.
@njit(fastmath=True, cache=True, parallel=True)
def _fused_effective_multi_l(dS_eff0, dS_eff1, alpha0, alpha1,
                              inv_k, inv_k2, out):
    n_l, Nint, Nk = dS_eff0.shape
    for li in prange(n_l):
        for j in range(Nk):
            out[li, j] = 0.0
        for i in range(Nint):
            a0 = alpha0[i]
            a1 = alpha1[i]
            for j in range(Nk):
                out[li, j] += (inv_k[j] * a0 * dS_eff0[li, i, j]
                               + inv_k2[j] * a1 * dS_eff1[li, i, j])


def hankel_transform_multi_l(x, fx, freqs, dS_eff0, dS_eff1):
    """
    Evaluate the order-l spherical Hankel transform of fx (piecewise-
    linear interpolation) for every l at once, given the per-l
    dS_eff0/dS_eff1 from `precompute_effective_dS`. No u, v, or m-loop
    here -- that reweighting cost has already been paid once, in
    `precompute_effective_dS`, and is reused across every fx you pass
    in on this same (x, freqs, l_values).

    Returns array of shape (len(l_values), len(freqs)).
    """
    interp_coeffs = get_linear_interp_coeffs(x, fx)
    inv_k = 1.0 / freqs
    inv_k2 = inv_k * inv_k

    n_l = dS_eff0.shape[0]
    out = np.empty((n_l, len(freqs)))
    _fused_effective_multi_l(dS_eff0, dS_eff1, interp_coeffs[0],
                              interp_coeffs[1], inv_k, inv_k2, out)
    return out


# ---------------------------------------------------------------------
# Bounded-memory alternative: kept under its own name for when you have
# very many/large l values and can't afford the O(n_l * Nx * Nk)
# footprint of dS_eff0/dS_eff1 above. Does the u[l,m]/v[l,m]-weighted
# combination *inside* the compiled kernel instead of precomputing it,
# so memory stays fixed at O(m_max * Nx * Nk) regardless of how many l
# values are requested, at the cost of redoing that reweighting work on
# every call. Parallelized over l (numba.prange), same reasoning as
# `_fused_effective_multi_l` above: each l is independent, and keeping
# l as the parallelized axis (rather than the frequency axis j)
# preserves the cache-friendly i-outer/j-inner access pattern within
# each thread's work.
# ---------------------------------------------------------------------

@njit(fastmath=True, cache=True, parallel=True)
def _fused_multi_l_full(dS, dC, u_sel, v_sel, alpha0, alpha1,
                         inv_k, inv_k2, n_min, out):
    n_l, n_m = u_sel.shape
    Nint = dS.shape[1]
    Nk = dS.shape[2]

    for li in prange(n_l):
        for j in range(Nk):
            out[li, j] = 0.0
        for i in range(Nint):
            a0 = alpha0[i]
            a1 = alpha1[i]
            for m in range(n_m):
                um = u_sel[li, m]
                vm = v_sel[li, m]
                if um == 0.0 and vm == 0.0:
                    continue
                idx0 = (-m) - n_min       # index of n = -m
                idx1 = (1 - m) - n_min    # index of n = 1-m
                c0S = um * a0
                c1S = um * a1
                c0C = vm * a0
                c1C = vm * a1
                for j in range(Nk):
                    out[li, j] += (inv_k[j] * (c0S * dS[idx0, i, j]
                                                + c0C * dC[idx0, i, j])
                                   + inv_k2[j] * (c1S * dS[idx1, i, j]
                                                  + c1C * dC[idx1, i, j]))


def hankel_transform_multi_l_bounded(x, fx, freqs, l_values, u, v, n_values,
                                      dS, dC, m_max=10):
    """
    Same as `hankel_transform_multi_l`, but does the u[l,m]/v[l,m]
    reweighting inside the hot kernel on every call instead of consuming
    a precomputed dS_eff0/dS_eff1 -- trades hot-path speed for memory
    that stays fixed regardless of how many l values are requested.
    See `precompute_dS_dC` for n_values/dS/dC.

    Returns array of shape (len(l_values), len(freqs)).
    """
    l_values = np.atleast_1d(np.asarray(l_values, dtype=np.int64))
    n_min = int(n_values[0])
    m_terms = m_max + 1

    u_sel = np.ascontiguousarray(u[l_values, :m_terms])
    v_sel = np.ascontiguousarray(v[l_values, :m_terms])

    interp_coeffs = get_linear_interp_coeffs(x, fx)
    inv_k = 1.0 / freqs
    inv_k2 = inv_k * inv_k

    out = np.empty((len(l_values), len(freqs)))
    _fused_multi_l_full(dS, dC, u_sel, v_sel, interp_coeffs[0],
                         interp_coeffs[1], inv_k, inv_k2, n_min, out)
    return out