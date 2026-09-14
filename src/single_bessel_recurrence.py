"""
Multi-l spherical Hankel transform via Eq. (33) of Bloomfield, Face &
Moss, "Indefinite Integrals of Spherical Bessel Functions"
(arXiv:1703.06428), using Miller's algorithm for the spherical Bessel
*function* values Eq(33)'s boundary term needs.

    I_l^n(z) = (l+n-1) I_{l-1}^{n-1}(z) - z^n j_{l-1}(z),   I_l^n(z) = int z^n j_l(z) dz

Unrolled down to the l=0 base case (I_0^n(z) = X_{n-1}(z) = eval_S(n-1,z)):

    I_l^n(z) = [prod_{i=0}^{l-1} c_i] X_{n-l-1}(z)
               - sum_{i=0}^{l-1} [prod_{j=0}^{i-1} c_j] z^{n-i} j_{l-i-1}(z)
    c_i = l + n - 1 - 2i

Why this is preferred over the Rayleigh sin/cos expansion (see
hankel_multi_l.py) for large l: that approach needs l+2 separate
*antiderivative* terms, each individually enormous (coefficients up to
~1e22 at l=23), all cancelling against EACH OTHER -- and this was found
to NOT resolve at any z, however large (still ~11% wrong even at
z=30000), because those coefficients are fixed, independent of z, and
antiderivatives decay far slower than the 1/z^m the point-value formula
divides by. Eq(33)'s boundary term instead uses ordinary spherical
Bessel FUNCTION values (bounded, |j_l(z)|<=1 always, via Miller's
algorithm) weighted by cumulative products -- only ONE antiderivative
term appears in the whole formula, not l+2 of them fighting each other.
Directly verified: the differenced quantity here converges to the
correct answer once z clears roughly 0.7*l, and STAYS accurate at any
larger z tested (checked up to z=1000) -- unlike the Rayleigh sum, no
floor, no gap.

For z below that, the same DLMF 10.53.1 small-argument series used in
hankel_multi_l.py is reused (imported from there) as a well-conditioned
fallback, since Eq(33)'s own leading term still fights its boundary sum
for small z (same cancellation shape as the Rayleigh case, just between
one term and a bounded sum instead of many huge terms).
"""

import numpy as np
from numba import njit
from trig_moment_integrals import eval_S, eval_C
from single_bessel_integrals import (get_linear_interp_coeffs,
                             _log_double_factorial_odd, _small_z_effective)


def spherical_jn_stack(z, l_max, buffer=20):
    """
    Compute j_0(z), ..., j_{l_max}(z) via Miller's algorithm (stable
    downward recursion), vectorized over array z.

    Returns array of shape (l_max+1,) + z.shape.

    The recursion must start well above both l_max AND max(z) (large
    arguments need a correspondingly high starting order for the
    seed's error to have decayed by the time the recursion reaches the
    orders actually wanted), but only the final l_max+1 levels are
    retained via two rolling buffers, so memory stays O(l_max * z.size)
    rather than O(l_start * z.size).
    """
    zmax = float(np.max(z))
    l_start = int(max(l_max + buffer, zmax + buffer))
    shape = z.shape

    result = np.empty((l_max + 1,) + shape)
    j_hi_p1 = np.zeros(shape)
    j_hi = np.ones(shape)
    if l_start <= l_max:
        result[l_start] = j_hi
    for l in range(l_start, 0, -1):
        j_lo = (2 * l + 1) / z * j_hi - j_hi_p1
        if l - 1 <= l_max:
            result[l - 1] = j_lo
        j_hi_p1, j_hi = j_hi, j_lo

    true_j0 = np.sin(z) / z
    scale = true_j0 / result[0]
    result *= scale[np.newaxis, ...]
    return result


def _I_l_n(arg_array, l, n, jstack):
    """int z^n j_l(z) dz at every grid point in arg_array (indefinite,
    up to the usual integration constant), via the unrolled Eq(33)
    recursion. jstack holds j_0(z)..j_{l-1}(z) at those same points
    (unused if l == 0)."""
    if l == 0:
        n0 = n - 1
        if n0 == -1:
            with np.errstate(divide='ignore', invalid='ignore'):
                out = eval_S(n0, arg_array)
            return np.where(arg_array == 0.0, 0.0, out)
        return eval_S(n0, arg_array)

    prod = 1.0
    boundary = np.zeros_like(arg_array)
    for i in range(l):
        boundary += prod * arg_array ** (n - i) * jstack[l - 1 - i]
        c = l + n - 1 - 2 * i
        prod = prod * c

    lead_n = n - l - 1
    if lead_n == -1:
        with np.errstate(divide='ignore', invalid='ignore'):
            lead = eval_S(lead_n, arg_array)
        lead = np.where(arg_array == 0.0, 0.0, lead)
    else:
        lead = eval_S(lead_n, arg_array)

    return prod * lead - boundary


def precompute_dI_eq33(x, freqs, l_values, small_z_factor=0.7, small_z_terms=40):
    """
    Precompute the per-l, interval-differenced dI0 = d[I_l^0(kx)],
    dI1 = d[I_l^1(kx)] arrays via the Eq(33) l-recursion, for use with
    `hankel_transform_multi_l_eq33`. Not part of the FFT-comparison hot
    path.

    Below z = small_z_factor * l (validated: 0.7 keeps accuracy within
    ~0.07% for l up to 23, ~1e-4 or better for smaller l -- see module
    tests), the well-conditioned small-argument series is used instead
    of Eq(33)'s own leading-term/boundary-sum combination, which -- like
    the Rayleigh sum -- suffers cancellation for kx small relative to l.
    l=0..3 are exact few-term cases with no cancellation risk at any z
    and are left untouched (matching the same exemption used for the
    Rayleigh method).

    Returns dI0, dI1, each of shape (len(l_values), Nx-1, Nk).
    """
    l_values = np.atleast_1d(np.asarray(l_values, dtype=int))
    l_max = int(l_values.max())
    arg_array = np.outer(x, freqs)
    Nx, Nk = arg_array.shape

    jstack = spherical_jn_stack(arg_array, max(l_max - 1, 0)) if l_max > 0 else None

    x0 = np.asarray(x)[:-1][:, None]
    x1 = np.asarray(x)[1:][:, None]
    k = np.asarray(freqs)[None, :]
    z0 = k * x0
    z1 = k * x1

    dI0 = np.empty((len(l_values), Nx - 1, Nk))
    dI1 = np.empty((len(l_values), Nx - 1, Nk))

    for li, l in enumerate(l_values):
        I0 = _I_l_n(arg_array, l, 0, jstack)
        I1 = _I_l_n(arg_array, l, 1, jstack)
        dI0[li] = np.diff(I0, axis=0)
        dI1[li] = np.diff(I1, axis=0)

        if l >= 4:
            thresh = small_z_factor * l
            mask = np.minimum(z0, z1) < thresh
            if np.any(mask):
                sz0, sz1 = _small_z_effective(l, z0, z1, n_terms=small_z_terms)
                dI0[li] = np.where(mask, sz0, dI0[li])
                dI1[li] = np.where(mask, sz1, dI1[li])

    return dI0, dI1


# Hot per-f kernel: same loop-order/single-threaded design as
# hankel_multi_l.py's kernels (i outer, j inner -- contiguous last
# axis -- no prange/target='parallel'), for the case where the l-
# specific arrays are already precomputed (no per-f Eq33 work at all).
@njit(fastmath=True, cache=True)
def _fused_sum_prediffed_multi_l_eq33(dI0, dI1, alpha0, alpha1,
                                       inv_k, inv_k2, out):
    n_l, Nint, Nk = dI0.shape
    for li in range(n_l):
        for j in range(Nk):
            out[li, j] = 0.0
        for i in range(Nint):
            a0 = alpha0[i]
            a1 = alpha1[i]
            for j in range(Nk):
                out[li, j] += (inv_k[j] * a0 * dI0[li, i, j]
                               + inv_k2[j] * a1 * dI1[li, i, j])


def hankel_transform_multi_l_eq33(x, fx, freqs, l_values, dI0, dI1):
    """
    Evaluate the order-l spherical Hankel transform of fx for every l in
    l_values at once, given dI0/dI1 from `precompute_dI_eq33`.

    Returns array of shape (len(l_values), len(freqs)).
    """
    l_values = np.atleast_1d(np.asarray(l_values, dtype=np.int64))
    interp_coeffs = get_linear_interp_coeffs(x, fx)
    inv_k = 1.0 / freqs
    inv_k2 = inv_k * inv_k

    out = np.empty((len(l_values), len(freqs)))
    _fused_sum_prediffed_multi_l_eq33(dI0, dI1, interp_coeffs[0], interp_coeffs[1],
                                       inv_k, inv_k2, out)
    return out