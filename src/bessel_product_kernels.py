"""
bessel_product_kernels.py
=========================

Integral kernels for

    I_l(k1, k2)  =  ∫_a^b  x^n  j_l(k1 x)  j_{l+N}(k2 x)  dx ,      0 < a < b,

with n and N integers (N may be negative). The kernels are built so that they
can be passed straight to the Legendre transform in cheb_legendre_transform.py.

How it works
------------
1.  Product formula. With R = κ x and

        κ(μ) = sqrt(k1² + k2² - 2 k1 k2 μ),        Q = |N|,

    the product of the two spherical Bessel functions is

        j_l(k1 x) j_{l+N}(k2 x)  =  1/2 ∫_{-1}^{1} dμ  j_Q(κ x)/κ^Q  Π(μ),

    where Π(μ) is a binomial combination of Legendre polynomials (see
    `binomial_weights`). All the x-dependence sits in j_Q(κ x)/κ^Q.

2.  Swap the integrals. The x-integral is done analytically:

        F(μ)  =  ∫_a^b x^n j_Q(κ x)/κ^Q dx
              =  κ^{-(Q+n+1)} [ G(κ b) - G(κ a) ],     G' (t) = t^n j_Q(t),

    and G is a finite combination of sin, cos, Si and Ci (`antiderivative`).
    F depends on (k1, k2, μ) only through κ. It is an entire function of
    κ² = k1² + k2² - 2 k1 k2 μ, so it is smooth in μ and Chebyshev
    interpolation converges fast.

3.  Legendre transform. The uploaded module computes, for every l,

        T_l  =  ∫_{-1}^{1} F(μ) P_l(μ) dμ          (no (2l+1)/2 factor).

4.  Combination. The integral is a short sum of shifted transforms:

        N >= 0:   I_l = 1/2 Σ_{k=0}^{Q} C(Q,k) k2^{Q-k} (-k1)^k  T_{l+k}
        N <  0:   I_l = 1/2 Σ_{k=0}^{Q} C(Q,k) k1^{Q-k} (-k2)^k  T_{l-Q+k}

    (`combine_legendre_integrals`). Entries of l that would need T beyond the
    computed range, or a negative Bessel order, are returned as NaN.

Accuracy
--------
The kernel itself is accurate to ~1e-14 relative. The binomial sum in step 4
alternates in sign, and for k1 ≈ k2 with large b and |N| >= 2 its terms can
be many orders of magnitude larger than the result, so digits cancel.
`cancellation_factor` returns that ratio on the same grid: the expected
relative error is about 1e-16 times it.
The number of Chebyshev points must resolve the kernel (about b*min(k1,k2)
coefficients) and reach l_max + |N|; see `recommended_n_points`.

Typical use
-----------
    from cheb_legendre_transform import DenseChebyshevLegendreTransform
    import bessel_product_kernels as bpk

    k1 = np.linspace(0.1, 2.0, 40)
    k2 = np.linspace(0.1, 2.0, 50)
    M  = bpk.recommended_n_points(k1.max(), k2.max(), b=10.0, l_max=60, N=2)
    transform = DenseChebyshevLegendreTransform(M, kind=2)

    I = bpk.bessel_product_integrals(k1, k2, n=2, N=2, a=0.5, b=10.0,
                                     transform=transform)
    # I.shape == (40, 50, M);  I[i, j, l] = ∫ x^2 j_l(k1_i x) j_{l+2}(k2_j x) dx

Or step by step, if you want to keep or reuse the kernel array:

    K1, K2, MU = bpk.build_mesh(k1, k2, M, kind=2)      # shapes (n1,1,1), (1,n2,1), (1,1,M)
    values     = bpk.integral_kernel(K1, K2, MU, n=2, N=2, a=0.5, b=10.0)   # (n1, n2, M)
    T          = transform(values)                                         # (n1, n2, M)
    I          = bpk.combine_legendre_integrals(T, K1, K2, N=2)            # (n1, n2, M)

Piecewise-polynomial weights (many tables in one pass)
-------------------------------------------------------
For integrals ∫ S(x) x^u j_l(k x) j_{l+N}(q x) dx with S a piecewise
polynomial (several S on the same edges), use `PiecewisePolynomial` and
`compute_tables`. Special functions are evaluated once per interval edge
and reused for every power, |N|, request and weight; see the section
"Piecewise-polynomial weights" below.

    weights = {"b": bpk.PiecewisePolynomial.fit(S_b, edges, degree=5),
               "f": bpk.PiecewisePolynomial.fit(S_f, edges, degree=5)}
    tables = bpk.compute_tables(ks, qs, weights,
                                requests=[("b", 2, 0), ("f", 2, -2), ("f", 2, 2)],
                                transform=transform)
    tables[("f", 2, -2)][i, j, l]   # ∫ S_f(x) x² j_l(ks[i] x) j_{l-2}(qs[j] x) dx
"""

from dataclasses import dataclass, field, replace
from math import comb, factorial, pi

import numpy as np
from scipy.special import sici


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def i_power(k):
    """i**k for an integer k, exactly (no rounding from complex powers)."""
    return (1, 1j, -1, -1j)[k % 4]


def beta_coefficient(order, m):
    """
    β_{Q,m} = (Q+m)! / (2^m m! (Q-m)!), the coefficients in the finite form

        j_Q(t) = Σ_{m=0}^{Q} β_{Q,m} sin(t - (Q-m)π/2) / t^{m+1}.
    """
    return factorial(order + m) / (2**m * factorial(m) * factorial(order - m))


def double_factorial_odd(k):
    """(2k+1)!! = 1 * 3 * 5 * ... * (2k+1)."""
    result = 1
    for odd in range(3, 2 * k + 2, 2):
        result *= odd
    return result


def kappa(k1, k2, mu):
    """κ = sqrt(k1² + k2² - 2 k1 k2 μ), broadcast over the inputs. Never negative."""
    kappa_squared = k1 * k1 + k2 * k2 - 2.0 * k1 * k2 * mu
    return np.sqrt(np.maximum(kappa_squared, 0.0))   # clip rounding below zero


# ---------------------------------------------------------------------------
# Closed-form antiderivative  G(t) = ∫ t^n j_Q(t) dt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AntiderivativeCoefficients:
    """
    G(t) = Im[ e^{it} Σ_s g_s t^s ]  +  c [ cos Θ Si(t) - sin Θ Ci(t) ],

    with s running over `powers` and Θ = (Q + n)π/2. Only one of Si, Ci ever
    appears: Si when Q + n is even, Ci when Q + n is odd (and then only for
    n <= -Q - 1); c is zero otherwise.
    """
    n: int
    order: int                 # Q = |N|
    powers: tuple              # the exponents s
    g: tuple                   # complex coefficients g_s, same order as powers
    trig_integral_coeff: float # c
    theta: float               # Θ


def antiderivative_coefficients(n, order):
    """Coefficients of G(t) = ∫ t^n j_Q(t) dt, for integers n and Q >= 0."""
    Q = order
    powers = tuple(range(min(n - Q, 0), max(n, 0)))
    g = []
    for s in powers:
        if s >= 0:
            total = sum(beta_coefficient(Q, m) * factorial(n - m - 1)
                        for m in range(0, min(Q, n - 1 - s) + 1))
            g.append(-i_power(n - Q - s) / factorial(s) * total)
        else:
            total = sum((-1)**m * beta_coefficient(Q, m) / factorial(m - n)
                        for m in range(max(n - s, 0), Q + 1))
            g.append(-i_power(s - Q - n) * factorial(-s - 1) * total)

    c = sum((-1)**m * beta_coefficient(Q, m) / factorial(m - n)
            for m in range(max(n, 0), Q + 1))

    return AntiderivativeCoefficients(n=n, order=Q, powers=powers, g=tuple(g),
                                      trig_integral_coeff=c,
                                      theta=(Q + n) * pi / 2)


def antiderivative(t, coeffs):
    """
    G(t) = ∫ t^n j_Q(t) dt in closed form, vectorized over t > 0.

    Accurate for moderate and large t. For small t the negative powers of t
    cancel against each other, so `radial_kernel` switches to a power series
    there instead of calling this function.
    """
    t = np.asarray(t, dtype=float)

    polynomial = np.zeros(t.shape, dtype=complex)
    for s, g_s in zip(coeffs.powers, coeffs.g):
        polynomial += g_s * t**s
    elementary = np.imag(np.exp(1j * t) * polynomial)

    if coeffs.trig_integral_coeff == 0.0:
        return elementary

    si, ci = sici(t)
    return elementary + coeffs.trig_integral_coeff * (
        np.cos(coeffs.theta) * si - np.sin(coeffs.theta) * ci)


# ---------------------------------------------------------------------------
# Power series for small arguments
# ---------------------------------------------------------------------------

def bessel_series_coefficients(order, n_terms):
    """
    c_k with  j_Q(t) = t^Q Σ_k c_k t^{2k},  c_k = (-1)^k / (2^k k! (2Q+2k+1)!!).

    Built by the recursion c_{k+1} = -c_k / (2 (k+1) (2Q+2k+3)).
    """
    c = np.empty(n_terms)
    c[0] = 1.0 / double_factorial_odd(order)
    for k in range(n_terms - 1):
        c[k + 1] = -c[k] / (2.0 * (k + 1) * (2 * order + 2 * k + 3))
    return c


def power_integral(lower, upper, exponent):
    """∫_lower^upper x^exponent dx, including the logarithmic case exponent = -1."""
    if exponent == -1:
        return np.log(upper / lower)
    return (upper**(exponent + 1) - lower**(exponent + 1)) / (exponent + 1)


def series_kernel(kap, n, order, a, b, series_coeffs):
    """
    F(κ) = ∫_a^b x^n j_Q(κx)/κ^Q dx  =  Σ_k c_k κ^{2k} ∫_a^b x^{n+Q+2k} dx.

    No division by κ, so this is exact at κ = 0 (k1 = k2, μ = 1).
    Use only where κ b is small.

    Evaluated in the scaled variable y = x / b, as a polynomial in (κ b)²,
    so that the moments stay bounded however many series terms are used
    (plain ∫ x^{n+Q+2k} dx overflows for large b and k).
    """
    poly_coeffs = [c_k * b**(n + order + 1) * power_integral(a / b, 1.0, n + order + 2 * k)
                   for k, c_k in enumerate(series_coeffs)]
    return horner((kap * b)**2, poly_coeffs)


def horner(x, coeffs):
    """
    Σ_k coeffs[k] x^k for an array x, by Horner's rule with in-place updates
    (one multiply and one add per coefficient, no temporary arrays).
    """
    result = np.full(np.shape(x), coeffs[-1], dtype=float)
    for c in coeffs[-2::-1]:
        result *= x
        result += c
    return result


def series_antiderivative(t, n, order, series_coeffs):
    """
    S(t) with S'(t) = t^n j_Q(t), from the power series of j_Q:

        S(t) = Σ_k c_k t^{e_k} / e_k,     e_k = n + Q + 1 + 2k,

    except that a term with e_k = 0 is c_k ln t. Written as one power times
    a polynomial in t²,

        S(t) = t^{e_0} Σ_k d_k (t²)^k  +  c_{k*} ln t,      d_k = c_k / e_k,

    so the array work is one power, one Horner pass and at most one log,
    instead of one power per series term.
    """
    t = np.asarray(t, dtype=float)
    first_exponent = n + order + 1
    exponents = first_exponent + 2 * np.arange(len(series_coeffs))

    d = np.zeros(len(series_coeffs))
    nonzero = exponents != 0
    d[nonzero] = series_coeffs[nonzero] / exponents[nonzero]

    result = t**first_exponent * horner(t * t, d)
    if not nonzero.all():                          # the logarithmic term
        k_log = int(np.flatnonzero(~nonzero)[0])
        result += series_coeffs[k_log] * np.log(t)
    return result


def series_definite_integral(lower, upper, n, order, series_coeffs):
    """∫_lower^upper t^n j_Q(t) dt from the power series (small upper only)."""
    return (series_antiderivative(upper, n, order, series_coeffs)
            - series_antiderivative(lower, n, order, series_coeffs))


# ---------------------------------------------------------------------------
# The kernel as a function of κ
# ---------------------------------------------------------------------------

def default_switch_point(order):
    """
    Argument t below which the power series is used instead of the closed form.
    Larger orders have stronger cancellation in the closed form, so switch later.
    """
    return float(min(max(1.5, order + 1.0), 8.0))


def default_series_terms(switch_point):
    """Enough series terms that truncation at t = switch_point is below 1e-17."""
    return int(20 + 4 * switch_point)


def radial_kernel(kap, n, order, a, b, switch_point=None, n_series=None):
    """
    F(κ) = κ^{-(Q+n+1)} ∫_{κa}^{κb} t^n j_Q(t) dt = ∫_a^b x^n j_Q(κx)/κ^Q dx,

    vectorized over κ >= 0. Three regimes, chosen point by point:
      small  (κ b <= t_s):     power series in κ², exact at κ = 0;
      large  (κ a >= t_s):     closed form G(κb) - G(κa);
      mixed  (κ a < t_s < κb): series from κa to t_s + closed form from t_s to κb.
    """
    if switch_point is None:
        switch_point = default_switch_point(order)
    if n_series is None:
        n_series = default_series_terms(switch_point)

    kap = np.asarray(kap, dtype=float)
    coeffs = antiderivative_coefficients(n, order)
    series_coeffs = bessel_series_coefficients(order, n_series)
    scale_power = order + n + 1          # F = κ^{-scale_power} × (t-integral)

    small = kap * b <= switch_point
    large = kap * a >= switch_point
    mixed = ~small & ~large

    out = np.empty(kap.shape)

    out[small] = series_kernel(kap[small], n, order, a, b, series_coeffs)

    k_large = kap[large]
    out[large] = (antiderivative(k_large * b, coeffs)
                  - antiderivative(k_large * a, coeffs)) / k_large**scale_power

    k_mixed = kap[mixed]
    near_zero_part = series_definite_integral(k_mixed * a, switch_point, n, order,
                                              series_coeffs)
    far_part = (antiderivative(k_mixed * b, coeffs)
                - antiderivative(switch_point, coeffs))
    out[mixed] = (near_zero_part + far_part) / k_mixed**scale_power

    return out


# ---------------------------------------------------------------------------
# Kernels on a (k1, k2, μ) mesh
# ---------------------------------------------------------------------------

def chebyshev_points(n_points, kind=2):
    """
    Chebyshev points ordered from +1 down to -1, the same convention as
    cheb_legendre_transform.chebyshev_points.
    """
    j = np.arange(n_points)
    if kind == 2:
        return np.cos(np.pi * j / (n_points - 1))
    if kind == 1:
        return np.cos(np.pi * (j + 0.5) / n_points)
    raise ValueError("kind must be 1 or 2")


def build_mesh(k1, k2, n_points, kind=2):
    """
    Broadcastable mesh for 1D arrays k1 (length n1) and k2 (length n2):

        K1 : (n1, 1, 1),   K2 : (1, n2, 1),   MU : (1, 1, M)

    Anything computed from them broadcasts to (n1, n2, M), with μ on the last
    axis as the transform expects.
    """
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    K1 = k1[:, None, None]
    K2 = k2[None, :, None]
    MU = chebyshev_points(n_points, kind)[None, None, :]
    return K1, K2, MU


def integral_kernel(k1, k2, mu, n, N, a, b, switch_point=None, n_series=None):
    """
    The function of μ to Legendre-transform, F(κ(k1, k2, μ)), broadcast over
    k1, k2, μ. With the mesh from `build_mesh` the result has shape (n1, n2, M).

    The same kernel serves N and -N (it depends only on Q = |N|);
    the sign of N only matters in `combine_legendre_integrals`.
    """
    if not 0 < a < b:
        raise ValueError("need 0 < a < b")
    k1, k2, mu = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (k1, k2, mu)))
    return radial_kernel(kappa(k1, k2, mu), n, abs(N), a, b, switch_point, n_series)


# ---------------------------------------------------------------------------
# From Legendre transforms to the integral
# ---------------------------------------------------------------------------

def binomial_weights(k1, k2, N):
    """
    Weights w_k and index shifts d_k such that

        I_l = 1/2 Σ_k w_k T_{l + d_k}.

    N >= 0:  w_k = C(Q,k) k2^{Q-k} (-k1)^k,   d_k = k
    N <  0:  w_k = C(Q,k) k1^{Q-k} (-k2)^k,   d_k = k - Q

    The argument with the higher Bessel order carries the positive powers.
    Returns a list of (w_k, d_k); each w_k broadcasts like k1 * k2.
    """
    Q = abs(N)
    if N >= 0:
        high, low, shift0 = k2, k1, 0
    else:
        high, low, shift0 = k1, k2, -Q
    return [(comb(Q, k) * high**(Q - k) * (-low)**k, shift0 + k) for k in range(Q + 1)]


def valid_l_range(n_points, N):
    """
    The l values for which I_l can be formed from T_0 ... T_{M-1}
    and both Bessel orders l and l + N are non-negative.
    """
    if N >= 0:
        return 0, n_points - 1 - N
    return -N, n_points - 1


def combine_legendre_integrals(T, k1, k2, N):
    """
    T  : (..., M) Legendre integrals T_l = ∫ F P_l dμ from the transform.
    k1, k2 : broadcastable to T.shape[:-1] + (1,) (e.g. K1, K2 from build_mesh).

    Returns I with the same shape as T; I[..., l] is the integral for that l,
    NaN where l is outside `valid_l_range`.
    """
    n_points = T.shape[-1]
    l_min, l_max = valid_l_range(n_points, N)

    I = np.full(np.broadcast_shapes(T.shape, np.shape(k1), np.shape(k2)), np.nan)
    if l_max < l_min:
        return I

    l = np.arange(l_min, l_max + 1)
    total = 0.0
    for weight, shift in binomial_weights(k1, k2, N):
        total = total + weight * T[..., l + shift]
    I[..., l] = 0.5 * total
    return I


def cancellation_factor(T, k1, k2, N):
    """
    Σ_k |w_k T_{l+d_k}|  /  |Σ_k w_k T_{l+d_k}|, same shape as the output of
    `combine_legendre_integrals`.

    The binomial sum is an alternating, finite-difference-like combination.
    When the terms are much larger than their sum (typically k1 ≈ k2 with
    large b and |N| >= 2), digits cancel: expect a relative error of about
    1e-16 times this factor. 1 means no cancellation.
    """
    n_points = T.shape[-1]
    l_min, l_max = valid_l_range(n_points, N)

    factor = np.full(np.broadcast_shapes(T.shape, np.shape(k1), np.shape(k2)), np.nan)
    if l_max < l_min:
        return factor

    l = np.arange(l_min, l_max + 1)
    signed_sum, absolute_sum = 0.0, 0.0
    for weight, shift in binomial_weights(k1, k2, N):
        term = weight * T[..., l + shift]
        signed_sum = signed_sum + term
        absolute_sum = absolute_sum + np.abs(term)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor[..., l] = absolute_sum / np.abs(signed_sum)
    return factor


# ---------------------------------------------------------------------------
# Convenience: the whole pipeline, and a resolution guide
# ---------------------------------------------------------------------------

def recommended_n_points(k1_max, k2_max, b, l_max, N=0, safety=40):
    """
    Heuristic number of Chebyshev points.

    With μ = cos θ the kernel oscillates in θ at most at the rate
    b * min(k1, k2), so about that many Chebyshev coefficients are needed to
    resolve it. On top of that, the transform must reach l_max + |N|.
    Rounded up to 2^p + 1 (efficient DCT-I lengths).
    """
    bandwidth = b * min(k1_max, k2_max)
    needed = int(np.ceil(max(1.25 * bandwidth, l_max + abs(N)))) + safety
    return int(2 ** np.ceil(np.log2(needed - 1)) + 1)


def bessel_product_integrals(k1, k2, n, N, a, b, transform,
                             switch_point=None, n_series=None):
    """
    I[i, j, l] = ∫_a^b x^n j_l(k1[i] x) j_{l+N}(k2[j] x) dx  for 1D arrays k1, k2.

    `transform` is a ChebyshevLegendreTransform or DenseChebyshevLegendreTransform
    from cheb_legendre_transform.py; its n_points and kind set the μ grid.
    Output shape (len(k1), len(k2), transform.n_points), NaN outside valid l.
    """
    K1, K2, MU = build_mesh(k1, k2, transform.n_points, transform.kind)
    values = integral_kernel(K1, K2, MU, n, N, a, b, switch_point, n_series)
    T = transform(values)
    return combine_legendre_integrals(T, K1, K2, N)


# ===========================================================================
# Piecewise-polynomial weights: many tables in one pass
# ===========================================================================
#
# Tables of the form
#
#     I[(name, u, N)][i, j, l] = ∫ S_name(x) x^u j_l(k_i x) j_{l+N}(q_j x) dx,
#
# where each S_name is a piecewise polynomial on the SAME interval edges
# e_0 < e_1 < ... < e_n. Five ideas avoid repeated work:
#
#   * Telescoping over intervals. For one power p = n + u and Q = |N|,
#
#         Σ_i c_i [H(κ e_{i+1}) - H(κ e_i)]  =  Σ_j (c_{j-1} - c_j) H(κ e_j),
#
#     (c_{-1} = c_last+1 = 0), so the antiderivative H is evaluated once per
#     edge instead of twice per interval.
#
#   * Special functions once per edge. H is a polynomial combination of
#     sin t, cos t, Si t and Ci t with t = κ e_j. Those four arrays are
#     computed once per edge and reused for every power, every Q, every
#     requested table and every weight function.
#
#   * One kernel per (weight, u, |N|). N and -N need the same kernel and the
#     same Legendre transform; only the final combination differs.
#
#   * Powers combined before evaluation. At one edge a kernel needs
#     Σ_n W_n h_{n+u,Q}(t) over the powers n of its weight. Every h has the
#     same form (sin t, cos t, Si t, Ci t and 1, each times a polynomial in
#     t and 1/t), so the weights are folded into the coefficients first and
#     each component is evaluated once per kernel and edge (`KernelPolynomials`).
#
#   * κ sorted once. Then t = κ e is sorted at every edge, the small-κ points
#     are a prefix, and the series / closed-form split at each edge is a
#     single index, so both branches work on slices (`EdgeArguments`).
#
# To keep everything well scaled, each edge contributes through
#
#     h(t) = H(t) / t^m,      m = p + Q + 1,
#
# because κ^{-m} H(κ e) = e^m h(κ e). H is one fixed antiderivative of
# t^p j_Q(t), continuous across the switch point t_s:
#
#     t <  t_s :  H = power series of j_Q integrated term by term
#     t >= t_s :  H = closed form G(t) + [S(t_s) - G(t_s)]
#
# Points where κ times the LARGEST edge is below t_s use the power series in
# κ² directly (no division by κ, exact at κ = 0).
# ===========================================================================

@dataclass(frozen=True)
class PiecewisePolynomial:
    """
    S(x) = Σ_n coeffs[i, n] x^n   on   edges[i] <= x <= edges[i+1].

    edges  : (n_intervals + 1,) increasing, edges[0] > 0
    coeffs : (n_intervals, degree + 1), column n multiplies x**n
    """
    edges: np.ndarray
    coeffs: np.ndarray

    def __post_init__(self):
        edges = np.asarray(self.edges, dtype=float)
        coeffs = np.atleast_2d(np.asarray(self.coeffs, dtype=float))
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be a 1D increasing array with at least 2 entries")
        if edges[0] <= 0:
            raise ValueError("edges[0] must be > 0")
        if coeffs.shape[0] != len(edges) - 1:
            raise ValueError("coeffs must have one row per interval")
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "coeffs", coeffs)

    @property
    def degree(self):
        return self.coeffs.shape[1] - 1

    @classmethod
    def fit(cls, func, edges, degree, n_nodes=200):
        """
        Least-squares polynomial fit of `func` on each interval, at Chebyshev
        points. The fit is done in a scaled variable (well conditioned) and
        then converted to plain powers of x.
        """
        edges = np.asarray(edges, dtype=float)
        coeffs = np.zeros((len(edges) - 1, degree + 1))
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            nodes = 0.5 * (b - a) * chebyshev_points(n_nodes, kind=2) + 0.5 * (b + a)
            power_series = np.polynomial.Polynomial.fit(nodes, func(nodes), degree).convert()
            coeffs[i, :len(power_series.coef)] = power_series.coef
        return cls(edges, coeffs)

    @classmethod
    def from_dict(cls, coeff_dict, edges):
        """From a {(n, interval): coefficient of x**n} dictionary."""
        degree = max(n for n, _ in coeff_dict)
        coeffs = np.zeros((len(edges) - 1, degree + 1))
        for (n, i), c in coeff_dict.items():
            coeffs[i, n] = c
        return cls(np.asarray(edges, dtype=float), coeffs)

    def __call__(self, x):
        """Evaluate S(x); zero outside [edges[0], edges[-1]]."""
        x = np.asarray(x, dtype=float)
        interval = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.coeffs) - 1)
        inside = (x >= self.edges[0]) & (x <= self.edges[-1])
        values = np.zeros(x.shape)
        for n in range(self.degree, -1, -1):          # Horner with per-point coefficients
            values = values * x + self.coeffs[interval, n]
        return np.where(inside, values, 0.0)

    def edge_weights(self):
        """
        w[j, n] = c_{j-1, n} - c_{j, n}: the weight of edge j in the telescoped sum
        (coefficients of the interval to its left minus those to its right).
        Shape (n_intervals + 1, degree + 1).
        """
        zero_row = np.zeros((1, self.degree + 1))
        left = np.vstack([zero_row, self.coeffs])
        right = np.vstack([self.coeffs, zero_row])
        return left - right


@dataclass(frozen=True)
class ScaledAntiderivative:
    """
    h(t) = H(t) / t^m for one power p and Bessel order Q, with m = p + Q + 1.
    See the section comment above for the definition of H.
    """
    power: int
    order: int
    m: int
    switch_point: float
    series_coeffs: np.ndarray       # d_k:  S(t) = t^m Σ_k d_k t^{2k}  (+ log term)
    log_coeff: float                # coefficient of ln t in S (0 if absent)
    min_power: int                  # lowest power of t in the sin/cos polynomials
    sin_coeffs: np.ndarray          # G = sin t Σ a_s t^s + cos t Σ b_s t^s + ...
    cos_coeffs: np.ndarray
    si_coeff: float                 # ... + si_coeff Si(t) + ci_coeff Ci(t)
    ci_coeff: float
    constant: float                 # S(t_s) - G(t_s), joins the two branches

    @classmethod
    def build(cls, power, order, switch_point, n_series):
        p, Q = power, order
        m = p + Q + 1

        # Series branch: S(t) = Σ_k c_k t^{m+2k} / (m+2k), with ln t where m+2k = 0.
        c = bessel_series_coefficients(Q, n_series)
        exponents = m + 2 * np.arange(n_series)
        series_coeffs = np.zeros(n_series)
        nonzero = exponents != 0
        series_coeffs[nonzero] = c[nonzero] / exponents[nonzero]
        log_coeff = float(c[~nonzero][0]) if not nonzero.all() else 0.0

        # Closed-form branch, written with real sin/cos polynomials.
        closed = antiderivative_coefficients(p, Q)
        g = np.array(closed.g, dtype=complex) if closed.g else np.zeros(1, dtype=complex)
        min_power = closed.powers[0] if closed.powers else 0
        quarter_turns = (Q + p) % 4                        # Θ = (Q + p) π / 2, exactly
        cos_theta = (1.0, 0.0, -1.0, 0.0)[quarter_turns]
        sin_theta = (0.0, 1.0, 0.0, -1.0)[quarter_turns]

        partial = cls(power=p, order=Q, m=m, switch_point=switch_point,
                      series_coeffs=series_coeffs, log_coeff=log_coeff,
                      min_power=min_power, sin_coeffs=g.real.copy(), cos_coeffs=g.imag.copy(),
                      si_coeff=closed.trig_integral_coeff * cos_theta,
                      ci_coeff=-closed.trig_integral_coeff * sin_theta,
                      constant=0.0)

        # Join the branches at t_s so that H is one continuous antiderivative.
        t_s = np.array([switch_point])
        si_s, ci_s = sici(t_s)
        G_s = partial._closed_form(t_s, np.sin(t_s), np.cos(t_s), si_s, ci_s)[0]
        S_s = partial._series_form(t_s)[0]
        return replace(partial, constant=S_s - G_s)

    @property
    def needs_trig_integrals(self):
        return self.si_coeff != 0.0 or self.ci_coeff != 0.0

    # -- the two branches of H (unscaled), used to join them -----------------

    def _series_form(self, t):
        H = t**self.m * horner(t * t, self.series_coeffs)
        if self.log_coeff:
            H += self.log_coeff * np.log(t)
        return H

    def _closed_form(self, t, sin_t, cos_t, si_t, ci_t):
        G = t**self.min_power * (sin_t * horner(t, self.sin_coeffs) + cos_t * horner(t, self.cos_coeffs))
        if self.si_coeff:
            G += self.si_coeff * si_t
        if self.ci_coeff:
            G += self.ci_coeff * ci_t
        return G

    # -- h(t) = H(t) / t^m, split into components ------------------------------

    def terms(self):
        """
        The components of h(t) = H(t) / t^m, each as {exponent of t: coefficient}:

            t >= t_s:  h = sin t · Σ_e sin[e] t^e + cos t · Σ_e cos[e] t^e
                           + Si(t) Σ_e si[e] t^e + Ci(t) Σ_e ci[e] t^e + Σ_e constant[e] t^e
            t <  t_s:  h = Σ_k series_coeffs[k] t^{2k} + ln t · Σ_e log[e] t^e

        (si, ci, constant and log each have the single exponent -m.)
        """
        shift = self.min_power - self.m
        return {
            "sin": {shift + i: c for i, c in enumerate(self.sin_coeffs) if c},
            "cos": {shift + i: c for i, c in enumerate(self.cos_coeffs) if c},
            "si": {-self.m: self.si_coeff} if self.si_coeff else {},
            "ci": {-self.m: self.ci_coeff} if self.ci_coeff else {},
            "constant": {-self.m: self.constant},
            "log": {-self.m: self.log_coeff} if self.log_coeff else {},
        }


def _coefficient_matrix(term_dicts, part, lowest, highest):
    """
    Coefficients of one component on the exponent grid lowest, ..., highest:
    row n holds term_dicts[n][part], column i the coefficient of t^(lowest + i).
    """
    matrix = np.zeros((len(term_dicts), highest - lowest + 1))
    for n, terms in enumerate(term_dicts):
        for exponent, coeff in terms[part].items():
            matrix[n, exponent - lowest] += coeff
    return matrix


def _polynomial(t, coeffs):
    """Σ_i coeffs[i] t^i (Horner), skipping trailing zero coefficients; None if all are zero."""
    nonzero = np.flatnonzero(coeffs)
    if len(nonzero) == 0:
        return None
    return horner(t, coeffs[:nonzero[-1] + 1])


@dataclass(frozen=True)
class KernelPolynomials:
    """
    The scaled antiderivatives h_n(t) of every power n of ONE kernel
    (weight, u, Q), with p = n + u, stored as coefficient matrices (one row
    per n). Each closed-form component (sin, cos, Si, Ci, constant) uses the
    same exponent grid, starting at t^lowest.

    One edge contributes Σ_n W_n h_n(t). Summing the coefficient rows first
    (W @ matrix) leaves a single polynomial per component, so the arrays are
    swept once per kernel and edge instead of once per power:

        t >= t_s:  t^lowest [ sin t A(t) + cos t B(t) + Si(t) C(t) + Ci(t) D(t) + E(t) ]
        t <  t_s:  P(t²) + ln t · t^log_lowest L(t)

    with A ... E, L ordinary polynomials in t and P a polynomial in t².
    """
    m: np.ndarray               # (n_powers,)  m_n = n + u + Q + 1
    lowest: int
    sin: np.ndarray             # (n_powers, n_exponents), all on the grid t^lowest, t^(lowest+1), ...
    cos: np.ndarray
    si: np.ndarray
    ci: np.ndarray
    constant: np.ndarray
    series: np.ndarray          # (n_powers, n_series): coefficients of t^{2k}
    log_lowest: int
    log: np.ndarray             # (n_powers, n_log_exponents)

    @classmethod
    def build(cls, antiderivatives):
        """From the ScaledAntiderivative of each power n = 0, 1, ..., in order."""
        term_dicts = [a.terms() for a in antiderivatives]
        closed_parts = ("sin", "cos", "si", "ci", "constant")

        # One common exponent grid for all closed-form components.
        closed_exponents = [e for terms in term_dicts for part in closed_parts for e in terms[part]]
        lowest, highest = min(closed_exponents), max(closed_exponents)
        grids = {part: _coefficient_matrix(term_dicts, part, lowest, highest) for part in closed_parts}

        # The log term of the series branch (absent in most cases).
        log_exponents = [e for terms in term_dicts for e in terms["log"]] or [0]
        log_lowest = min(log_exponents)
        log = _coefficient_matrix(term_dicts, "log", log_lowest, max(log_exponents))

        return cls(m=np.array([a.m for a in antiderivatives]), lowest=lowest,
                   sin=grids["sin"], cos=grids["cos"], si=grids["si"], ci=grids["ci"],
                   constant=grids["constant"],
                   series=np.vstack([a.series_coeffs for a in antiderivatives]),
                   log_lowest=log_lowest, log=log)

    def closed_branch(self, edge, weights):
        """Σ_n weights[n] h_n(t) on the closed-form points of one edge (t >= t_s)."""
        t = edge.t_closed
        total = np.zeros(t.shape)
        for values, coeffs in ((edge.sin, self.sin), (edge.cos, self.cos),
                               (edge.si, self.si), (edge.ci, self.ci)):
            poly = _polynomial(t, weights @ coeffs)
            if poly is not None:
                total += values * poly
        poly = _polynomial(t, weights @ self.constant)
        if poly is not None:
            total += poly
        return total * edge.closed_power(self.lowest)

    def series_branch(self, edge, weights):
        """Σ_n weights[n] h_n(t) on the series points of one edge (t < t_s)."""
        t = edge.t_series
        total = horner(t * t, weights @ self.series)
        log_poly = _polynomial(t, weights @ self.log)
        if log_poly is not None:
            total += np.log(t) * t**self.log_lowest * log_poly
        return total


@dataclass
class EdgeArguments:
    """
    t = κ e at one interval edge, for κ sorted in increasing order. Because t
    is then sorted too, the switch point splits it at a single index:
    t[:split] uses the series branch and t[split:] the closed form, both as
    plain slices. sin, cos (and Si, Ci if any kernel needs them) are evaluated
    once on the closed-form part; powers of t there are computed on demand
    and cached, since several kernels often need the same one.
    """
    t_series: np.ndarray
    t_closed: np.ndarray
    sin: np.ndarray
    cos: np.ndarray
    si: np.ndarray = None
    ci: np.ndarray = None
    _powers: dict = field(default_factory=dict)

    @classmethod
    def evaluate(cls, kap_sorted, edge, switch_point, need_trig_integrals):
        t = kap_sorted * edge
        split = int(np.searchsorted(t, switch_point, side="left"))   # t[split:] >= t_s
        t_closed = t[split:]
        si = ci = None
        if need_trig_integrals:
            si, ci = sici(t_closed)
        return cls(t_series=t[:split], t_closed=t_closed,
                   sin=np.sin(t_closed), cos=np.cos(t_closed), si=si, ci=ci)

    @property
    def split(self):
        return len(self.t_series)

    def closed_power(self, exponent):
        """t^exponent on the closed-form points, computed once per edge."""
        if exponent not in self._powers:
            self._powers[exponent] = self.t_closed ** exponent
        return self._powers[exponent]


def _check_shared_edges(weights):
    names = list(weights)
    edges = weights[names[0]].edges
    for name in names[1:]:
        if not np.array_equal(weights[name].edges, edges):
            raise ValueError(f"weight '{name}' has different edges from '{names[0]}'; "
                             "all piecewise polynomials must share the same intervals")
    return edges


def _small_kappa_kernel(kap, weight, u, Q, series_coeffs):
    """
    Σ_k c_k κ^{2k} ∫ S(x) x^{u+Q+2k} dx  — the kernel where κ e_last <= t_s.
    The moments ∫ S x^{...} are summed over intervals and powers as scalars first.

    With X = e_last and y = x / X this is a polynomial in (κX)² whose
    coefficients use only moments of y in (0, 1], so nothing overflows
    however many series terms are needed:

        c_k κ^{2k} ∫ x^{p+2k} dx  =  c_k (κX)^{2k} · X^{p+1} ∫ y^{p+2k} dy.
    """
    edges = weight.edges
    X = edges[-1]
    poly = np.zeros(len(series_coeffs))
    for k, c_k in enumerate(series_coeffs):
        moment = 0.0
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            for n in range(weight.degree + 1):
                if weight.coeffs[i, n] != 0.0:
                    p = n + u + Q
                    moment += weight.coeffs[i, n] * X**(p + 1) * power_integral(a / X, b / X, p + 2 * k)
        poly[k] = c_k * moment
    return horner((kap * X)**2, poly)


def piecewise_kernels(kap, weights, kernel_keys, switch_point=None, n_series=None):
    """
    Kernels for several weights and (u, |N|) pairs at once:

        F[(name, u, Q)](κ) = ∫ S_name(x) x^u j_Q(κ x) / κ^Q dx

    kap         : array of κ values (any shape), e.g. from `kappa` on a mesh.
    weights     : {name: PiecewisePolynomial}, all on the same edges.
    kernel_keys : iterable of (name, u, Q) with Q >= 0.
    Returns {(name, u, Q): array shaped like kap}.
    """
    kernel_keys = sorted(set(kernel_keys))
    edges = _check_shared_edges(weights)
    max_order = max(Q for _, _, Q in kernel_keys)
    if switch_point is None:
        switch_point = default_switch_point(max_order)
    if n_series is None:
        n_series = default_series_terms(switch_point)

    # Sort κ once. The small-κ points (κ e_last <= t_s) are then a prefix, and
    # at every edge the series / closed-form split is a single index.
    kap_flat = np.asarray(kap, dtype=float).ravel()
    order = np.argsort(kap_flat, kind="stable")
    kap_sorted = kap_flat[order]
    n_small = int(np.searchsorted(kap_sorted * edges[-1], switch_point, side="right"))
    kap_small, kap_rest = kap_sorted[:n_small], kap_sorted[n_small:]

    # Antiderivatives per distinct (power, Q), shared between kernels; then
    # each kernel's powers gathered into coefficient matrices.
    antiderivatives = {}

    def antiderivative(p, Q):
        if (p, Q) not in antiderivatives:
            antiderivatives[(p, Q)] = ScaledAntiderivative.build(p, Q, switch_point, n_series)
        return antiderivatives[(p, Q)]

    polynomials = {(name, u, Q): KernelPolynomials.build(
                       [antiderivative(n + u, Q) for n in range(weights[name].degree + 1)])
                   for name, u, Q in kernel_keys}
    need_trig_integrals = any(a.needs_trig_integrals for a in antiderivatives.values())
    edge_weights = {name: w.edge_weights() for name, w in weights.items()}

    # Telescoped sum over edges, on the points with κ e_last > t_s (sorted order).
    accumulators = {key: np.zeros(len(kap_rest)) for key in kernel_keys}
    for j, e in enumerate(edges):
        edge = EdgeArguments.evaluate(kap_rest, e, switch_point, need_trig_integrals)
        for key, poly in polynomials.items():
            weights_j = edge_weights[key[0]][j] * e**poly.m        # one weight per power n
            if not weights_j.any():
                continue
            accumulator = accumulators[key]
            accumulator[:edge.split] += poly.series_branch(edge, weights_j)
            accumulator[edge.split:] += poly.closed_branch(edge, weights_j)

    # Assemble: small-κ series first, telescoped sum after, then undo the sort.
    series_by_order = {}
    kernels = {}
    for key in kernel_keys:
        name, u, Q = key
        if Q not in series_by_order:
            series_by_order[Q] = bessel_series_coefficients(Q, n_series)
        values_sorted = np.empty(kap_flat.shape)
        values_sorted[:n_small] = _small_kappa_kernel(kap_small, weights[name], u, Q, series_by_order[Q])
        values_sorted[n_small:] = accumulators.pop(key)
        values = np.empty(kap_flat.shape)
        values[order] = values_sorted
        kernels[key] = values.reshape(np.shape(kap))
    return kernels


def compute_tables(ks, qs, weights, requests, transform, switch_point=None):
    """
    Integral tables for piecewise-polynomial weights, all in one pass:

        tables[(name, u, N)][i, j, l] = ∫ S_name(x) x^u j_l(ks[i] x) j_{l+N}(qs[j] x) dx

    ks, qs    : 1D arrays.
    weights   : {name: PiecewisePolynomial}, all on the same edges.
    requests  : iterable of (name, u, N); N may be negative.
    transform : ChebyshevLegendreTransform or DenseChebyshevLegendreTransform.

    Each array has shape (len(ks), len(qs), transform.n_points), NaN where l is
    outside the valid range for that N (see `valid_l_range`). Requests with
    the same (name, u, |N|) share one kernel and one Legendre transform.
    """
    requests = list(dict.fromkeys((name, int(u), int(N)) for name, u, N in requests))
    unknown = {name for name, _, _ in requests} - set(weights)
    if unknown:
        raise KeyError(f"no weight function named {sorted(unknown)}")

    K1, K2, MU = build_mesh(ks, qs, transform.n_points, transform.kind)
    kernel_keys = {(name, u, abs(N)) for name, u, N in requests}
    kernels = piecewise_kernels(kappa(K1, K2, MU), weights, kernel_keys, switch_point)

    tables = {}
    for key in sorted(kernel_keys):
        T = transform(kernels.pop(key))
        for name, u, N in requests:
            if (name, u, abs(N)) == key:
                tables[(name, u, N)] = combine_legendre_integrals(T, K1, K2, N)
    return {request: tables[request] for request in requests}


# ---------------------------------------------------------------------------
# Reference value by direct quadrature (slow; for testing)
# ---------------------------------------------------------------------------

def reference_integral(k1, k2, l, n, N, a, b):
    """∫_a^b x^n j_l(k1 x) j_{l+N}(k2 x) dx for scalar inputs, by adaptive quadrature."""
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    integrand = lambda x: x**n * spherical_jn(l, k1 * x) * spherical_jn(l + N, k2 * x)
    value, _ = quad(integrand, a, b, limit=2000, epsabs=1e-15, epsrel=1e-12)
    return value


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_antiderivative():
    """G(t1) - G(t0) against quadrature of t^n j_Q(t)."""
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    worst = 0.0
    for n in range(-5, 7):
        for Q in range(0, 6):
            coeffs = antiderivative_coefficients(n, Q)
            t0, t1 = 2.5, 17.0
            exact, _ = quad(lambda t: t**n * spherical_jn(Q, t), t0, t1, limit=500)
            got = antiderivative(t1, coeffs) - antiderivative(t0, coeffs)
            worst = max(worst, abs(got - exact) / max(1.0, abs(exact)))
    return worst


def _test_radial_kernel():
    """F(κ) in all three regimes, including κ = 0, against quadrature in x."""
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    worst = 0.0
    a, b = 0.3, 9.0
    for n in (-3, -1, 0, 2, 4):
        for Q in range(0, 5):
            kaps = np.array([0.0, 1e-6, 0.05, 0.4, 1.0, 3.0, 7.5])
            got = radial_kernel(kaps, n, Q, a, b)
            for kap, g in zip(kaps, got):
                if kap == 0.0:
                    f = lambda x: x**n * x**Q / double_factorial_odd(Q)
                else:
                    f = lambda x: x**n * spherical_jn(Q, kap * x) / kap**Q
                exact, _ = quad(f, a, b, limit=500, epsabs=1e-15, epsrel=1e-13)
                worst = max(worst, abs(g - exact) / max(1e-2, abs(exact)))
    return worst


def _test_pipeline(transform_class):
    """Full pipeline on a small (k1, k2) mesh, against direct quadrature."""
    rng = np.random.default_rng(1)
    k1 = np.array([0.2, 0.9, 1.7])
    k2 = np.array([0.9, 1.3])                  # includes k1 = k2 = 0.9
    a, b = 0.5, 12.0
    cases = [(2, 0), (0, 1), (1, -1), (2, 3), (-1, -2), (0, 4), (3, -4)]
    worst = 0.0
    for n, N in cases:
        M = recommended_n_points(k1.max(), k2.max(), b, l_max=30, N=N)
        transform = transform_class(M, kind=2)
        I = bessel_product_integrals(k1, k2, n, N, a, b, transform)
        l_min, _ = valid_l_range(M, N)
        errors, sizes = [], []
        for l in (l_min, l_min + 1, l_min + 7, 25):
            i, j = rng.integers(len(k1)), rng.integers(len(k2))
            exact = reference_integral(k1[i], k2[j], l, n, N, a, b)
            errors.append(abs(I[i, j, l] - exact))
            sizes.append(abs(exact))
        # error relative to the largest value in the case (individual values
        # can be tiny, where quadrature itself is the limiting factor)
        worst = max(worst, max(errors) / max(sizes))
    return worst


def _test_piecewise_tables(transform_class):
    """compute_tables with two piecewise-polynomial weights against quadrature."""
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    edges = np.linspace(0.5, 12.0, 5)
    weights = {"A": PiecewisePolynomial.fit(lambda x: np.exp(-(x - 5)**2 / 8), edges, 3),
               "B": PiecewisePolynomial.fit(lambda x: 1 / (1 + x), edges, 2)}
    ks, qs = np.array([0.3, 1.1]), np.array([0.2, 1.1, 1.6])
    requests = [("A", 2, 0), ("A", 2, 2), ("A", 2, -2), ("B", -1, 1), ("B", 0, -3)]
    M = recommended_n_points(ks.max(), qs.max(), edges[-1], l_max=30, N=3)
    tables = compute_tables(ks, qs, weights, requests, transform_class(M, kind=2))
    worst = 0.0
    for name, u, N in requests:
        errors, sizes = [], []
        for i, j, l in [(0, 0, 4), (1, 1, 6), (0, 2, 12), (1, 2, 25)]:
            integrand = lambda x: weights[name](x) * x**u * spherical_jn(l, ks[i] * x) * spherical_jn(l + N, qs[j] * x)
            exact = sum(quad(integrand, a, b, limit=500, epsabs=0, epsrel=1e-13)[0]
                        for a, b in zip(edges[:-1], edges[1:]))
            errors.append(abs(tables[(name, u, N)][i, j, l] - exact))
            sizes.append(abs(exact))
        worst = max(worst, max(errors) / max(sizes))
    return worst


if __name__ == "__main__":
    import sys
    import warnings
    from scipy.integrate import IntegrationWarning
    warnings.simplefilter("ignore", IntegrationWarning)   # reference quadrature noise

    print(f"antiderivative G, worst error          : {_test_antiderivative():.2e}")
    print(f"radial kernel F(κ), worst rel. error    : {_test_radial_kernel():.2e}"
          "  (limited by the scipy quad reference)")

    try:
        import cheb_legendre_transform as clt
    except ImportError:
        sys.exit("cheb_legendre_transform.py not on the path: skipping pipeline tests")

    for cls in (clt.DenseChebyshevLegendreTransform, clt.ChebyshevLegendreTransform):
        print(f"pipeline, {cls.__name__:33s}: max error / max |I| = {_test_pipeline(cls):.2e}")
    for cls in (clt.DenseChebyshevLegendreTransform, clt.ChebyshevLegendreTransform):
        print(f"piecewise tables, {cls.__name__:25s}: max error / max |I| = {_test_piecewise_tables(cls):.2e}")
