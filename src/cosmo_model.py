"""
cosmo_model.py
==============

Radial kernels of the form used in spherical Fourier-Bessel analyses, built
from the Bessel-product tables of bessel_product_kernels.py.

A kernel is a sum of terms,

    W(k, q, ℓ) = Σ_terms  scale · k^{k_power} · q^{q_power} · a(ℓ)
                          · ∫ S_weight(x) x^u  j_{ℓ+k_shift}(k x)  j_{ℓ+q_shift}(q x) dx ,

where S_weight is a piecewise polynomial (bpk.PiecewisePolynomial) that
absorbs every smooth function of x in the integrand (selection, growth,
bias, growth rate, ...). Each term reads one table,

    (weight, u, N = q_shift - k_shift)   at index   ℓ + k_shift,

so any kernel made of j_ℓ, j_ℓ', j_ℓ'' on either side, powers of x and
smooth functions of x reduces to tables the engine computes in one pass.

Derivatives are expanded without inverse powers of the argument:

    j_ℓ'  = [ ℓ j_{ℓ-1} - (ℓ+1) j_{ℓ+1} ] / (2ℓ+1)
    j_ℓ'' = α_ℓ j_{ℓ-2} - β_ℓ j_ℓ + γ_ℓ j_{ℓ+2}
            α_ℓ = ℓ(ℓ-1) / ((2ℓ-1)(2ℓ+1))
            β_ℓ = (2ℓ² + 2ℓ - 1) / ((2ℓ-1)(2ℓ+3))
            γ_ℓ = (ℓ+1)(ℓ+2) / ((2ℓ+1)(2ℓ+3))

Example: the monopole gSFB kernel with linear RSD (Wen et al. 2024, Eqs. 9-10)

    W^ℓ_ℓ(k, q) = sqrt(2/π) q ∫ x² j_ℓ(kx) [ S_b(x) j_ℓ(qx) - S_f(x) j_ℓ''(qx) ] dx,
    S_b = R D b_g,   S_f = R D f,

is `monopole_rsd_kernel()`. The Jacobian (velocity) term of Eq. 14,

    W^ℓ_ℓ(k, q) = -sqrt(2/π) ∫ x j_ℓ(kx) S_jac(x) j_ℓ'(qx) dx,   S_jac = R α f D,

is `jacobian_kernel()` (weight "jac"; build S_jac with
`jacobian_weight_function`). The radial velocity of McCarthy et al. 2025
(arXiv:2511.15701, Eq. 24, without its common 4π i^L),

    W^ℓ_ℓ(k, q) = sqrt(2/π) ∫ x² j_ℓ(kx) S_vel(x) j_ℓ'(qx) dx,   S_vel = R_v a H f D,

is `velocity_kernel()` (weight "vel"; `velocity_weight_function`). The
gravitational redshift of the sources (McDonald 2009, Yoo et al. 2009),

    W^ℓ_ℓ(k, q) = sqrt(2/π) ∫ x² j_ℓ(kx) S_grav(x) j_ℓ'(qx) dx,   S_grav = -(3/2) Ω_m a H D R / c,

is `gravitational_redshift_kernel()` (weight "grav";
`gravitational_redshift_weight_function`). It adds to the galaxy kernel:
`add_kernels(monopole_rsd_kernel(), gravitational_redshift_kernel())`.
Kernels add with `add_kernels`, e.g. the full
Newtonian field `add_kernels(monopole_rsd_kernel(), jacobian_kernel())`,
and any kernel works with the multipole functions below. A new kernel is a
new list of terms built with `bessel_product_terms`.

Typical use
-----------
    import bessel_product_kernels as bpk
    import cosmo_model as cm

    weights = {"b": bpk.PiecewisePolynomial.fit(S_b, edges, degree=5),
               "f": bpk.PiecewisePolynomial.fit(S_f, edges, degree=5),
               "jac": bpk.PiecewisePolynomial.fit(cm.jacobian_weight_function(R, f, D), edges, degree=7)}
    transform = DenseChebyshevLegendreTransform(m_pts)
    W = cm.compute_kernels([cm.monopole_rsd_kernel()], ks, qs, weights, transform)
    W["monopole_rsd"]          # array (len(ks), len(qs), m_pts)

Power-spectrum multipoles (Eq. 6), even and odd L, all tables in one pass:

    P = cm.compute_multipoles(cm.monopole_rsd_kernel(), [0, 1, 2, 3, 4], ks, qs, weights,
                              transform, q_weights, P_lin(qs), l_max, I22=I22)
    P[2]                       # real array over ks;  P[1], P[3] are purely imaginary

For a cross-spectrum (e.g. galaxies x radial velocities) pass the second
field's kernel as `cross_kernel`; `kernel` is the field carrying the shifted
Bessel order a in Eq. 6.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from sympy.physics.wigner import wigner_3j

import bessel_product_kernels as bpk


# ---------------------------------------------------------------------------
# ℓ-dependent coefficients of the derivative expansions
# ---------------------------------------------------------------------------

def _one(ell):
    return np.ones(np.shape(ell))


def _first_derivative_lower(ell):          # coefficient of j_{ℓ-1} in j_ℓ'
    return ell / (2 * ell + 1)


def _first_derivative_upper(ell):          # coefficient of j_{ℓ+1} in j_ℓ'
    return -(ell + 1) / (2 * ell + 1)


def _second_derivative_lower(ell):         # +α_ℓ, coefficient of j_{ℓ-2} in j_ℓ''
    return ell * (ell - 1) / ((2 * ell - 1) * (2 * ell + 1))


def _second_derivative_middle(ell):        # -β_ℓ, coefficient of j_ℓ in j_ℓ''
    return -(2 * ell**2 + 2 * ell - 1) / ((2 * ell - 1) * (2 * ell + 3))


def _second_derivative_upper(ell):         # +γ_ℓ, coefficient of j_{ℓ+2} in j_ℓ''
    return (ell + 1) * (ell + 2) / ((2 * ell + 1) * (2 * ell + 3))


DERIVATIVE_EXPANSIONS = {
    0: [(0, _one)],
    1: [(-1, _first_derivative_lower), (+1, _first_derivative_upper)],
    2: [(-2, _second_derivative_lower), (0, _second_derivative_middle), (+2, _second_derivative_upper)],
}
"""j_ℓ^{(d)}(z) = Σ a(ℓ) j_{ℓ+shift}(z) for each derivative order d: list of (shift, a)."""


def derivative_expansion(order):
    """[(shift, a(ℓ)), ...] with j_ℓ^{(order)} = Σ a(ℓ) j_{ℓ+shift}."""
    if order not in DERIVATIVE_EXPANSIONS:
        raise ValueError(f"derivative order {order} not available (have {sorted(DERIVATIVE_EXPANSIONS)})")
    return DERIVATIVE_EXPANSIONS[order]


# ---------------------------------------------------------------------------
# Terms and kernels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Term:
    """
    scale · k^{k_power} q^{q_power} · a(ℓ) · ∫ S_weight(x) x^u j_{ℓ+k_shift}(kx) j_{ℓ+q_shift}(qx) dx
    """
    weight: str
    x_power: int
    k_shift: int = 0
    q_shift: int = 0
    ell_coeff: Optional[Callable] = None     # a(ℓ); None means 1
    scale: float = 1.0
    k_power: int = 0
    q_power: int = 0

    @property
    def table_key(self):
        """The table this term reads: (weight, u, N)."""
        return (self.weight, self.x_power, self.q_shift - self.k_shift)

    def coefficient(self, ell):
        return _one(ell) if self.ell_coeff is None else self.ell_coeff(ell)

    def evaluate(self, tables, ks, qs):
        """This term on the (k, q, ℓ) grid of the tables."""
        table = tables[self.table_key]
        n_ell = table.shape[-1]
        ell = np.arange(n_ell)

        # Read the table at ℓ + k_shift; NaN where that index does not exist.
        source = ell + self.k_shift
        exists = (source >= 0) & (source < n_ell)
        values = np.full(table.shape, np.nan)
        values[..., exists] = table[..., source[exists]]

        # A zero coefficient removes the term even where the table is NaN
        # (e.g. α_ℓ = 0 at ℓ = 0, 1 where j_{ℓ-2} does not exist).
        a = self.coefficient(ell.astype(float))
        contribution = np.where(a == 0.0, 0.0, a * values)

        k_factor = np.asarray(ks, dtype=float)[:, None, None] ** self.k_power
        q_factor = np.asarray(qs, dtype=float)[None, :, None] ** self.q_power
        return self.scale * k_factor * q_factor * contribution


def _product(f, g):
    """a(ℓ) = f(ℓ) g(ℓ), keeping the identity coefficient cheap."""
    if f is _one:
        return g
    if g is _one:
        return f
    return lambda ell: f(ell) * g(ell)


def bessel_product_terms(weight, x_power, k_derivative=0, q_derivative=0,
                         scale=1.0, k_power=0, q_power=0):
    """
    Terms for  scale · k^{k_power} q^{q_power} ∫ S_weight(x) x^u j_ℓ^{(k_derivative)}(kx) j_ℓ^{(q_derivative)}(qx) dx,
    with the derivatives taken with respect to the Bessel argument.
    """
    terms = []
    for k_shift, a_k in derivative_expansion(k_derivative):
        for q_shift, a_q in derivative_expansion(q_derivative):
            coeff = _product(a_k, a_q)
            terms.append(Term(weight=weight, x_power=x_power, k_shift=k_shift, q_shift=q_shift,
                              ell_coeff=None if coeff is _one else coeff,
                              scale=scale, k_power=k_power, q_power=q_power))
    return terms


@dataclass(frozen=True)
class RadialKernel:
    """A named sum of terms; see the module docstring."""
    name: str
    terms: tuple

    def required_tables(self):
        """The (weight, u, N) tables this kernel needs."""
        return sorted({term.table_key for term in self.terms})

    def evaluate(self, tables, ks, qs):
        """W(k, q, ℓ) as an array (len(ks), len(qs), n_ell), NaN where undefined."""
        missing = [key for key in self.required_tables() if key not in tables]
        if missing:
            raise KeyError(f"kernel '{self.name}' needs tables {missing}")
        return sum(term.evaluate(tables, ks, qs) for term in self.terms)


# ---------------------------------------------------------------------------
# Kernels used so far
# ---------------------------------------------------------------------------

def monopole_rsd_kernel(bias_weight="b", growth_weight="f", name="monopole_rsd"):
    """
    W^ℓ_ℓ(k, q) = sqrt(2/π) q ∫ x² j_ℓ(kx) [ S_b(x) j_ℓ(qx) - S_f(x) j_ℓ''(qx) ] dx

    (Wen et al. 2024, Eqs. 9-10, diagonal a = ℓ), with S_b = R D b_g under
    `bias_weight` and S_f = R D f under `growth_weight`.
    """
    prefactor = np.sqrt(2.0 / np.pi)
    terms = (bessel_product_terms(bias_weight, x_power=2, scale=prefactor, q_power=1)
             + bessel_product_terms(growth_weight, x_power=2, q_derivative=2,
                                    scale=-prefactor, q_power=1))
    return RadialKernel(name=name, terms=tuple(terms))


def jacobian_kernel(jacobian_weight="jac", name="jac"):
    """
    Velocity term generated by the Jacobian of the RSD coordinate change
    (Wen et al. 2024, Eq. 14):

        Δ^jac_ℓ(q, x) = -(α(x)/x) (f(x)/q) D(x) j_ℓ'(qx),     α = ∂ ln(x² R) / ∂ ln x.

    Inserted in Eq. 9 (diagonal a = ℓ), the q of Eq. 9 cancels the 1/q and
    x²/x leaves one power of x:

        W^ℓ_ℓ(k, q) = -sqrt(2/π) ∫ x j_ℓ(kx) S_jac(x) j_ℓ'(qx) dx,
        S_jac = R α f D = f D (2R + x R')        (see `jacobian_weight_function`).

    Same API as `monopole_rsd_kernel`: works with shift_first_bessel,
    multipole_partner_kernel and compute_multipoles. To model the full
    Newtonian field, add it to the RSD kernel with `add_kernels`.
    """
    terms = bessel_product_terms(jacobian_weight, x_power=1, q_derivative=1,
                                 scale=-np.sqrt(2.0 / np.pi), q_power=0)
    return RadialKernel(name=name, terms=tuple(terms))


def velocity_kernel(velocity_weight="vel", name="vel"):
    """
    Radial peculiar velocity v_r = n̂·v (McCarthy et al. 2025, arXiv:2511.15701, Eq. 24):

        K^v_L(χ, k) = 4π i^L f H a / ((2L+1) k) [ L j_{L-1}(kχ) - (L+1) j_{L+1}(kχ) ]
                    = 4π i^L (a H f / k) j_L'(kχ).

    Notation mapping to Wen et al. (2024):
      * 4π i^L is the plane-wave factor shared by every field; it is already
        accounted for by the normalization of Eq. 8 and the i^{-a+b} of Eq. 6,
        so it is dropped here.
      * Their Eq. 23 uses the unequal-time P_lin(χ1, χ2, k); Eq. 8 uses the
        z = 0 spectrum, so the growth factor D(x) moves into the kernel.
    The angular kernel is therefore

        Δ^vel_ℓ(q, x) = a(x) H(x) f(x) D(x) j_ℓ'(qx) / q,

    and inserted in Eq. 9 (diagonal a = ℓ; the q of Eq. 9 cancels the 1/q):

        W^ℓ_ℓ(k, q) = sqrt(2/π) ∫ x² j_ℓ(kx) S_vel(x) j_ℓ'(qx) dx,
        S_vel = R_v a H f D          (see `velocity_weight_function`),

    with R_v the radial selection of the velocity sample. Sign convention:
    v_r = -a H f ∂_r ∇^{-2} δ, the same as Eqs. 10 and 14 of Wen et al.
    (Δ^jac = -α / (x a H) · Δ^vel). The units of v_r are those of a H.

    j_ℓ' is expanded as in Eq. 24, [ℓ j_{ℓ-1} - (ℓ+1) j_{ℓ+1}] / (2ℓ+1),
    with no inverse powers of qx (see DERIVATIVE_EXPANSIONS).
    """
    terms = bessel_product_terms(velocity_weight, x_power=2, q_derivative=1,
                                 scale=np.sqrt(2.0 / np.pi), q_power=0)
    return RadialKernel(name=name, terms=tuple(terms))


def velocity_weight_function(R, f=1.0, D=1.0, a=1.0, H=1.0):
    """
    S_vel(x) = R(x) a(x) H(x) f(x) D(x) for `velocity_kernel`.
    R is a callable of x; f, D, a, H may be callables or constants.
    """
    f, D, a, H = (_as_function(v) for v in (f, D, a, H))

    def S_vel(x):
        x = np.asarray(x, dtype=float)
        return R(x) * a(x) * H(x) * f(x) * D(x)
    return S_vel


SPEED_OF_LIGHT_KM_S = 299792.458


def gravitational_redshift_kernel(grav_weight="grav", name="grav"):
    """
    Gravitational redshift of the sources (McDonald 2009, arXiv:0907.5220;
    the -A term of the redshift in Yoo, Fitzpatrick & Zaldarriaga 2009, arXiv:0907.0707).

    The potential at the source shifts the observed redshift, hence the radial
    position, by Δχ = -ψ / (a H c) (McDonald Eq. 2). Like the velocity term
    (-∂_r of v_r / aH) this adds  +∂_r ψ / (a H c)  to the number counts. With
    Poisson, ψ(q) = -(3/2) Ω_m(a) a² H² δ(q) / q², and in the notation of
    Wen et al. (Eq. 9, D(x) explicit):

        Δ^grav_ℓ(q, x) = -(3/2) Ω_m(x) (a H / c) D(x) j_ℓ'(qx) / q
                       = -(3 Ω_m / (2 f c)) Δ^vel_ℓ(q, x),

        W^ℓ_ℓ(k, q) = sqrt(2/π) ∫ x² j_ℓ(kx) S_grav(x) j_ℓ'(qx) dx,
        S_grav = -(3/2) Ω_m a H D R / c          (see `gravitational_redshift_weight_function`).

    In the plane-parallel limit this is δ_grav(k) = -i (3/2) Ω_m (aH / ck) μ δ(k)
    in our Fourier convention. McDonald's Eq. 9 has +i because his convention
    is the complex conjugate of ours (his v_∥ = -i f μ / k_c δ; ours is +i).

    Only the leading term, of order (aH / ck), is included. Its full-sky
    companion α ψ / (x a H c) (the analogue of the Jacobian term) and the
    non-derivative potential terms of order (aH / ck)² are not.
    """
    terms = bessel_product_terms(grav_weight, x_power=2, q_derivative=1,
                                 scale=np.sqrt(2.0 / np.pi), q_power=0)
    return RadialKernel(name=name, terms=tuple(terms))


def gravitational_redshift_weight_function(R, Omega_m, D=1.0, a=1.0, H=1.0, c=SPEED_OF_LIGHT_KM_S):
    """
    S_grav(x) = -(3/2) Ω_m(x) a(x) H(x) D(x) R(x) / c  for `gravitational_redshift_kernel`.

    Ω_m is the time-dependent Ω_m(a) at distance x. R is a callable; Ω_m, D,
    a, H may be callables or constants. H / c must have units of inverse
    distance in the units of x (e.g. H in km/s/(Mpc/h) with c in km/s and
    x in Mpc/h).
    """
    Omega_m, D, a, H = (_as_function(v) for v in (Omega_m, D, a, H))

    def S_grav(x):
        x = np.asarray(x, dtype=float)
        return -1.5 * Omega_m(x) * a(x) * H(x) * D(x) * R(x) / c
    return S_grav


def add_kernels(*kernels, name=None):
    """Sum of kernels as one RadialKernel (e.g. RSD + Jacobian for the full field)."""
    terms = tuple(term for kernel in kernels for term in kernel.terms)
    return RadialKernel(name=name or "+".join(kernel.name for kernel in kernels), terms=terms)


def _as_function(value):
    """A callable of x from a callable or a constant."""
    return value if callable(value) else (lambda x: np.full(np.shape(x), float(value)))


def jacobian_weight_function(R, f=1.0, D=1.0, dR_dx=None, relative_step=1e-5):
    """
    S_jac(x) = R(x) α(x) f(x) D(x) = f(x) D(x) [2 R(x) + x R'(x)],
    with α = ∂ ln(x² R) / ∂ ln x as in Eq. 14.

    R, f, D  : callables of x (f and D may also be constants).
    dR_dx    : R'(x) if known; otherwise a central finite difference is used
               (relative step 1e-5, error ~1e-10 relative for smooth R).
    Returns a vectorized callable, ready for bpk.PiecewisePolynomial.fit.
    """
    f, D = _as_function(f), _as_function(D)
    if dR_dx is None:
        def dR_dx(x):
            h = relative_step * np.abs(x)
            return (R(x + h) - R(x - h)) / (2 * h)

    def S_jac(x):
        x = np.asarray(x, dtype=float)
        return f(x) * D(x) * (2 * R(x) + x * dR_dx(x))
    return S_jac


# ---------------------------------------------------------------------------
# Power-spectrum multipoles (Wen et al. 2024, Eq. 6)
# ---------------------------------------------------------------------------
#
# Eq. 6 with b = ℓ and a = ℓ + s:
#
#   P_L(k) = 4π(2L+1)/I22 Σ_ℓ (2ℓ+1) ∫ dq P(q) W^ℓ_ℓ[Y](k,q) V^L_ℓ[X](k,q)
#
#   V^L_ℓ = Σ_s i^{-s} (2ℓ+2s+1) (ℓ+s L ℓ; 0 0 0)² W^{ℓ+s}_ℓ ,
#
# where W^{ℓ+s}_ℓ is the kernel with its first (k-side) Bessel order shifted
# by s (`shift_first_bessel`), and the 3j symbol restricts s to
# -L, -L+2, ..., L. For an auto-spectrum X = Y. For a cross-spectrum the
# field X carries the shifted index a (the first W in Eq. 8).
#
# The phase i^{-s} is 1 or -1 for even L. For odd L it is imaginary:
# i^{-s} = -i · i^{1-s}, with i^{1-s} = ±1. V^L is kept real and the common
# phase (1 for even L, -i for odd L) is applied to P_L at the end.
#
# The prefactor 4π(2L+1) (instead of the paper's (4π)²(2L+1)) normalizes
# P_L so that P_0 -> P(k) for a large window; it was checked against the
# Kaiser limit for L = 0, 2, 4.

@lru_cache(maxsize=None)
def three_j_squared(a, L, b):
    """(a L b; 0 0 0)² from sympy (exact, cached). Zero if any order is negative."""
    if a < 0 or b < 0:
        return 0.0
    return float(wigner_3j(a, L, b, 0, 0, 0) ** 2)


def multipole_phase(L):
    """Common phase of V^L: 1 for even L, -i for odd L."""
    return 1.0 if L % 2 == 0 else -1j


def multipole_shifts(L):
    """The shifts s = a - ℓ allowed by the 3j symbol: -L, -L+2, ..., L."""
    return list(range(-L, L + 1, 2))


@dataclass(frozen=True)
class MultipoleCoefficient:
    """
    c_L(ℓ, s) = i^{-s} / phase(L) · (2ℓ+2s+1) (ℓ+s L ℓ; 0 0 0)²   (real),
    zero wherever ℓ + s < 0 or the 3j symbol vanishes.
    """
    L: int
    s: int

    def __call__(self, ell):
        ell = np.asarray(ell).astype(int)
        sign = round((1j ** (-self.s) / multipole_phase(self.L)).real)   # exactly ±1
        values = [three_j_squared(int(l) + self.s, self.L, int(l)) * (2 * (l + self.s) + 1)
                  for l in ell.ravel()]
        return sign * np.array(values, dtype=float).reshape(ell.shape)


def shift_first_bessel(kernel, s, name=None):
    """
    The same kernel with the first Bessel function j_{ℓ+k_shift}(kx) replaced
    by j_{ℓ+k_shift+s}(kx): W^ℓ_ℓ -> W^{ℓ+s}_ℓ. Works for any RadialKernel.
    """
    terms = tuple(replace(term, k_shift=term.k_shift + s) for term in kernel.terms)
    return RadialKernel(name=name or f"{kernel.name}_a{s:+d}", terms=terms)


def multipole_partner_kernel(kernel, L, name=None):
    """
    V^L_ℓ = Σ_s c_L(ℓ, s) W^{ℓ+s}_ℓ as one RadialKernel (real; see multipole_phase).
    For L = 0 it equals `kernel` itself.
    """
    terms = []
    for s in multipole_shifts(L):
        c = MultipoleCoefficient(L, s)
        for term in shift_first_bessel(kernel, s).terms:
            a = term.ell_coeff
            terms.append(replace(term, ell_coeff=c if a is None else _product(a, c)))
    return RadialKernel(name=name or f"{kernel.name}_V{L}", terms=tuple(terms))


def power_spectrum_multipole(W_diagonal, V, q_weights, P_q, L, l_max, I22=None):
    """
    P_L(k) = phase(L) · 4π(2L+1) Σ_{ℓ<=l_max} (2ℓ+1) Σ_q w_q P(q_q) W[k,q,ℓ] V^L[k,q,ℓ]  (/ I22)

    W_diagonal : W^ℓ_ℓ of field Y on the (k, q, ℓ) grid
    V          : V^L_ℓ of field X (same grid), from multipole_partner_kernel
    q_weights, P_q : quadrature weights and P(q) at the same q values
    Returns a real array for even L and a purely imaginary one for odd L.
    """
    W_used, V_used = W_diagonal[..., :l_max + 1], V[..., :l_max + 1]
    if np.isnan(W_used).any() or np.isnan(V_used).any():
        raise ValueError("NaN in the kernels below l_max: increase m_pts to at least l_max + L + 3")
    ell = np.arange(l_max + 1)
    total = np.einsum("q,kql,kql,l->k", q_weights * P_q, W_used, V_used, 2 * ell + 1)
    result = multipole_phase(L) * 4 * np.pi * (2 * L + 1) * total
    return result / I22 if I22 is not None else result


def compute_multipoles(kernel, multipoles, ks, qs, weights, transform, q_weights, P_q,
                       l_max, I22=None, cross_kernel=None, tables_function=None):
    """
    P_L(k) for every L in `multipoles`, with all tables computed in one pass.

    kernel       : W^ℓ_ℓ of the field carrying the shifted index a (X): the field
                   weighted by L_L(k̂·x̂) in the Yamamoto estimator, F_L[X](k) F_0[Y](-k).
    cross_kernel : W^ℓ_ℓ of the second field (Y); None for an auto-spectrum.
    tables_function : optional table backend, as in compute_kernels.
    Returns {L: array over ks}.

    Sign check (plane-parallel limit, kernel = galaxies, cross_kernel = velocity):
    P_1 -> -i (aHf/k)(b + 3f/5) P(k); swapping the two fields flips the sign.
    """
    diagonal = cross_kernel or kernel
    partners = {L: multipole_partner_kernel(kernel, L) for L in multipoles}
    W = compute_kernels([diagonal, *partners.values()], ks, qs, weights, transform,
                        tables_function=tables_function)
    return {L: power_spectrum_multipole(W[diagonal.name], W[partner.name], q_weights, P_q,
                                        L, l_max, I22)
            for L, partner in partners.items()}


# ---------------------------------------------------------------------------
# One call: tables for all kernels, then the kernels
# ---------------------------------------------------------------------------

def compute_kernels(kernels, ks, qs, weights, transform, return_tables=False, tables_function=None):
    """
    Evaluate several RadialKernel objects on the (ks, qs, ℓ) grid.

    All tables needed by all kernels are computed in a single pass, so shared
    weights, powers, edges and |N| are reused.

    tables_function : optional callable (ks, qs, weights, requests) -> tables,
                      used instead of bpk.compute_tables (then `transform` is
                      ignored and may be None). E.g. a
                      gauss_legendre_tables.GaussLegendreTables built for ks, qs.

    Returns {kernel.name: array (len(ks), len(qs), n_ell)},
    and the tables as a second value if return_tables is True.
    """
    requests = sorted({key for kernel in kernels for key in kernel.required_tables()})
    if tables_function is None:
        tables = bpk.compute_tables(ks, qs, weights, requests, transform)
    else:
        tables = tables_function(ks, qs, weights, requests)
    results = {kernel.name: kernel.evaluate(tables, ks, qs) for kernel in kernels}
    return (results, tables) if return_tables else results
