"""
cheb_legendre_transform.py
==========================

Batched Legendre integral transform of functions sampled at Chebyshev points:

    I_l  =  ∫_{-1}^{1} f(t) P_l(t) dt ,        l = 0, 1, ..., M-1

where M is the number of Chebyshev points. (No (2l+1)/2 prefactor: if you want
the Legendre-series coefficients c_l, multiply the output by (l + 1/2).)

Method
------
1.  Values -> Chebyshev coefficients a_k, with f(t) ≈ Σ_k a_k T_k(t), using a
    batched DCT (scipy.fft.dct).

2.  Chebyshev coefficients -> Legendre integrals with the exact connection
    matrix
            I_l = Σ_k  J_{lk} a_k ,          J_{lk} = ∫ T_k(t) P_l(t) dt .

    J is upper triangular, and J_{lk} = 0 unless k - l is even, so it splits
    into an even block (l, k even) and an odd block (l, k odd). Inside a block,
    write l = 2i + p, k = 2j + p (p = parity). Then, for j > i,

            J_{lk} = -k * t(j - i) * h(i + j + p)

        Toeplitz part   t(d) = Λ(d - 1) / (2d)                (d >= 1)
        Hankel part     h(s) = Λ(s - 1/2) / (2s + 1)          (s >= 1)
        with            Λ(z) = Γ(z + 1/2) / Γ(z + 1)

    and the diagonal is  J_{00} = 2,  J_{kk} = sqrt(pi) / ((2k + 1) Λ(k)).

    The Hankel matrix H_{ij} = h(i + j + p) is numerically low rank, so a
    truncated SVD gives  H ≈ Σ_r σ_r u_r v_rᵀ  and therefore

            (T ∘ H) x  ≈  Σ_r  diag(σ_r u_r) · T · diag(v_r) · x ,

    where T is the upper-triangular Toeplitz matrix. Every product with T is a
    convolution, done with zero-padded real FFTs, all rank terms and all
    functions in one batched call.

Cost per function after precomputation: one DCT plus about 4R real FFTs of
length ~M (R = Hankel rank, typically 20-40 at double precision).

Dense alternative
-----------------
DenseChebyshevLegendreTransform has the same interface and output, but builds
the exact parity blocks of J from rational recursions and applies them with a
triangular BLAS multiply (dtrmm): about M²/4 multiply-adds per function and
two (M/2 x M/2) matrices in memory. It is exact up to rounding and is usually
faster until M is large or the matrices no longer fit in memory.

Chebyshev point conventions (the order matters: both go from +1 down to -1)
-------------------------------------------------------------------------
    kind=2 (extrema / Lobatto, includes ±1):  x_j = cos(pi * j / (M - 1))
    kind=1 (roots, interior only):            x_j = cos(pi * (j + 1/2) / M)
If your samples are ordered from -1 to +1, flip the last axis first.
"""

import numpy as np
from scipy import fft as sfft
from scipy.linalg import blas


# ---------------------------------------------------------------------------
# Chebyshev points
# ---------------------------------------------------------------------------

def chebyshev_points(n_points, kind=2):
    """Chebyshev points in the order this module expects (from +1 to -1)."""
    j = np.arange(n_points)
    if kind == 2:
        return np.cos(np.pi * j / (n_points - 1))
    if kind == 1:
        return np.cos(np.pi * (j + 0.5) / n_points)
    raise ValueError("kind must be 1 or 2")


def chebyshev_coefficients(values, kind=2):
    """
    a_k with f(t) ≈ Σ_k a_k T_k(t), from samples at Chebyshev points of the
    given kind. The batched DCT acts along the last axis.
    """
    m = values.shape[-1]
    if kind == 2:
        # DCT-I: y_k = f_0 + (-1)^k f_{M-1} + 2 Σ_{j=1}^{M-2} f_j cos(pi j k / (M-1))
        a = sfft.dct(values, type=1, axis=-1, workers=-1) / (m - 1)
        a[..., 0] /= 2.0
        a[..., -1] /= 2.0
    elif kind == 1:
        # DCT-II: y_k = 2 Σ_j f_j cos(pi k (j + 1/2) / M)
        a = sfft.dct(values, type=2, axis=-1, workers=-1) / m
        a[..., 0] /= 2.0
    else:
        raise ValueError("kind must be 1 or 2")
    return a


# ---------------------------------------------------------------------------
# Lambda function, by recursion
# ---------------------------------------------------------------------------

def lambda_at_integers(n_max):
    """
    Λ(n) = Γ(n + 1/2) / Γ(n + 1) for n = 0, 1, ..., n_max.

    Uses the recursion  Λ(0) = sqrt(pi),  Λ(n + 1) = Λ(n) * (n + 1/2) / (n + 1),
    which never forms the (overflowing) gamma functions themselves.
    """
    n = np.arange(n_max)
    ratios = (n + 0.5) / (n + 1.0)
    return np.sqrt(np.pi) * np.concatenate(([1.0], np.cumprod(ratios)))


def lambda_at_half_integers(lam_int):
    """
    Λ(n + 1/2) for n = 0, 1, ..., len(lam_int) - 1, from the identity
    Λ(z) Λ(z - 1/2) = 1/z, i.e.  Λ(n + 1/2) = 1 / ((n + 1/2) Λ(n)).
    """
    n = np.arange(len(lam_int))
    return 1.0 / ((n + 0.5) * lam_int)


# ---------------------------------------------------------------------------
# Pieces of the connection matrix J_{lk} = ∫ T_k P_l dt, for one parity block
# ---------------------------------------------------------------------------

def toeplitz_symbol(n, lam_int):
    """t(d) for d = 0, ..., n-1:  t(0) = 0,  t(d) = Λ(d - 1) / (2d)."""
    t = np.zeros(n)
    d = np.arange(1, n)
    t[1:] = lam_int[d - 1] / (2.0 * d)
    return t


def hankel_matrix(n, parity, lam_int):
    """
    H_{ij} = h(i + j + parity), i, j = 0..n-1, with h(s) = Λ(s - 1/2) / (2s + 1).

    h(0) is infinite, but it only appears at (i, j) = (0, 0) in the even block,
    where the Toeplitz factor is zero anyway, so we set it to 0.
    """
    s = np.arange(2 * n - 1) + parity           # all values of i + j + parity
    lam_half = lambda_at_half_integers(lam_int)  # lam_half[m] = Λ(m + 1/2)
    h = np.zeros(len(s))
    pos = s >= 1
    h[pos] = lam_half[s[pos] - 1] / (2.0 * s[pos] + 1.0)   # Λ(s - 1/2) = lam_half[s-1]
    i = np.arange(n)
    return h[i[:, None] + i[None, :]]


def diagonal_entries(k, lam_int):
    """J_{kk} = ∫ T_k P_k dt:  2 for k = 0,  sqrt(pi) / ((2k + 1) Λ(k)) otherwise."""
    diag = np.sqrt(np.pi) / ((2.0 * k + 1.0) * lam_int[k])
    diag[k == 0] = 2.0
    return diag


def low_rank_svd(matrix, tol):
    """
    Truncated SVD:  matrix ≈ left @ right, keeping singular values above
    tol * (largest singular value). Returns left (n, R) and right (R, n),
    with the singular values folded into `left`.
    """
    u, sigma, vh = np.linalg.svd(matrix)
    rank = int(np.sum(sigma > tol * sigma[0]))
    return u[:, :rank] * sigma[:rank], vh[:rank, :]


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------

class ChebyshevLegendreTransform:
    """
    Precomputes everything for a fixed number of Chebyshev points, then maps
    arrays of sampled functions to their Legendre integrals.

        transform = ChebyshevLegendreTransform(n_points=1025)
        integrals = transform(values)    # values: (..., 1025) -> (..., 1025)

    Parameters
    ----------
    n_points   : number of Chebyshev points M (last axis of the input).
    kind       : 1 or 2, which Chebyshev points the samples are on.
    tol        : relative SVD truncation tolerance for the Hankel factor.
    batch_size : functions processed at once; bounds memory, which is about
                 batch_size * rank * 2 * M complex numbers.
    """

    def __init__(self, n_points, kind=2, tol=1e-15, batch_size=256):
        if kind not in (1, 2):
            raise ValueError("kind must be 1 or 2")
        self.n_points = n_points
        self.kind = kind
        self.batch_size = batch_size

        # Λ(n) is needed up to n ≈ M (Hankel and diagonal entries).
        lam_int = lambda_at_integers(n_points + 1)

        self.blocks = [self._build_parity_block(p, lam_int, tol) for p in (0, 1)]

    # -- precomputation ------------------------------------------------------

    def _build_parity_block(self, parity, lam_int, tol):
        k = np.arange(parity, self.n_points, 2)    # Chebyshev degrees in this block
        n = len(k)

        # Linear (not circular) convolution needs FFT length >= 2n - 1.
        nfft = sfft.next_fast_len(2 * n - 1, real=True)
        t_hat = sfft.rfft(toeplitz_symbol(n, lam_int), n=nfft)

        left, right = low_rank_svd(hankel_matrix(n, parity, lam_int), tol)

        return {
            "parity": parity,
            "n": n,
            "nfft": nfft,
            "minus_k": -k.astype(float),          # column scaling, -k
            "diag": diagonal_entries(k, lam_int),
            "t_hat": t_hat,                       # FFT of the Toeplitz symbol
            "left": left.T.copy(),                # (R, n): σ_r u_r
            "right": right,                       # (R, n): v_r
        }

    @property
    def hankel_ranks(self):
        """SVD ranks kept for the (even, odd) blocks."""
        return tuple(b["left"].shape[0] for b in self.blocks)

    # -- step 1: values -> Chebyshev coefficients -----------------------------

    def chebyshev_coefficients(self, values):
        """a_k with f(t) ≈ Σ_k a_k T_k(t); the DCT acts along the last axis."""
        return chebyshev_coefficients(values, self.kind)

    # -- step 2: Chebyshev coefficients -> Legendre integrals ----------------

    @staticmethod
    def _upper_toeplitz_product(x, t_hat, n, nfft):
        """
        y_i = Σ_{j > i} t(j - i) x_j along the last axis of x (any leading shape).

        This is a correlation; reversing x turns it into a convolution, which
        is done with a zero-padded real FFT.
        """
        x_reversed = x[..., ::-1]
        x_hat = sfft.rfft(x_reversed, n=nfft, axis=-1, workers=-1)
        y_reversed = sfft.irfft(x_hat * t_hat, n=nfft, axis=-1, workers=-1)[..., :n]
        return y_reversed[..., ::-1]

    def _apply_parity_block(self, a_block, block):
        """J_block @ a_block for a batch: a_block has shape (B, n)."""
        w = a_block * block["minus_k"]                           # (B, n)

        # One Toeplitz product per SVD term, all done in a single FFT call.
        x = w[:, None, :] * block["right"][None, :, :]           # (B, R, n)
        y = self._upper_toeplitz_product(x, block["t_hat"], block["n"], block["nfft"])
        off_diagonal = np.einsum("brn,rn->bn", y, block["left"])  # Σ_r σ_r u_r ⊙ (...)

        return off_diagonal + block["diag"] * a_block

    def chebyshev_to_legendre_integrals(self, coeffs):
        """(B, M) Chebyshev coefficients -> (B, M) integrals ∫ f P_l dt."""
        out = np.empty_like(coeffs)
        for block in self.blocks:
            p = block["parity"]
            out[:, p::2] = self._apply_parity_block(coeffs[:, p::2], block)
        return out

    # -- full transform -------------------------------------------------------

    def __call__(self, values):
        """
        values : array (..., M), functions sampled at the Chebyshev points.
        returns: array (..., M), entry [..., l] = ∫_{-1}^{1} f(t) P_l(t) dt.
        """
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != self.n_points:
            raise ValueError(f"last axis must have length {self.n_points}")

        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self.n_points)
        out = np.empty_like(flat)

        for start in range(0, flat.shape[0], self.batch_size):
            chunk = slice(start, start + self.batch_size)
            coeffs = self.chebyshev_coefficients(flat[chunk])
            out[chunk] = self.chebyshev_to_legendre_integrals(coeffs)

        return out.reshape(*leading_shape, self.n_points)


# ---------------------------------------------------------------------------
# Dense alternative: exact matrix, applied with a triangular BLAS multiply
# ---------------------------------------------------------------------------

def legendre_integral_diagonal(n_points):
    """
    J_{ll} = ∫ T_l P_l dt for l = 0..n_points-1, by the recursion
    J_00 = 2,  J_11 = 2/3,  J_{l+1,l+1} = J_{ll} (2l + 2) / (2l + 3)  (l >= 1).
    """
    diag = np.empty(n_points)
    diag[0] = 2.0
    if n_points > 1:
        diag[1] = 2.0 / 3.0
        l = np.arange(1, n_points - 1)
        diag[2:] = diag[1] * np.cumprod((2.0 * l + 2.0) / (2.0 * l + 3.0))
    return diag


def legendre_integral_matrix_block(n_points, parity):
    """
    Exact parity block of J_{lk} = ∫ T_k P_l dt (rows: Legendre degree l,
    columns: Chebyshev degree k, both = 2i + parity). Upper triangular.

    Built column by column from the rational recursions (no gamma functions):
        diagonal          J_{ll}  (see legendre_integral_diagonal)
        first off-diag.   J_{l,l+2} = -J_{ll} (l + 2) / (2l + 3)   (l >= 1),
                          J_{0,2}   = -2/3
        along a row       J_{l,k+2} = J_{lk} (k+2)(k+l)(k-l-1) / (k (k+l+3)(k-l+2))
    """
    degrees = np.arange(parity, n_points, 2)
    n = len(degrees)
    block = np.zeros((n, n))

    diag = legendre_integral_diagonal(n_points)[degrees]
    block[np.arange(n), np.arange(n)] = diag

    if n > 1:
        l = degrees[:-1].astype(float)
        first_off = -diag[:-1] * (l + 2.0) / (2.0 * l + 3.0)
        if parity == 0:
            first_off[0] = -2.0 / 3.0
        block[np.arange(n - 1), np.arange(1, n)] = first_off

    # Fill columns j >= 2 from column j - 1, for all rows at least two above
    # the diagonal at once (k = degree of column j - 1).
    for j in range(2, n):
        rows = slice(0, j - 1)
        l = degrees[rows].astype(float)
        k = float(degrees[j - 1])
        ratio = (k + 2) * (k + l) * (k - l - 1) / (k * (k + l + 3) * (k - l + 2))
        block[rows, j] = block[rows, j - 1] * ratio

    return block


class DenseChebyshevLegendreTransform:
    """
    Same interface and output as ChebyshevLegendreTransform, but applies the
    exact connection matrix J directly:

        values --DCT--> a  -->  I = J a ,   done as two parity blocks.

    Each block is upper triangular, so it is applied with the BLAS routine
    dtrmm, which skips the zero half. Cost per function: about M²/4
    multiply-adds, plus the DCT. Memory: two (M/2 x M/2) matrices.
    Exact apart from rounding; usually the faster choice up to M of several
    thousand. Benchmark against the FFT version on your machine.
    """

    def __init__(self, n_points, kind=2, batch_size=4096):
        if kind not in (1, 2):
            raise ValueError("kind must be 1 or 2")
        self.n_points = n_points
        self.kind = kind
        self.batch_size = batch_size
        # Fortran order is what BLAS expects, so no copies at call time.
        self.blocks = [np.asfortranarray(legendre_integral_matrix_block(n_points, p))
                       for p in (0, 1)]

    def chebyshev_coefficients(self, values):
        """a_k with f(t) ≈ Σ_k a_k T_k(t); the DCT acts along the last axis."""
        return chebyshev_coefficients(values, self.kind)

    def chebyshev_to_legendre_integrals(self, coeffs):
        """(B, M) Chebyshev coefficients -> (B, M) integrals ∫ f P_l dt."""
        out = np.empty_like(coeffs)
        for parity, block in enumerate(self.blocks):
            # We want a_block @ Jᵀ. BLAS works column-major, so compute the
            # equivalent J @ a_blockᵀ: a_blockᵀ is already a column-major view.
            a_block_t = np.ascontiguousarray(coeffs[:, parity::2]).T
            result_t = blas.dtrmm(1.0, block, a_block_t, side=0, lower=0, trans_a=0)
            out[:, parity::2] = result_t.T
        return out

    def __call__(self, values):
        """
        values : array (..., M), functions sampled at the Chebyshev points.
        returns: array (..., M), entry [..., l] = ∫_{-1}^{1} f(t) P_l(t) dt.
        """
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != self.n_points:
            raise ValueError(f"last axis must have length {self.n_points}")

        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self.n_points)
        out = np.empty_like(flat)

        for start in range(0, flat.shape[0], self.batch_size):
            chunk = slice(start, start + self.batch_size)
            coeffs = self.chebyshev_coefficients(flat[chunk])
            out[chunk] = self.chebyshev_to_legendre_integrals(coeffs)

        return out.reshape(*leading_shape, self.n_points)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from numpy.polynomial import chebyshev as C, legendre as Leg
    from scipy.special import eval_legendre, spherical_in

    rng = np.random.default_rng(0)
    methods = {"fft": ChebyshevLegendreTransform, "dense": DenseChebyshevLegendreTransform}

    for kind in (2, 1):
        print(f"\n=== Chebyshev points of the {'second' if kind == 2 else 'first'} kind ===")

        for name, method in methods.items():
            print(f"--- {name} ---")

            # 1. Legendre polynomials themselves: ∫ P_m P_l = 2/(2l+1) δ_lm (exact).
            m_pts = 513
            transform = method(m_pts, kind=kind)
            x = chebyshev_points(m_pts, kind)
            degrees = np.array([0, 1, 2, 7, 100, 333, 512])
            values = eval_legendre(degrees[:, None], x[None, :])
            expected = np.zeros((len(degrees), m_pts))
            expected[np.arange(len(degrees)), degrees] = 2.0 / (2 * degrees + 1)
            err = np.max(np.abs(transform(values) - expected))
            print(f"P_l inputs             max abs error: {err:.2e}")

            # 2. Exponentials: ∫ e^{a t} P_l(t) dt = 2 i_l(a)  (modified spherical Bessel).
            a_values = np.array([1.0, 20.0, 100.0])
            values = np.exp(a_values[:, None] * x[None, :])
            ell = np.arange(m_pts)
            exact = 2.0 * spherical_in(ell[None, :], a_values[:, None])
            result = transform(values)
            rel = np.max(np.abs(result - exact), axis=1) / np.max(np.abs(exact), axis=1)
            for a, r in zip(a_values, rel):
                print(f"exp({a:5.0f} t)           max error / max |I_l|: {r:.2e}")

            # 3. Random polynomials vs. Gauss-Legendre quadrature of the interpolant.
            m_pts = 300
            transform = method(m_pts, kind=kind)
            x = chebyshev_points(m_pts, kind)
            decay = np.exp(-np.arange(m_pts) / 60.0)
            cheb_coeffs = rng.standard_normal((4, 5, m_pts)) * decay      # extra batch axes
            flat_coeffs = cheb_coeffs.reshape(-1, m_pts).T
            values = C.chebval(x, flat_coeffs, tensor=True).reshape(4, 5, m_pts)
            nodes, weights = Leg.leggauss(m_pts)                            # exact to degree 2M-1
            f_at_nodes = C.chebval(nodes, flat_coeffs, tensor=True)
            p_at_nodes = eval_legendre(np.arange(m_pts)[:, None], nodes[None, :])
            reference = (f_at_nodes * weights) @ p_at_nodes.T
            result = transform(values)
            err = np.max(np.abs(result.reshape(-1, m_pts) - reference)) / np.max(np.abs(reference))
            print(f"random, shape {values.shape} -> {result.shape}, "
                  f"rel. error vs Gauss-Legendre: {err:.2e}")

    # 4. Timing on a larger batch, and agreement between the two methods.
    m_pts, n_funcs = 4097, 2000
    x = chebyshev_points(m_pts)
    values = np.exp(np.outer(rng.uniform(-50, 50, n_funcs), x))
    print(f"\n=== Timing: M = {m_pts}, N = {n_funcs} ===")
    results = {}
    for name, method in methods.items():
        t0 = time.perf_counter()
        transform = method(m_pts, kind=2)
        t1 = time.perf_counter()
        results[name] = transform(values)
        t2 = time.perf_counter()
        print(f"{name:6s} precompute {t1 - t0:6.2f} s, transform {t2 - t1:6.2f} s "
              f"({1e3 * (t2 - t1) / n_funcs:.3f} ms per function)")
    diff = np.max(np.abs(results["fft"] - results["dense"])) / np.max(np.abs(results["dense"]))
    print(f"fft vs dense, max difference / max |I_l|: {diff:.2e}")
