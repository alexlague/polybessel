"""
gauss_legendre_tables.py
========================

Bessel-product tables by direct Gauss-Legendre quadrature, with the Bessel
functions precomputed once per grid:

    tables[(name, u, N)][i, j, l] = ∫_a^b S_name(x) x^u j_l(ks[i] x) j_{l+N}(qs[j] x) dx
                                 ≈ Σ_m w_m S(x_m) x_m^u  j_l(ks[i] x_m)  j_{l+N}(qs[j] x_m).

Same requests and output layout as bessel_product_kernels.compute_tables
(dict keyed by (name, u, N), arrays of shape (len(ks), len(qs), n_ell),
NaN where the second Bessel order l + N would be negative).

When to use it
--------------
The tables j_l(k x_m) and j_l(q x_m) do not depend on S. Once they exist,
each new weight costs one batched matrix product per (u, N):

    for every l:   (n_k × n_x) · diag(w S x^u) · (n_x × n_q).

That makes it the natural choice when many weights S(x) are evaluated on
the same (ks, qs, [a, b]) grid. S can be ANY vectorized callable
(including bpk.PiecewisePolynomial): no polynomial fit is needed.

Cost and accuracy
-----------------
The integrand oscillates at frequencies up to k_max + q_max, so the number
of nodes grows with (k_max + q_max)(b - a). The default uses panels of
32 nodes, each spanning at most `phase_per_panel` radians of that phase;
Gauss-Legendre then converges exponentially (check by doubling the panels).
If S has kinks (e.g. a piecewise polynomial), pass its edges as
`breakpoints` so that no panel straddles one.

Memory: the two Bessel tables hold n_ell × (n_k + n_q) × n_x numbers.
Large k ranges or long x ranges make n_x, and so the tables, large; the
product-formula path in bessel_product_kernels scales better there.

Typical use
-----------
    import gauss_legendre_tables as glt

    gl = glt.GaussLegendreTables(ks, qs, a=1000.0, b=2000.0, n_ell=201, max_order_shift=2)
    tables = gl.compute_tables({"F": F}, [("F", 2, 0), ("F", 0, 1), ("F", 1, -2)])
    # ... any number of further weights on the same grid:
    tables_G = gl.compute_tables({"G": G}, [("G", 2, 0)])

With cosmo_model (kernels and multipoles), pass the object as the table function:

    W = cm.compute_kernels([kernel], ks, qs, weights, None, tables_function=gl)
"""

import numpy as np
from scipy.special import spherical_jn


# ---------------------------------------------------------------------------
# Quadrature nodes
# ---------------------------------------------------------------------------

def gauss_legendre_panels(panel_edges, nodes_per_panel):
    """Nodes and weights of composite Gauss-Legendre on the given panel edges."""
    t, w = np.polynomial.legendre.leggauss(nodes_per_panel)
    lo = np.asarray(panel_edges[:-1], dtype=float)[:, None]
    hi = np.asarray(panel_edges[1:], dtype=float)[:, None]
    nodes = 0.5 * (hi - lo) * t + 0.5 * (hi + lo)
    weights = 0.5 * (hi - lo) * w
    return nodes.ravel(), weights.ravel()


def default_panel_edges(a, b, max_frequency, phase_per_panel=40.0, breakpoints=None):
    """
    Panel edges on [a, b] so that each panel spans at most `phase_per_panel`
    radians of the fastest oscillation (frequency `max_frequency`), with every
    breakpoint inside (a, b) kept as an edge.
    """
    fixed = [a, b] + [float(x) for x in (breakpoints if breakpoints is not None else []) if a < x < b]
    fixed = np.unique(fixed)
    edges = [fixed[0]]
    for lo, hi in zip(fixed[:-1], fixed[1:]):
        n_panels = max(1, int(np.ceil(max_frequency * (hi - lo) / phase_per_panel)))
        edges.extend(np.linspace(lo, hi, n_panels + 1)[1:])
    return np.array(edges)


# ---------------------------------------------------------------------------
# Bessel tables
# ---------------------------------------------------------------------------

def bessel_table(n_orders, first, second, orders_per_chunk=32):
    """
    j_l(first[i] * second[j]) for l = 0 ... n_orders-1.
    Shape (n_orders, len(first), len(second)), built in chunks of orders to
    bound the size of the temporary arrays. Called as (ks, nodes) it gives
    j_l(k x) indexed [l, k, x]; as (nodes, qs) it gives [l, x, q].
    """
    z = np.outer(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    table = np.empty((n_orders, *z.shape))
    for start in range(0, n_orders, orders_per_chunk):
        orders = np.arange(start, min(start + orders_per_chunk, n_orders))
        table[orders] = spherical_jn(orders[:, None, None], z[None, :, :])
    return table


# ---------------------------------------------------------------------------
# The reusable grid
# ---------------------------------------------------------------------------

class GaussLegendreTables:
    """
    Quadrature nodes and Bessel tables for one (ks, qs, [a, b]) grid.

    ks, qs          : 1D arrays, the first and second Bessel arguments.
    a, b            : integration range, 0 <= a < b.
    n_ell           : number of output orders l = 0 ... n_ell-1 (the last axis).
    max_order_shift : largest positive N that will be requested; the q table
                      holds orders up to n_ell - 1 + max_order_shift.
    nodes_per_panel, phase_per_panel, breakpoints : see default_panel_edges.
    panel_edges     : explicit panel edges (overrides the defaults).
    """

    def __init__(self, ks, qs, a, b, n_ell, max_order_shift=6, nodes_per_panel=32,
                 phase_per_panel=40.0, breakpoints=None, panel_edges=None):
        if not 0 <= a < b:
            raise ValueError("need 0 <= a < b")
        self.ks = np.asarray(ks, dtype=float)
        self.qs = np.asarray(qs, dtype=float)
        self.a, self.b = float(a), float(b)
        self.n_ell = int(n_ell)
        self.max_order_shift = int(max_order_shift)

        if panel_edges is None:
            max_frequency = self.ks.max() + self.qs.max()
            panel_edges = default_panel_edges(self.a, self.b, max_frequency, phase_per_panel, breakpoints)
        self.panel_edges = np.asarray(panel_edges, dtype=float)
        self.nodes, self.weights = gauss_legendre_panels(self.panel_edges, nodes_per_panel)

        # The expensive part, done once: independent of every weight function.
        # Layouts chosen so that each order is a plain (k × x) · (x × q) product:
        #   k_table[l, i, m] = j_l(ks[i] x_m),   q_table[l, m, j] = j_l(x_m qs[j]).
        self.k_table = bessel_table(self.n_ell, self.ks, self.nodes)
        self.q_table = bessel_table(self.n_ell + self.max_order_shift, self.nodes, self.qs)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def table_bytes(self):
        return self.k_table.nbytes + self.q_table.nbytes

    # -- one table ------------------------------------------------------------

    def integrate(self, node_values, N, orders_per_chunk=64):
        """
        Σ_m node_values[m] j_l(k x_m) j_{l+N}(q x_m) for all k, q and l,
        where node_values already include the quadrature weights.
        Returns (len(ks), len(qs), n_ell), NaN where l + N < 0.
        """
        if N > self.max_order_shift:
            raise ValueError(f"N = {N} exceeds max_order_shift = {self.max_order_shift}; "
                             "rebuild the grid with a larger max_order_shift")
        out = np.full((len(self.ks), len(self.qs), self.n_ell), np.nan)
        for l0 in range(max(0, -N), self.n_ell, orders_per_chunk):
            l1 = min(l0 + orders_per_chunk, self.n_ell)
            weighted_k = self.k_table[l0:l1] * node_values                    # (n_l, n_k, n_x)
            products = np.matmul(weighted_k, self.q_table[l0 + N:l1 + N])     # (n_l, n_k, n_q)
            out[:, :, l0:l1] = products.transpose(1, 2, 0)
        return out

    # -- the bessel_product_kernels API -------------------------------------

    def compute_tables(self, weights, requests):
        """
        tables[(name, u, N)][i, j, l] = ∫_a^b S_name(x) x^u j_l(ks[i] x) j_{l+N}(qs[j] x) dx

        weights  : {name: vectorized callable S(x)} (bpk.PiecewisePolynomial works).
        requests : iterable of (name, u, N); N may be negative.
        """
        requests = list(dict.fromkeys((name, int(u), int(N)) for name, u, N in requests))
        unknown = {name for name, _, _ in requests} - set(weights)
        if unknown:
            raise KeyError(f"no weight function named {sorted(unknown)}")

        sampled = {name: np.asarray(weights[name](self.nodes), dtype=float)
                   for name in {name for name, _, _ in requests}}
        return {(name, u, N): self.integrate(self.weights * sampled[name] * self.nodes**u, N)
                for name, u, N in requests}

    def __call__(self, ks, qs, weights, requests):
        """Table function for cosmo_model (`tables_function=`): checks the grid, then compute_tables."""
        if not (np.array_equal(np.asarray(ks, dtype=float), self.ks)
                and np.array_equal(np.asarray(qs, dtype=float), self.qs)):
            raise ValueError("ks / qs differ from the grid this GaussLegendreTables was built for")
        return self.compute_tables(weights, requests)


def compute_tables(ks, qs, weights, requests, a, b, n_ell, **grid_options):
    """One-shot version: build the grid for these requests, then compute the tables."""
    max_shift = max([0] + [int(N) for _, _, N in requests])
    grid = GaussLegendreTables(ks, qs, a, b, n_ell, max_order_shift=max_shift, **grid_options)
    return grid.compute_tables(weights, requests)
