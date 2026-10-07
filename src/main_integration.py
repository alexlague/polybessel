import math
import numpy as np
from numba import jit
from interpolation import cheb8, cheb8_deriv, jv_interpolation
from IN_integration import IN_Bessel_s2_single_func, IN_Bessel_s2_product_func, IN_Bessel_s2_hybrid_func
#from bessel_interp import   # adjust to wherever this lives
from integration import weights_clenshaw_curtis


def make_w_levels(ncheb, nmin=16):
    """
    Clenshaw-Curtis weights for the nested rules n = ncheb, ncheb/2, ..., nmin
    (row l holds the n = ncheb >> l weights in its first n+1 entries; rest is 0).
    Nodes of level l are u[::2**l], since Chebyshev-Lobatto nodes are nested.
    Call once, outside numba, alongside u and w.
    """
    nlev = int(math.log2(ncheb // nmin)) + 1
    w_levels = np.zeros((nlev, ncheb + 1))
    for l in range(nlev):
        n = ncheb >> l
        w_levels[l, :n+1] = weights_clenshaw_curtis(n)
    return w_levels


@jit(nopython=True)
def _hybrid_nested(c, u, w_levels, nu1, omega1, nu2, omega2, lo, hi, a, b, fplo, fphi):
    """
    IN_Bessel_s2_hybrid_func on [lo, hi] using the coarsest nested CC rule
    (nodes u[::2**l]) that resolves the difference phase xi1 - xi2
    """
    # local rate of the difference phase, |xi1' - xi2'|, at lo, mid and hi
    rate = 0.0
    for xr in (float(lo), 0.5*(lo+hi), float(hi)):
        d = math.sqrt(omega1*omega1 - (nu1/xr)**2) - math.sqrt(omega2*omega2 - (nu2/xr)**2)
        rate = max(rate, abs(d))
    n_needed = 1.3*rate*(hi-lo)/math.pi + 64.0

    # coarsest level with enough nodes (never finer than the full rule)
    ncheb = u.size - 1
    lev = 0
    n = ncheb
    while lev + 1 < w_levels.shape[0] and (n // 2) >= n_needed:
        n //= 2
        lev += 1
    step = ncheb // n

    nodes = 0.5*(hi-lo) * u[::step] + 0.5*(hi+lo)
    fx = cheb8(c, nodes, a, b)
    wl = 0.5*(hi-lo) * w_levels[lev, :n+1]
    return IN_Bessel_s2_hybrid_func(nu1, omega1, nu2, omega2, lo, hi, fplo, fphi, nodes[::-1], fx[::-1], wl[::-1])


@jit(nopython=True)
def _cc_composite(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, lo, hi, a, b):
    """
    Clenshaw-Curtis on [lo, hi] resolving the Bessel oscillations: uses the coarsest
    nested rule (nodes u[::2**l], weights w_levels[l]) with enough nodes, and only
    splits into panels of the full ncheb rule if even that is not enough
    """
    if hi <= lo:
        return 0.0
    # product J1*J2 oscillates at up to omega1 + omega2
    phase = (omega1 + omega2) * (hi - lo)
    n_needed = 1.3*phase/math.pi + 64.0

    ncheb = u.size - 1
    if n_needed <= ncheb:
        # coarsest nested level with enough nodes, single panel
        npan = 1
        lev = 0
        n = ncheb
        while lev + 1 < w_levels.shape[0] and (n // 2) >= n_needed:
            n //= 2
            lev += 1
    else:
        # full rule on several panels
        npan = int(math.ceil(n_needed / ncheb))
        lev = 0
        n = ncheb
    step = ncheb // n
    un = u[::step]
    wn = w_levels[lev, :n+1]

    h = (hi - lo) / npan
    integral = 0.0
    for k in range(npan):
        plo = lo + k*h
        phi = plo + h
        nodes = 0.5*(phi-plo) * un + 0.5*(phi+plo)
        f_at_nodes = cheb8(c, nodes, a, b)
        jv1 = jv_interpolation(Jv_int_table, nu1, omega1*nodes)
        jv2 = jv_interpolation(Jv_int_table, nu2, omega2*nodes)
        integral += np.dot(wn*0.5*(phi-plo), f_at_nodes*jv1*jv2)
    return integral


@jit(nopython=True)
def _single_oscillator_piece(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, lo, hi, a, b):
    """
    Integral over [lo, hi] where only the Bessel with the smaller nu/omega is in its
    asymptotic regime: fold the other one into the amplitude and use single-Bessel IN.
    Falls back to composite CC if the amplitude varies too fast.
    """
    if hi <= lo:
        return 0.0
    # oscillator (o) = smaller nu/omega, amplitude Bessel (m) = larger nu/omega
    if nu1/omega1 > nu2/omega2:
        nu_o, om_o, nu_m, om_m = nu2, omega2, nu1, omega1
    else:
        nu_o, om_o, nu_m, om_m = nu1, omega1, nu2, omega2

    # local rates: oscillator phase speed (slowest at lo) vs amplitude Bessel's
    # growth/oscillation rate (largest at one of the endpoints)
    rate_o = om_o * math.sqrt(1.0 - (nu_o/(om_o*lo))**2)
    rate_m = om_m * max(math.sqrt(abs(1.0 - (nu_m/(om_m*lo))**2)),
                        math.sqrt(abs(1.0 - (nu_m/(om_m*hi))**2)))
    if rate_m > 0.02 * rate_o:
        return _cc_composite(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, lo, hi, a, b)

    endp = np.array([lo, hi])
    f_endp = cheb8(c, endp, a, b)
    fp_endp = cheb8_deriv(c, endp, a, b)
    jv_endp = jv_interpolation(Jv_int_table, nu_m, om_m*endp)
    # d/dx J_nu(om x) = (nu/x) J_nu(om x) - om J_{nu+1}(om x)   (valid for nu = 0)
    jv_endp_deriv = nu_m/endp * jv_endp - om_m * jv_interpolation(Jv_int_table, nu_m+1, om_m*endp)

    # amplitude g = f * J_m and its derivative g' = f' J_m + f J_m'
    g = f_endp * jv_endp
    gp = fp_endp * jv_endp + f_endp * jv_endp_deriv
    return IN_Bessel_s2_single_func(nu_o, om_o, lo, hi, g[0], g[1], gp[0], gp[1])


@jit(nopython=True)
def main_integration_Bessel_product(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a, b, ncheb=2048):
    """
    Main wrapper to integrate any smooth function f (callable) with two Bessel functions
    f(x) * J_nu1(omega1*x) J_nu2(omega2*x)
    """
    
    # Get Cheb nodes on [-1, 1]
    #u = nodes_clenshaw_curtis(ncheb, -1, 1)
    
    # Bounds as floats (int a, b break numba tuple typing and give int arrays in cheb8)
    a = float(a)
    b = float(b)
    
    # Get Cheb nodes on [a, b]
    xcc = 0.5*(b-a) * u + 0.5*(b+a)
    
    # Get Cheb weights
    #w = weights_clenshaw_curtis(ncheb) # for [-1, 1]
    w = w_levels[0] # full ncheb rule (row 0 of the nested weights)
    
    # Cheb decomposition of deg 7 of f over the full interval
    #c = Cheb_coeffs_dct(f(xcc[::ncheb//8]))

    if (b*omega1/nu1 < 0.4 and nu1 >= 30) or (b*omega2/nu2 < 0.4 and nu2 >= 30):
        #print("Case 0")
        integral = 0.0
    
    elif a >= 2*max((nu1/omega1, nu2/omega2)):
        #print("Case A")
        fa, fb = cheb8(c, np.array([a, b]), a, b)
        fpa, fpb = cheb8_deriv(c, np.array([a, b]), a, b) # get from endpoints of Cheb expansion
        
        if np.abs(omega1-omega2) >= 0.5:
            #print("Case A1")
            integral = IN_Bessel_s2_product_func(nu1, omega1, nu2, omega2, a, b, fa, fb, fpa, fpb)
        else:
            #print("Case A2")
            integral = _hybrid_nested(c, u, w_levels, nu1, omega1, nu2, omega2, a, b, a, b, fpa, fpb)
    
    elif b <= 2*max((nu1/omega1, nu2/omega2)): # CC quadrature + single-oscillator IN
        #print("Case B")
        a_tilde = min(b, max(a, 2*min((nu1/omega1, nu2/omega2))))
        
        # Piece 1/2: a to a tilde (neither Bessel asymptotic)
        integral = _cc_composite(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a, a_tilde, a, b)
        
        # Piece 2/2: a tilde to b (one Bessel asymptotic)
        integral += _single_oscillator_piece(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a_tilde, b, a, b)
    
    else:
        #print("Case C")
        a_tilde = max(a, 2*min((nu1/omega1, nu2/omega2)))
        b_tilde = 2*max((nu1/omega1, nu2/omega2))
        
        # Piece 1/3: a to a tilde
        integral = _cc_composite(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a, a_tilde, a, b)
    
        # Piece 2/3: a tilde to b tilde
        integral += _single_oscillator_piece(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a_tilde, b_tilde, a, b)
        
        # Piece 3/3: b tilde to b
        fb_tilde, fb = cheb8(c, np.array([b_tilde, b]), a, b)
        fpb_tilde, fpb = cheb8_deriv(c, np.array([b_tilde, b]), a, b)
        if np.abs(omega1-omega2) >= 0.5:
            integral += IN_Bessel_s2_product_func(nu1, omega1, nu2, omega2, b_tilde, b, fb_tilde, fb, fpb_tilde, fpb)
        else:
            integral += _hybrid_nested(c, u, w_levels, nu1, omega1, nu2, omega2, b_tilde, b, a, b, fpb_tilde, fpb)
    
    return integral