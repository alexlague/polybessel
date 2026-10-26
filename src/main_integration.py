import math
import numpy as np
from numba import jit
from interpolation import cheb_eval, cheb_eval_deriv, jv_interpolation, bessel_switch
from IN_integration import IN_Bessel_s2_single_func, IN_Bessel_s2_product_func, IN_Bessel_s2_hybrid_func
#from bessel_interp import   # adjust to wherever this lives
from integration import weights_clenshaw_curtis

# MOVED TO POLYBESSEL MAIN
#def make_w_levels(ncheb, nmin=16):
#    """
#    Clenshaw-Curtis weights for the nested rules n = ncheb, ncheb/2, ..., nmin
#    (row l holds the n = ncheb >> l weights in its first n+1 entries; rest is 0).
#    Nodes of level l are u[::2**l], since Chebyshev-Lobatto nodes are nested.
#    Call once, outside numba, alongside u and w.
#    """
#    nlev = int(math.log2(ncheb // nmin)) + 1
#    w_levels = np.zeros((nlev, ncheb + 1))
#    for l in range(nlev):
#        n = ncheb >> l
#        w_levels[l, :n+1] = weights_clenshaw_curtis(n)
#    return w_levels

# Precision parameters (might make them variable later)
cc_density = 1.6 # used to be 1.3
cc_min_nodes = 16 # used to be 64

@jit(nopython=True)
def _log_jmax(nu, omega, b):
    t = omega*b/nu
    if t < 1.0:
        s = math.sqrt(1.0 - t*t)
        return nu*(math.log(t) + s - math.log1p(s))   # Kapteyn, J increasing below nu
    return min(0.0, math.log(0.6749) - math.log(nu)/3.0)  # Landau

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
    n_needed = cc_density*rate*(hi-lo)/math.pi + cc_min_nodes

    # coarsest level with enough nodes (never finer than the full rule)
    ncheb = u.size - 1
    lev = 0
    n = ncheb
    while lev + 1 < w_levels.shape[0] and (n // 2) >= n_needed:
        n //= 2
        lev += 1
    step = ncheb // n

    nodes = 0.5*(hi-lo) * u[::step] + 0.5*(hi+lo)
    fx = cheb_eval(c, nodes, a, b) #cheb8(c, nodes, a, b)
    wl = 0.5*(hi-lo) * w_levels[lev, :n+1]
    return IN_Bessel_s2_hybrid_func(nu1, omega1, nu2, omega2, lo, hi, fplo, fphi, nodes[::-1], fx[::-1], wl[::-1])

@jit(nopython=True)
def _use_IN_product(nu1, omega1, nu2, omega2, lo, hi, kmin=100.0):
    """Analytic s2 IN product is accurate only if xi1'-xi2' has no zero on
    [lo, hi] and |xi1'-xi2'|*x >= kmin at both ends (error ~ (|dxi'| x)^-2)."""
    dlo = math.sqrt(omega1*omega1 - (nu1/lo)**2) - math.sqrt(omega2*omega2 - (nu2/lo)**2)
    dhi = math.sqrt(omega1*omega1 - (nu1/hi)**2) - math.sqrt(omega2*omega2 - (nu2/hi)**2)
    if dlo*dhi <= 0.0:
        return False
    return min(abs(dlo)*lo, abs(dhi)*hi) >= kmin

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
    n_needed = cc_density*phase/math.pi + cc_min_nodes

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
        f_at_nodes = cheb_eval(c, nodes, a, b) #cheb8(c, nodes, a, b)
        jv1 = jv_interpolation(Jv_int_table, nu1, omega1*nodes)
        jv2 = jv_interpolation(Jv_int_table, nu2, omega2*nodes)
        integral += np.dot(wn*0.5*(phi-plo), f_at_nodes*jv1*jv2)
    return integral


@jit(nopython=True)
def _single_oscillator_piece(c, dc, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, lo, hi, a, b):
    """
    Integral over [lo, hi] where only the Bessel with the smaller nu/omega is in its
    asymptotic regime: fold the other one into the amplitude and use single-Bessel IN.
    Falls back to composite CC if the amplitude varies too fast.
    """
    if hi <= lo:
        return 0.0
    # oscillator (o) = Bessel that becomes asymptotic first, amplitude Bessel (m) = the other
    if bessel_switch(nu1)/omega1 > bessel_switch(nu2)/omega2:
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
    f_endp = cheb_eval(c, endp, a, b) #cheb8(c, endp, a, b)
    fp_endp = cheb_eval_deriv(dc, endp,a ,b) #cheb8_deriv(c, endp, a, b)
    jv_endp = jv_interpolation(Jv_int_table, nu_m, om_m*endp)
    # d/dx J_nu(om x) = (nu/x) J_nu(om x) - om J_{nu+1}(om x)   (valid for nu = 0)
    jv_endp_deriv = nu_m/endp * jv_endp - om_m * jv_interpolation(Jv_int_table, nu_m+1, om_m*endp)

    # amplitude g = f * J_m and its derivative g' = f' J_m + f J_m'
    g = f_endp * jv_endp
    gp = fp_endp * jv_endp + f_endp * jv_endp_deriv
    return IN_Bessel_s2_single_func(nu_o, om_o, lo, hi, g[0], g[1], gp[0], gp[1])


#========================================#
# Single-Bessel integrals: int_a^b f(x) J_nu(omega x) dx
#========================================#

@jit(nopython=True)
def _use_IN_single(nu, omega, lo, hi, f_e, fp_e, fpp_e, kmin=100.0, eta=1e-3):
    """
    True if the analytic s2 IN single-Bessel formula is accurate on [lo, hi]:
    - Bessel side: xi'(x) x = sqrt(omega^2 x^2 - nu^2) >= kmin at lo (smallest there);
    - f side: f must vary slowly on the oscillation scale at both endpoints,
      |f''| <= eta * xi'^2 * (|f| + |f'|/xi').
    The relative s2 truncation error scales like (xi' x)^-2 on the Bessel side and
    like f''/(f xi'^2) on the f side.
    """
    if omega*omega*lo*lo - nu*nu < kmin*kmin:
        return False
    for k in range(2):
        x = lo if k == 0 else hi
        xp = math.sqrt(omega*omega - (nu/x)**2)
        if abs(fpp_e[k]) > eta*xp*xp*(abs(f_e[k]) + abs(fp_e[k])/xp):
            return False
    return True


@jit(nopython=True)
def _cc_composite_single(c, u, w_levels, Jv_int_table, nu, omega, lo, hi, a, b):
    """
    Clenshaw-Curtis on [lo, hi] for f(x) J_nu(omega x): coarsest nested rule that
    resolves the oscillation (rate <= omega), split into panels of the full rule if needed.
    """
    if hi <= lo:
        return 0.0
    phase = omega * (hi - lo)
    n_needed = cc_density*phase/math.pi + cc_min_nodes

    ncheb = u.size - 1
    if n_needed <= ncheb:
        npan = 1
        lev = 0
        n = ncheb
        while lev + 1 < w_levels.shape[0] and (n // 2) >= n_needed:
            n //= 2
            lev += 1
    else:
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
        f_at_nodes = cheb_eval(c, nodes, a, b)
        jv = jv_interpolation(Jv_int_table, nu, omega*nodes)
        integral += np.dot(wn*0.5*(phi-plo), f_at_nodes*jv)
    return integral


@jit(nopython=True)
def _single_asymptotic_piece(c, dc, ddc, u, w_levels, Jv_int_table, nu, omega, lo, hi, a, b):
    """[lo, hi] with J_nu asymptotic: analytic IN if accurate, else composite CC."""
    if hi <= lo:
        return 0.0
    endp = np.array([lo, hi])
    f_endp = cheb_eval(c, endp, a, b)
    fp_endp = cheb_eval(dc, endp, a, b)
    fpp_endp = cheb_eval(ddc, endp, a, b)
    if _use_IN_single(nu, omega, lo, hi, f_endp, fp_endp, fpp_endp):
        return IN_Bessel_s2_single_func(nu, omega, lo, hi, f_endp[0], f_endp[1], fp_endp[0], fp_endp[1])
    return _cc_composite_single(c, u, w_levels, Jv_int_table, nu, omega, lo, hi, a, b)


@jit(nopython=True)
def main_integration_Bessel_single(c, dc, ddc, u, w_levels, Jv_int_table, nu, omega, a, b, ncheb=2048, tol=1e-6):
    """
    Integral of f(x) J_nu(omega x) on [a, b], with f given by its Chebyshev
    coefficients c on [a, b] (dc, ddc: coefficients of f' and f'').
    Case 0: a-priori bound (b-a) max|f| max|J| below tol -> 0.
    Otherwise CC quadrature below x_t = 2 nu/omega, single-Bessel IN (or CC) above.
    """
    a = float(a)
    b = float(b)

    fmax = np.sum(np.abs(c))
    if math.log((b-a)*fmax) + _log_jmax(nu, omega, b) < math.log(tol):
        return 0.0

    x_t = bessel_switch(nu)/omega
    if a >= x_t:
        # Case A: asymptotic on the whole interval
        return _single_asymptotic_piece(c, dc, ddc, u, w_levels, Jv_int_table, nu, omega, a, b, a, b)
    elif b <= x_t:
        # Case B: transition region only
        return _cc_composite_single(c, u, w_levels, Jv_int_table, nu, omega, a, b, a, b)
    else:
        # Case C: CC up to x_t, asymptotic beyond
        integral = _cc_composite_single(c, u, w_levels, Jv_int_table, nu, omega, a, x_t, a, b)
        integral += _single_asymptotic_piece(c, dc, ddc, u, w_levels, Jv_int_table, nu, omega, x_t, b, a, b)
        return integral


@jit(nopython=True)
def main_integration_Bessel_single_many(c, dc, ddc, u, w_levels, Jv_int_table, nus, omegas, a, b, ncheb=2048, tol=1e-6):
    """
    main_integration_Bessel_single over the grid nus x omegas, reusing the same
    Chebyshev fit (c, dc, ddc) of f. Returns out[i, j] for (nus[i], omegas[j]).
    """
    N = nus.shape[0]
    M = omegas.shape[0]
    flat = np.empty(N * M)
    for idx in range(N * M):
        i = idx // M
        j = idx % M
        flat[idx] = main_integration_Bessel_single(c, dc, ddc, u, w_levels, Jv_int_table,
                                                   nus[i], omegas[j], a, b, ncheb, tol)
    return flat.reshape((N, M))


#========================================#
# Double-Bessel integrals: int_a^b f(x) J_nu1(omega1 x) J_nu2(omega2 x) dx
#========================================#


@jit(nopython=True)
def main_integration_Bessel_product(c, dc, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a, b, ncheb=2048, tol=1e-6):
    """
    Main wrapper to integrate any smooth function f (callable) with two Bessel functions
    f(x) * J_nu1(omega1*x) J_nu2(omega2*x)
    """
    
    # Get Cheb nodes on [-1, 1]
    #u = nodes_clenshaw_curtis(ncheb, -1, 1)
    
    # Bounds as floats (int a, b break numba tuple typing and give int arrays in cheb interpolation)
    a = float(a)
    b = float(b)
    
    # Get Cheb nodes on [a, b]
    xcc = 0.5*(b-a) * u + 0.5*(b+a)
    
    # Get Cheb weights
    #w = weights_clenshaw_curtis(ncheb) # for [-1, 1]
    w = w_levels[0] # full ncheb rule (row 0 of the nested weights)
    
    # Cheb decomposition of deg 7 of f over the full interval
    #c = Cheb_coeffs_dct(f(xcc[::ncheb//8]))

    #if (b*omega1/nu1 < 0.4 and nu1 >= 30) or (b*omega2/nu2 < 0.4 and nu2 >= 30):
    #    #print("Case 0")
    #    integral = 0.0
    # Case 0 condition (before Case A)
    fmax = np.sum(np.abs(c))
    if math.log((b-a)*fmax) + _log_jmax(nu1, omega1, b) + _log_jmax(nu2, omega2, b) < math.log(tol):
        integral = 0.0
    
    elif a >= max((bessel_switch(nu1)/omega1, bessel_switch(nu2)/omega2)):
        #print("Case A")
        fa, fb = cheb_eval(c, np.array([a, b]), a, b) #cheb8(c, np.array([a, b]), a, b)
        fpa, fpb = cheb_eval_deriv(dc, np.array([a, b]), a, b) #cheb8_deriv(c, np.array([a, b]), a, b) # get from endpoints of Cheb expansion
        
        if _use_IN_product(nu1, omega1, nu2, omega2, a, b, kmin=100.0):
            #print("Case A1")
            integral = IN_Bessel_s2_product_func(nu1, omega1, nu2, omega2, a, b, fa, fb, fpa, fpb)
        else:
            #print("Case A2")
            integral = _hybrid_nested(c, u, w_levels, nu1, omega1, nu2, omega2, a, b, a, b, fpa, fpb)
    
    elif b <= max((bessel_switch(nu1)/omega1, bessel_switch(nu2)/omega2)): # CC quadrature + single-oscillator IN
        #print("Case B")
        a_tilde = min(b, max(a, min((bessel_switch(nu1)/omega1, bessel_switch(nu2)/omega2))))
        
        # Piece 1/2: a to a tilde (neither Bessel asymptotic)
        integral = _cc_composite(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a, a_tilde, a, b)
        
        # Piece 2/2: a tilde to b (one Bessel asymptotic)
        integral += _single_oscillator_piece(c, dc, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a_tilde, b, a, b)
    
    else:
        #print("Case C")
        a_tilde = max(a, min((bessel_switch(nu1)/omega1, bessel_switch(nu2)/omega2)))
        b_tilde = max((bessel_switch(nu1)/omega1, bessel_switch(nu2)/omega2))
        
        # Piece 1/3: a to a tilde
        integral = _cc_composite(c, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a, a_tilde, a, b)
    
        # Piece 2/3: a tilde to b tilde
        integral += _single_oscillator_piece(c, dc, u, w_levels, Jv_int_table, nu1, omega1, nu2, omega2, a_tilde, b_tilde, a, b)
        
        # Piece 3/3: b tilde to b
        fb_tilde, fb = cheb_eval(c, np.array([b_tilde, b]), a, b) #cheb8(c, np.array([b_tilde, b]), a, b)
        fpb_tilde, fpb = cheb_eval_deriv(dc, np.array([b_tilde, b]), a, b) #cheb8_deriv(c, np.array([b_tilde, b]), a, b)
        if _use_IN_product(nu1, omega1, nu2, omega2, b_tilde, b):
            integral += IN_Bessel_s2_product_func(nu1, omega1, nu2, omega2, b_tilde, b, fb_tilde, fb, fpb_tilde, fpb)
        else:
            integral += _hybrid_nested(c, u, w_levels, nu1, omega1, nu2, omega2, b_tilde, b, a, b, fpb_tilde, fpb)
    
    return integral

@jit(nopython=True)
def _skip_pair(nu1, nu2, max_sep, skip_parity):
    """True if the (nu1, nu2) combination is excluded by the max_sep / parity selection."""
    if abs(nu1 - nu2) > max_sep:
        return True
    if skip_parity >= 0 and int(round(nu1 + nu2)) % 2 == skip_parity:
        return True
    return False

@jit(nopython=True)
def main_integration_Bessel_product_many(c, dc, u, w_levels, Jv_int_table, nu1s, omega1s, nu2s, omega2s, a, b, ncheb=2048, tol=1e-6, max_sep=np.inf, skip_parity=-1):
    """
    main_integration_Bessel_product over the full grid nu1s x omega1s x nu2s x omega2s,
    reusing the same Chebyshev fit (c, dc) of f. Returns out[i1, j1, i2, j2].
    The flattened grid is split across threads (numba.set_num_threads).
    Combinations with |nu1 - nu2| > max_sep are not integrated and set to 0.
    skip_parity: -1 (off), 0 or 1: combinations with round(nu1 + nu2) % 2 == skip_parity
    are not integrated and set to 0 (parity of the orders passed to this kernel).
    """
    N1 = nu1s.shape[0]
    M1 = omega1s.shape[0]
    N2 = nu2s.shape[0]
    M2 = omega2s.shape[0]
    ntot = N1 * M1 * N2 * M2
    flat = np.empty(ntot)
    for idx in range(ntot):
        j2 = idx % M2
        r = idx // M2
        i2 = r % N2
        r //= N2
        j1 = r % M1
        i1 = r // M1
        if _skip_pair(nu1s[i1], nu2s[i2], max_sep, skip_parity):
            flat[idx] = 0.0
            continue
        flat[idx] = main_integration_Bessel_product(c, dc, u, w_levels, Jv_int_table,
                                                    nu1s[i1], omega1s[j1], nu2s[i2], omega2s[j2],
                                                    a, b, ncheb, tol)
    return flat.reshape((N1, M1, N2, M2))

@jit(nopython=True)
def main_integration_Bessel_product_many_sym(c, dc, u, w_levels, Jv_int_table, nus, omegas, a, b, ncheb=2048, tol=1e-6, max_sep=np.inf, skip_parity=-1):
    """
    Symmetric version of main_integration_Bessel_product_many for nu1s = nu2s = nus and
    omega1s = omega2s = omegas. Uses I(nu1, om1, nu2, om2) = I(nu2, om2, nu1, om1): only
    pairs p <= q of the flattened (nu, omega) index p = i*M + j are integrated, and the
    result is mirrored. Returns out[i1, j1, i2, j2] with shape (N, M, N, M).
    Combinations with |nu1 - nu2| > max_sep are not integrated and set to 0.
    skip_parity: -1 (off), 0 or 1: combinations with round(nu1 + nu2) % 2 == skip_parity
    are not integrated and set to 0 (parity of the orders passed to this kernel).
    """
    N = nus.shape[0]
    M = omegas.shape[0]
    P = N * M
    ntri = P * (P + 1) // 2
    out = np.empty((N, M, N, M))
    for k in range(ntri):
        # invert k -> (p, q), p <= q, row-major over the upper triangle
        p = int((2*P + 1 - math.sqrt((2*P + 1)**2 - 8.0*k)) // 2)
        # guard against rounding in the sqrt
        while p > 0 and p*(2*P - p + 1)//2 > k:
            p -= 1
        while (p + 1)*(2*P - p)//2 <= k:
            p += 1
        q = p + k - p*(2*P - p + 1)//2
        i1 = p // M
        j1 = p % M
        i2 = q // M
        j2 = q % M
        if _skip_pair(nus[i1], nus[i2], max_sep, skip_parity):
            out[i1, j1, i2, j2] = 0.0
            out[i2, j2, i1, j1] = 0.0
            continue
        val = main_integration_Bessel_product(c, dc, u, w_levels, Jv_int_table,
                                              nus[i1], omegas[j1], nus[i2], omegas[j2],
                                              a, b, ncheb, tol)
        out[i1, j1, i2, j2] = val
        out[i2, j2, i1, j1] = val
    return out

