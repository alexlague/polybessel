import math
from numba import jit

PI4 = math.pi / 4
TWO_OVER_PI = 2.0 / math.pi

@jit(nopython=True, fastmath=True)
def IN_Bessel_s2_single_func(nu, omega, a, b, fa, fb, fpa, fpb):
    """
    Now with s2 term
    """
    # Re-used values
    om2 = omega*omega
    nu2 = nu*nu
    nu3 = nu2*nu
    
    a2 = a*a
    a3 = a2*a
    b2 = b*b
    b3 = b2*b
    
    sqrt_om2_a2_nu2 = (om2*a2-nu2)**0.5
    sqrt_om2_b2_nu2 = (om2*b2-nu2)**0.5

    # Cotangent function and its powers (change to list?)
    ca = nu / sqrt_om2_a2_nu2
    cb = nu / sqrt_om2_b2_nu2
    ca2 = ca*ca
    cb2 = cb*cb
    ca3 = ca2*ca
    cb3 = cb2*cb
    ca4 = ca3*ca
    cb4 = cb3*cb
    ca5 = ca4*ca
    cb5 = cb4*cb
    ca6 = ca5*ca
    cb6 = cb5*cb

    # Amplitude prefactor
    ampa = (ca*TWO_OVER_PI/nu)**0.5
    ampb = (cb*TWO_OVER_PI/nu)**0.5

    # xi function and derivatives
    xa = sqrt_om2_a2_nu2 - nu*math.acos(nu/omega/a) - PI4
    xb = sqrt_om2_b2_nu2 - nu*math.acos(nu/omega/b) - PI4
    xpa = math.sqrt(om2-nu2/a2)
    xpb = math.sqrt(om2-nu2/b2)
    xppa = nu2/a3/xpa
    xppb = nu2/b3/xpb

    xpa2 = xpa*xpa
    xpb2 = xpb*xpb
    xpa3 = xpa2*xpa
    xpb3 = xpb2*xpb

    # Intermediate functions and first derivatives
    Aa = 1 - (81*ca2+462*ca4+385*ca6)/1152/nu2
    Ab = 1 - (81*cb2+462*cb4+385*cb6)/1152/nu2

    Ba = (3*ca+5*ca3)/24/nu
    Ba += -ca3*(30375+369603*ca2+765765*ca4+425425*ca6)/414720/nu3
    Bb = (3*cb+5*cb3)/24/nu
    Bb += -cb3*(30375+369603*cb2+765765*cb4+425425*cb6)/414720/nu3

    Apa = -ca*(385*ca4+308*ca2+27)/192/nu2
    Apb = -cb*(385*cb4+308*cb2+27)/192/nu2

    Bpa = (5*ca2+1)/8/nu
    Bpa += -ca2*(85085*ca6+119119*ca4+41067*ca2+2025)/9216/nu3
    Bpb = (5*cb2+1)/8/nu
    Bpb += -cb2*(85085*cb6+119119*cb4+41067*cb2+2025)/9216/nu3

    # Polynomial coefficients functions and first derivatives
    Pa = ampa * Aa
    Pb = ampb * Ab
    Qa = ampa * Ba
    Qb = ampb * Bb
    
    prefaca = -nu*om2/a2/xpa3 * ampa
    prefacb = -nu*om2/b2/xpb3 * ampb
    Ppa = prefaca * (Apa + Aa/2/ca)
    Ppb = prefacb * (Apb + Ab/2/cb)
    Qpa = prefaca * (Bpa + Ba/2/ca)
    Qpb = prefacb * (Bpb + Bb/2/cb)

    # Combining with user function (splitting real and img parts)
    FaR, FaI = fa*Pa, -fa*Qa
    FbR, FbI = fb*Pb, -fb*Qb
    FpaR, FpaI = fa*Ppa+fpa*Pa, -fa*Qpa-fpa*Qa
    FpbR, FpbI = fb*Ppb+fpb*Pb, -fb*Qpb-fpb*Qb
    
    # Phi = -i F/xp + F'/xp^2 - F xpp/xp^3, then Re[Phi * e^{i x}]
    ka = xppa/xpa3
    kb = xppb/xpb3

    PhiaR =  FaI/xpa + FpaR/xpa2 - FaR*ka
    PhiaI = -FaR/xpa + FpaI/xpa2 - FaI*ka

    PhibR =  FbI/xpb + FpbR/xpb2 - FbR*kb
    PhibI = -FbR/xpb + FpbI/xpb2 - FbI*kb

    result = (PhibR*math.cos(xb) - PhibI*math.sin(xb))
    result -= (PhiaR*math.cos(xa) - PhiaI*math.sin(xa))
    
    return result

    

#========================================#
# Functions for Two-Bessel case
#========================================#




@jit(nopython=True, fastmath=True)
def _PQxi_single(nu, omega, x):
    """
    Per-Bessel amplitude/phase block, factored out of the single-Bessel
    function so it can be reused for both (nu1, omega1) and (nu2, omega2).
    Returns P, Q, P', Q', xi, xi', xi'' at the point x -- same formulas as
    in IN_Bessel_s2_single_func, just without the f(x) folded in yet.
    """
    om2 = omega * omega
    nu2 = nu * nu
    nu3 = nu2 * nu

    x2 = x * x
    x3 = x2 * x

    sqrt_om2_x2_nu2 = (om2 * x2 - nu2) ** 0.5
    c = nu / sqrt_om2_x2_nu2
    c2 = c * c
    c3 = c2 * c
    c4 = c3 * c
    c5 = c4 * c
    c6 = c5 * c

    amp = (c * TWO_OVER_PI / nu) ** 0.5

    xi_ = sqrt_om2_x2_nu2 - nu * math.acos(nu / omega / x) - PI4
    xip_ = math.sqrt(om2 - nu2 / x2)
    xipp_ = nu2 / x3 / xip_
    xip2_ = xip_ * xip_
    xip3_ = xip2_ * xip_

    A = 1 - (81 * c2 + 462 * c4 + 385 * c6) / 1152 / nu2
    Bv = (3 * c + 5 * c3) / 24 / nu
    Bv += -c3 * (30375 + 369603 * c2 + 765765 * c4 + 425425 * c6) / 414720 / nu3
    Ap = -c * (385 * c4 + 308 * c2 + 27) / 192 / nu2
    Bp = (5 * c2 + 1) / 8 / nu
    Bp += -c2 * (85085 * c6 + 119119 * c4 + 41067 * c2 + 2025) / 9216 / nu3

    P = amp * A
    Q = amp * Bv

    prefac = -nu * om2 / x2 / xip3_ * amp
    Pp = prefac * (Ap + A / 2 / c)
    Qp = prefac * (Bp + Bv / 2 / c)

    return P, Q, Pp, Qp, xi_, xip_, xipp_


@jit(nopython=True, fastmath=True)
def IN_Bessel_s2_product_func(nu1, omega1, nu2, omega2, a, b, fa, fb, fpa, fpb):
    """
    s2 term for int f(x) J_nu1(omega1 x) J_nu2(omega2 x) dx, valid away from
    the transition region of either Bessel function and away from any
    resonance where xi1'(x) = xi2'(x) on [a, b] (frequencies sufficiently
    far apart). Same f, f' convention as IN_Bessel_s2_single_func.
    """
    P1a, Q1a, P1pa, Q1pa, xi1a, xip1a, xipp1a = _PQxi_single(nu1, omega1, a)
    P1b, Q1b, P1pb, Q1pb, xi1b, xip1b, xipp1b = _PQxi_single(nu1, omega1, b)
    P2a, Q2a, P2pa, Q2pa, xi2a, xip2a, xipp2a = _PQxi_single(nu2, omega2, a)
    P2b, Q2b, P2pb, Q2pb, xi2b, xip2b, xipp2b = _PQxi_single(nu2, omega2, b)

    # ---------------- sum-phase term: Phi_+ = xi1 + xi2 ----------------
    xa_p = xi1a + xi2a
    xb_p = xi1b + xi2b
    xpa_p = xip1a + xip2a
    xpb_p = xip1b + xip2b
    xppa_p = xipp1a + xipp2a
    xppb_p = xipp1b + xipp2b
    xpa_p2 = xpa_p * xpa_p
    xpa_p3 = xpa_p2 * xpa_p
    xpb_p2 = xpb_p * xpb_p
    xpb_p3 = xpb_p2 * xpb_p
    ka_p = xppa_p / xpa_p3
    kb_p = xppb_p / xpb_p3

    # G_+ = f * a1 * a2, with a_i = P_i - i Q_i
    Ga_pR = fa * (P1a * P2a - Q1a * Q2a)
    Ga_pI = -fa * (P1a * Q2a + Q1a * P2a)
    Gb_pR = fb * (P1b * P2b - Q1b * Q2b)
    Gb_pI = -fb * (P1b * Q2b + Q1b * P2b)

    Gpa_pR = fpa * (P1a * P2a - Q1a * Q2a) + fa * (
        (P1pa * P2a - Q1pa * Q2a) + (P1a * P2pa - Q1a * Q2pa)
    )
    Gpa_pI = -fpa * (P1a * Q2a + Q1a * P2a) - fa * (
        (P1pa * Q2a + Q1pa * P2a) + (P1a * Q2pa + Q1a * P2pa)
    )
    Gpb_pR = fpb * (P1b * P2b - Q1b * Q2b) + fb * (
        (P1pb * P2b - Q1pb * Q2b) + (P1b * P2pb - Q1b * Q2pb)
    )
    Gpb_pI = -fpb * (P1b * Q2b + Q1b * P2b) - fb * (
        (P1pb * Q2b + Q1pb * P2b) + (P1b * Q2pb + Q1b * P2pb)
    )

    Phia_pR = Ga_pI / xpa_p + Gpa_pR / xpa_p2 - Ga_pR * ka_p
    Phia_pI = -Ga_pR / xpa_p + Gpa_pI / xpa_p2 - Ga_pI * ka_p
    Phib_pR = Gb_pI / xpb_p + Gpb_pR / xpb_p2 - Gb_pR * kb_p
    Phib_pI = -Gb_pR / xpb_p + Gpb_pI / xpb_p2 - Gb_pI * kb_p

    result_p = Phib_pR * math.cos(xb_p) - Phib_pI * math.sin(xb_p)
    result_p -= Phia_pR * math.cos(xa_p) - Phia_pI * math.sin(xa_p)

    # -------------- difference-phase term: Phi_- = xi1 - xi2 --------------
    xa_m = xi1a - xi2a
    xb_m = xi1b - xi2b
    xpa_m = xip1a - xip2a
    xpb_m = xip1b - xip2b
    xppa_m = xipp1a - xipp2a
    xppb_m = xipp1b - xipp2b
    xpa_m2 = xpa_m * xpa_m
    xpa_m3 = xpa_m2 * xpa_m
    xpb_m2 = xpb_m * xpb_m
    xpb_m3 = xpb_m2 * xpb_m
    ka_m = xppa_m / xpa_m3
    kb_m = xppb_m / xpb_m3

    # G_- = f * a1 * conj(a2), with conj(a2) = P2 + i Q2
    Ga_mR = fa * (P1a * P2a + Q1a * Q2a)
    Ga_mI = fa * (P1a * Q2a - Q1a * P2a)
    Gb_mR = fb * (P1b * P2b + Q1b * Q2b)
    Gb_mI = fb * (P1b * Q2b - Q1b * P2b)

    Gpa_mR = fpa * (P1a * P2a + Q1a * Q2a) + fa * (
        (P1pa * P2a + Q1pa * Q2a) + (P1a * P2pa + Q1a * Q2pa)
    )
    Gpa_mI = fpa * (P1a * Q2a - Q1a * P2a) + fa * (
        (P1pa * Q2a - Q1pa * P2a) + (P1a * Q2pa - Q1a * P2pa)
    )
    Gpb_mR = fpb * (P1b * P2b + Q1b * Q2b) + fb * (
        (P1pb * P2b + Q1pb * Q2b) + (P1b * P2pb + Q1b * Q2pb)
    )
    Gpb_mI = fpb * (P1b * Q2b - Q1b * P2b) + fb * (
        (P1pb * Q2b - Q1pb * P2b) + (P1b * Q2pb - Q1b * P2pb)
    )

    Phia_mR = Ga_mI / xpa_m + Gpa_mR / xpa_m2 - Ga_mR * ka_m
    Phia_mI = -Ga_mR / xpa_m + Gpa_mI / xpa_m2 - Ga_mI * ka_m
    Phib_mR = Gb_mI / xpb_m + Gpb_mR / xpb_m2 - Gb_mR * kb_m
    Phib_mI = -Gb_mR / xpb_m + Gpb_mI / xpb_m2 - Gb_mI * kb_m

    result_m = Phib_mR * math.cos(xb_m) - Phib_mI * math.sin(xb_m)
    result_m -= Phia_mR * math.cos(xa_m) - Phia_mI * math.sin(xa_m)

    return 0.5 * (result_p + result_m)


@jit(nopython=True, fastmath=True)
def IN_Bessel_s2_hybrid_func(nu1, omega1, nu2, omega2, a, b, fpa, fpb, x, fx, wx):
    """
    Hybrid version for the CLOSE-FREQUENCIES case (xi1'(x)-xi2'(x) small
    somewhere on [a, b], so the difference-phase term can't be trusted to
    the analytic s2 boundary expansion).

    - Sum-phase term  (Phi_+ = xi1+xi2) is always robustly oscillatory,
      regardless of how close omega1, omega2 are -- handled exactly as in
      IN_Bessel_s2_product_func, via the analytic s2 boundary expansion.
    - Difference-phase term (Phi_- = xi1-xi2) is instead integrated directly
      by user-supplied quadrature (nodes x, weights wx, f already evaluated
      as fx), since it is only weakly oscillatory in this regime.

    x, wx: precomputed quadrature nodes/weights on [a, b], with x[0]=a and
           x[-1]=b so that fx[0] = f(a), fx[-1] = f(b).
    fx:    f(x) evaluated at the nodes x.
    fpa, fpb: f'(a), f'(b), still needed for the analytic sum-phase term.
    """
    fa = fx[0]
    fb = fx[-1]

    P1a, Q1a, P1pa, Q1pa, xi1a, xip1a, xipp1a = _PQxi_single(nu1, omega1, a)
    P1b, Q1b, P1pb, Q1pb, xi1b, xip1b, xipp1b = _PQxi_single(nu1, omega1, b)
    P2a, Q2a, P2pa, Q2pa, xi2a, xip2a, xipp2a = _PQxi_single(nu2, omega2, a)
    P2b, Q2b, P2pb, Q2pb, xi2b, xip2b, xipp2b = _PQxi_single(nu2, omega2, b)

    # ---------------- sum-phase term: Phi_+ = xi1 + xi2 (analytic, as before) ----------------
    xa_p = xi1a + xi2a
    xb_p = xi1b + xi2b
    xpa_p = xip1a + xip2a
    xpb_p = xip1b + xip2b
    xppa_p = xipp1a + xipp2a
    xppb_p = xipp1b + xipp2b
    xpa_p2 = xpa_p * xpa_p
    xpa_p3 = xpa_p2 * xpa_p
    xpb_p2 = xpb_p * xpb_p
    xpb_p3 = xpb_p2 * xpb_p
    ka_p = xppa_p / xpa_p3
    kb_p = xppb_p / xpb_p3

    Ga_pR = fa * (P1a * P2a - Q1a * Q2a)
    Ga_pI = -fa * (P1a * Q2a + Q1a * P2a)
    Gb_pR = fb * (P1b * P2b - Q1b * Q2b)
    Gb_pI = -fb * (P1b * Q2b + Q1b * P2b)

    Gpa_pR = fpa * (P1a * P2a - Q1a * Q2a) + fa * (
        (P1pa * P2a - Q1pa * Q2a) + (P1a * P2pa - Q1a * Q2pa)
    )
    Gpa_pI = -fpa * (P1a * Q2a + Q1a * P2a) - fa * (
        (P1pa * Q2a + Q1pa * P2a) + (P1a * Q2pa + Q1a * P2pa)
    )
    Gpb_pR = fpb * (P1b * P2b - Q1b * Q2b) + fb * (
        (P1pb * P2b - Q1pb * Q2b) + (P1b * P2pb - Q1b * Q2pb)
    )
    Gpb_pI = -fpb * (P1b * Q2b + Q1b * P2b) - fb * (
        (P1pb * Q2b + Q1pb * P2b) + (P1b * Q2pb + Q1b * P2pb)
    )

    Phia_pR = Ga_pI / xpa_p + Gpa_pR / xpa_p2 - Ga_pR * ka_p
    Phia_pI = -Ga_pR / xpa_p + Gpa_pI / xpa_p2 - Ga_pI * ka_p
    Phib_pR = Gb_pI / xpb_p + Gpb_pR / xpb_p2 - Gb_pR * kb_p
    Phib_pI = -Gb_pR / xpb_p + Gpb_pI / xpb_p2 - Gb_pI * kb_p

    result_p = Phib_pR * math.cos(xb_p) - Phib_pI * math.sin(xb_p)
    result_p -= Phia_pR * math.cos(xa_p) - Phia_pI * math.sin(xa_p)

    # -------- difference-phase term: Phi_- = xi1 - xi2 (direct quadrature) --------
    result_m = 0.0
    n = x.shape[0]
    for i in range(n):
        xi_pt = x[i]
        fxi = fx[i]
        P1, Q1, _, _, xi1_, _, _ = _PQxi_single(nu1, omega1, xi_pt)
        P2, Q2, _, _, xi2_, _, _ = _PQxi_single(nu2, omega2, xi_pt)

        # G_- = f * a1 * conj(a2), with conj(a2) = P2 + i Q2
        GR = fxi * (P1 * P2 + Q1 * Q2)
        GI = fxi * (P1 * Q2 - Q1 * P2)
        phase = xi1_ - xi2_

        result_m += wx[i] * (GR * math.cos(phase) - GI * math.sin(phase))

    return 0.5 * result_p + 0.5 * result_m