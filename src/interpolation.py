##
# Interpolation routines
##

import numpy as np
from numba import jit, vectorize
from scipy.fft import dct
import math

@jit(nopython=True)
def get_linear_interp_coeffs(x, fx):
    """
    returns an array of (2, len(x)-1) of coefficients
    being the linear mapping connecting the points of x and fx
    (linear interpolation)
    """
    N = len(x)
    alphas = np.zeros((2, N-1))
    for i in range(N-1):
        alphas[1][i] = (fx[i+1]-fx[i]) / (x[i+1]-x[i])
        alphas[0][i] = fx[i] - x[i] * alphas[1][i]
        
    return alphas


def RDP_subsample(x, y, abs_tol=1e-6, rel_tol=1e-2):
    """
    Simplify sampled function data using a Ramer-Douglas-Peucker-like
    algorithm with vertical error |y - linear_interpolation|.

    Parameters
    ----------
    x : array_like
        Increasing x coordinates.
    y : array_like
        Function values f(x).
    abs_tol : float
        Maximum allowed (absolute) vertical interpolation error.
    rel_tol : float
        Maximum allowed (relative) vertical interpolation error.

    Returns
    -------
    indices : ndarray
        Indices of the retained points.
    """
    #x = np.asarray(x)
    #y = np.asarray(y)
    # TO-DO: add assertion that the coordinates are increasing?

    keep = {0, len(x) - 1}

    def recurse(i, j):
        if j <= i + 1:
            return

        t = (x[i+1:j] - x[i]) / (x[j] - x[i])
        y_line = y[i] + t * (y[j] - y[i])

        error = np.abs(y[i+1:j] - y_line)

        tolerance = abs_tol + rel_tol * np.abs(y[i+1:j])

        # Normalised error
        score = error / tolerance

        k = np.argmax(score)

        if score[k] > 1:
            k = i + 1 + k
            keep.add(k)

            recurse(i, k)
            recurse(k, j)

    recurse(0, len(x) - 1)

    return np.array(sorted(keep))

# Replaced with function which only interpolates the "easy" region
# from x: 0 to l
@jit(nopython=True) # make parallel?
def interp_jl(xinterp, jl_table, l, xeval):
    """
    Fast interpolation for spherical Bessel function
    from pre-computed table (must be integer order)
    """
    res = np.zeros(xeval.shape)
    for j, x in enumerate(xeval):
    
        if x < (l+1)/5.:
            res[j] = 0.
        
        #if x > 100.*l:
        #    res[j] = np.sin(x-l*np.pi/2) / x
           
        if l == 0:
            res[j] = np.sin(x)/x
        
        elif l == 1:
            res[j] = np.sin(x)/x**2 - np.cos(x)/x
        
        elif l == 2:
            res[j] = (3./x**2-1) * np.sin(x)/x - 3.*np.cos(x)/x**2
    
        elif l == 3:
            res[j] = (15./x**3-6./x) * np.sin(x)/x - (15./x**2-1)*np.cos(x)/x
    
        else:
            # find nearest point in array
            # TO-DO: add bounds check
            log10_dx = np.log10(xinterp[1]) - np.log10(xinterp[0])
            x0_start = xinterp[0]
            
            ixstar = int((np.log10(x) - np.log10(x0_start)) / log10_dx) # for log-spaced
            xstar = xinterp[ixstar]
            Delta_x = x - xstar
                
            j0 = jl_table[l, ixstar]
            jm1 = jl_table[l-1, ixstar]
            jp1 = jl_table[l+1, ixstar]
        
            deriv1 = (l*jm1 - (l+1)*jp1) / (2*l+1)
            deriv2 = -(2./xstar)*deriv1 + (l*(l+1)/xstar**2 - 1.)*j0
            deriv3 = -(4./xstar)*deriv2 - (2. + xstar**2 - l*(l+1))/xstar**2 * deriv1 - (2./xstar)*j0
        
            taylor = j0 + Delta_x*deriv1 + 0.5*Delta_x**2*deriv2 + (1./6.)*Delta_x**3*deriv3
        
            res[j] = taylor
    
    return res

#############################################
# Chebyshev decomposition and interpolation
#############################################

# The use case here is to compute the nodes for 8
# Cheb points in the domain a, b

def Cheb_coeffs_dct(f_at_nodes):
    """
    Compute the Chebyshev coefficients of the expansion of f
    """
    N = len(f_at_nodes)-1
    # Compute DCT-I (norm=None ensures unnormalized/raw analytical mapping)
    # Note: SciPy orders DCT output matching the node ordering.
    c = dct(f_at_nodes, type=1) / N
    
    # Correct the boundary scaling definitions 
    # Standard Chebyshev series halves the c_0 and c_N terms
    c[0] /= 2
    c[-1] /= 2
    
    # Because u goes from 1 to -1 (descending), flip the coefficients 
    # to fit standard ascending polynomial orders [c0, c1, ..., cN]
    #c = c * (-1) ** np.arange(N + 1)

    return c


@jit(nopython=True, fastmath=True)
def cheb8(c, x, a, b):
    """Evaluate sum_{k=0}^{7} c[k] * T_k(t), t = (2x - a - b) / (b - a), at points x."""
    # Chebyshev -> monomial coefficients in t (from T_0..T_7)
    m0 = c[0] - c[2] + c[4] - c[6]
    m1 = c[1] - 3.0*c[3] + 5.0*c[5] - 7.0*c[7]
    m2 = 2.0*c[2] - 8.0*c[4] + 18.0*c[6]
    m3 = 4.0*c[3] - 20.0*c[5] + 56.0*c[7]
    m4 = 8.0*c[4] - 48.0*c[6]
    m5 = 16.0*c[5] - 112.0*c[7]
    m6 = 32.0*c[6]
    m7 = 64.0*c[7]
 
    # fold the domain scaling in: t = s*(x - mid)  =>  m_k -> m_k * s^k
    mid = 0.5 * (a + b)
    s = 2.0 / (b - a)
    m1 *= s; s2 = s*s
    m2 *= s2; m3 *= s2*s
    s4 = s2*s2
    m4 *= s4; m5 *= s4*s; m6 *= s4*s2; m7 *= s4*s2*s
 
    out = np.empty_like(x)
    for i in range(x.size):
        u = x[i] - mid
        out[i] = ((((((m7*u + m6)*u + m5)*u + m4)*u + m3)*u + m2)*u + m1)*u + m0
    return out

@jit(nopython=True)
def cheb8_deriv(c, x, a, b):
    """Derivative d/dx of sum_{k=0}^{7} c[k] T_k(t) on [a, b], at points x."""
    s = 2.0 / (b - a)          # chain rule: dt/dx
    d = np.empty(8)
    d[7] = 0.0
    d[6] = 14.0 * c[7]
    d[5] = 12.0 * c[6]
    d[4] = d[6] + 10.0 * c[5]
    d[3] = d[5] + 8.0 * c[4]
    d[2] = d[4] + 6.0 * c[3]
    d[1] = d[3] + 4.0 * c[2]
    d[0] = 0.5 * d[2] + c[1]
    for k in range(7):
        d[k] *= s
    return cheb8(d, x, a, b)


@jit(nopython=True)
def cheb8_deriv_endpoints(c, a, b):
    """Return (p'(a), p'(b)) using T_k'(1) = k^2 and T_k'(-1) = (-1)^(k+1) k^2."""
    s = 2.0 / (b - a)
    even = 4.0*c[2] + 16.0*c[4] + 36.0*c[6]
    odd = c[1] + 9.0*c[3] + 25.0*c[5] + 49.0*c[7]
    return -s * (odd + even), -s * (odd - even)



@jit(nopython=True, inline='always')
def _jv_table_scalar(row, scale, imax, z):
    p = z * scale
    i = int(p)
    if i < 1:
        i = 1
    elif i > imax:
        i = imax
    t = p - i
    y0 = row[i-1]; y1 = row[i]; y2 = row[i+1]; y3 = row[i+2]
    return y1 + 0.5 * t * (y2 - y0 + t * (2.0*y0 - 5.0*y1 + 4.0*y2 - y3
                           + t * (3.0*(y1 - y2) + y3 - y0)))

@jit(nopython=True, inline='always')
def _jv_debye_scalar(nu, nu2, nu3, z):
    s = math.sqrt(z*z - nu2)
    c = nu / s
    c2 = c * c
    c4 = c2 * c2
    c6 = c4 * c2
    A = 1.0 - (81.0*c2 + 462.0*c4 + 385.0*c6) / 1152.0 / nu2
    B = (3.0*c + 5.0*c2*c) / 24.0 / nu \
        - c2*c * (30375.0 + 369603.0*c2 + 765765.0*c4 + 425425.0*c6) / 414720.0 / nu3
    xi = s - nu * math.acos(nu / z) - 0.25*math.pi
    return math.sqrt(2.0 / (math.pi * s)) * (A*math.cos(xi) + B*math.sin(xi))

@jit(nopython=True, fastmath=True)
def jv_interpolation(table, nu, x):
    """
    J_nu(x) for integer 1 <= nu <= table.shape[0]-1:
    tabulated (cubic) for x < 2*nu, Debye expansion for x >= 2*nu.
    """
    row = table[nu]
    scale = (row.shape[0] - 1) / (2.0 * nu)
    imax = row.shape[0] - 3
    nuf = float(nu)
    nu2 = nuf * nuf
    nu3 = nu2 * nuf
    xmax = 2.0 * nuf
    out = np.empty(x.shape[0])
    for j in range(x.shape[0]):
        z = x[j]
        if z < xmax:
            out[j] = _jv_table_scalar(row, scale, imax, z)
        else:
            out[j] = _jv_debye_scalar(nuf, nu2, nu3, z)
    return out


'''
@jit(nopython=True)
def interp_sin_int(xinterp, int_sin_table, int_cos_table, n, xeval):
    """
    Interpolation routine for the sine integral: \int x^n sin(omega x) dx
    """

    res = np.zeros(xeval.shape)
    for j, x in enumerate(xeval):
        log10_dx = np.log10(xinterp[1]) - np.log10(xinterp[0])
        x0_start = xinterp[0]
            
        ixstar = int((np.log10(x) - np.log10(x0_start)) / log10_dx) # for log-spaced
        xstar = xinterp[ixstar]
        Delta_x = x - xstar

        taylor = int_sin_table[n][ixstar]
        taylor += Delta_x * int_cos_table[n+1][ixstar]
        taylor += -Delta_x**2/2 * int_sin_table[n+2][ixstar]
        #taylor += -Delta_x**3/6 * xstar**3 * int_cos_table[ixstar]

        res[j] = taylor
    
    return res
'''