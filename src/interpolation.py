##
# Interpolation routines
##

import numpy as np
from numba import jit, vectorize

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