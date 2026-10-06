##
# Functions for integral routines
##

import numpy as np
from scipy.fft import dct
from numba import jit, vectorize

# Filon quadrature for sine/cosine integration


@vectorize
def filon_sin_quad(n, a, b, omega, o_intervals):
    """
    Evaluates the integral of f(x) * sin(omega * x) from a to b 
    using Filon's Quadrature rule.
    
    Parameters:
        f           : Callable function f(x)
        a, b        : Integration limits (floats)
        omega       : Frequency coefficient (float)
        o_intervals : Number of double-subintervals (integer). 
                      Total sampling points N = 2 * o_intervals + 1.
    """
    # Step size calculation
    n_panels = 2 * o_intervals
    h = (b - a) / n_panels
    theta = omega * h
    
    # Precompute trigonometric terms needed for Filon coefficients
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    # Handle the limit when theta approaches 0 (Standard Simpson's Rule limit)
    if abs(theta) < 1e-8:
        alpha = 0.0
        beta = 2.0 / 3.0
        gamma = 4.0 / 3.0
    else:
        theta2 = theta ** 2
        theta3 = theta ** 3
        # Filon coefficients for parabolic interpolation
        alpha = (theta2 + theta * sin_t * cos_t - 2.0 * (sin_t**2)) / theta3
        beta  = (2.0 * theta * (1.0 + (cos_t**2)) - 4.0 * sin_t * cos_t) / theta3
        gamma = 4.0 * (sin_t - theta * cos_t) / theta3

    # Generate grid nodes
    x = np.linspace(a, b, n_panels + 1)
    fx = x**n
    
    # Separate points into even indices, odd indices, and boundary terms
    # Filon formulation pairs elements across adjacent subintervals
    c_vals = fx * np.cos(omega * x)
    s_vals = fx * np.sin(omega * x)
    
    # Calculate regular boundary patterns
    C_even = np.sum(c_vals[2:-2:2]) + 0.5 * (c_vals[0] + c_vals[-1])
    C_odd  = np.sum(c_vals[1::2])
    
    S_even = np.sum(s_vals[2:-2:2]) + 0.5 * (s_vals[0] + s_vals[-1])
    S_odd  = np.sum(s_vals[1::2])
    
    # Compute the final integration summation
    term_boundary = alpha * (fx[0] * np.cos(omega * a) - fx[-1] * np.cos(omega * b))
    term_even = beta * S_even
    term_odd = gamma * S_odd
    
    integral = h * (term_boundary + term_even + term_odd)
    return integral

@vectorize
def filon_cos_quad(n, a, b, omega, o_intervals):
    """
    Evaluates the integral of f(x) * cos(omega * x) from a to b 
    using Filon's Quadrature rule.
    
    Parameters:
        f           : Callable function f(x)
        a, b        : Integration limits (floats)
        omega       : Frequency coefficient (float)
        o_intervals : Number of double-subintervals (integer). 
                      Total sampling points N = 2 * o_intervals + 1.
    """
    # Step size calculation
    n_panels = 2 * o_intervals
    h = (b - a) / n_panels
    theta = omega * h
    
    # Precompute trigonometric terms needed for Filon coefficients
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    
    # Handle the limit when theta approaches 0 (Standard Simpson's Rule limit)
    if abs(theta) < 1e-8:
        alpha = 0.0
        beta = 2.0 / 3.0
        gamma = 4.0 / 3.0
    else:
        theta2 = theta ** 2
        theta3 = theta ** 3
        # Filon coefficients for parabolic interpolation
        alpha = (theta2 + theta * sin_t * cos_t - 2.0 * (sin_t**2)) / theta3
        beta  = (2.0 * theta * (1.0 + (cos_t**2)) - 4.0 * sin_t * cos_t) / theta3
        gamma = 4.0 * (sin_t - theta * cos_t) / theta3
    # Generate grid nodes
    x = np.linspace(a, b, n_panels + 1)
    fx = x**n
    
    # Separate points into even indices, odd indices, and boundary terms
    # Filon formulation pairs elements across adjacent subintervals
    c_vals = fx * np.cos(omega * x)
    s_vals = fx * np.sin(omega * x)
    
    # Calculate regular boundary patterns
    C_even = np.sum(c_vals[2:-2:2]) + 0.5 * (c_vals[0] + c_vals[-1])
    C_odd  = np.sum(c_vals[1::2])
    
    S_even = np.sum(s_vals[2:-2:2]) + 0.5 * (s_vals[0] + s_vals[-1])
    S_odd  = np.sum(s_vals[1::2])
    
    # Compute the final integration summation
    term_boundary = alpha * (fx[-1] * np.sin(omega * b) - fx[0] * np.sin(omega * a))
    term_even = beta * C_even
    term_odd = gamma * C_odd
    
    integral = h * (term_boundary + term_even + term_odd)
    return integral

# REPLACED IN TRIG MODULE
# Sine integrals
@vectorize
def sine_integrals_analytic(n, omega, a, b):
    out = [0.0, 0.0]
    for i, x in enumerate([a, b]):
        if n == 0:
            out[i] = -np.cos(omega*x)/omega
        elif n == 1:
            out[i] = (-omega*x*np.cos(omega*x) 
                      + np.sin(omega*x))/omega**2
        elif n == 2:
            out[i] = ((2 - omega**2*x**2)*np.cos(omega*x) 
                      + 2*omega*x*np.sin(omega*x))/omega**3
        elif n == 3:
            out[i] = (-omega*x *(-6 + omega**2 *x**2) *np.cos(omega*x) 
                      + 3* (-2 + omega**2 * x**2)*np.sin(omega*x))/omega**4
        elif n == 4:
            out[i] = (-((24 - 12*omega**2*x**2 + omega**4 *x**4)*np.cos(omega*x)) 
                      + 4*omega*x*(-6 + omega**2*x**2)*np.sin(omega*x))/omega**5
        elif n == 5:
            out[i] = (-omega*x *(120 - 20*omega**2*x**2 + omega**4*x**4)* np.cos(omega*x) 
                      + 5*(24 - 12* omega**2*x**2 + omega**4 *x**4)*np.sin(omega*x))/omega**6
        elif n == 6:
            out[i] = ((720 - 360*omega**2*x**2 + 30*omega**4*x**4 - omega**6*x**6)* np.cos(omega*x) 
                      + 6* omega*x* (120 - 20* omega**2*x**2 + omega**4 *x**4)*np.sin(omega*x))/omega**7
    
    return out[1] - out[0]


@jit(nopython=True)
def integral_from_c(c, N, a, b):
    """
    Separate from the DCT function
    since numba does not allow scipy's dct
    in jitted functions
    """
    integral = c[0]
    for n in range(2, N, 2):
        integral += 2 * c[n] / (1 - n**2)
    if N % 2 == 0:
        integral += c[N] / (1 - N**2)
    return (b - a) / 2 * integral

def npts_clenshaw_curtis(k1, k2, a, b):
    """
    """
    opt = (k1+k2)*(b-a)/np.pi
    opt = 2**(int(np.log10(opt)/0.3010299956639812) + 1) #find nearest power of 2 above minimum
    return opt

def nodes_clenshaw_curtis(N, a, b):
    """
    """
    # Chebyshev-Lobatto points on [-1, 1]
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)

    # Map points from [-1, 1] to [a, b]
    x = (a + b) / 2 + (b - a) / 2 * t
    
    return x

def clenshaw_curtis_any_interval(fx, N, a, b):
    """
    fx is the integrand evaluated at the (scaled) Chebyshev-Lobatto points
    Use: nodes_clenshaw_curtis for the number of nodes and
    nodes_clenshaw_curtis for their location
    """

    # Chebyshev coefficients
    c = dct(fx, type=1) / N

    return integral_from_c(c, N, a, b)

def weights_clenshaw_curtis(Ncc):
    c = np.zeros(Ncc + 1)
    c[0] = 1.0
    for k in range(2, Ncc + 1, 2):
        c[k] = 2.0 / (1.0 - k**2)
            
    # 3. Adjust boundary terms for standard DCT-I scaling conventions
    c[Ncc] /= 2.0
    c[1:Ncc] /= 2.0
        
    # 4. Compute weights via Type-I DCT
    weights = 2.0 * dct(c, type=1) / Ncc
        
    # 5. Apply the endpoint corrections (c_j factor for boundary points)
    weights[0] /= 2.0
    weights[Ncc] /= 2.0
    return weights

# pyFFTW interface 

import pyfftw
_plan_cache = {}

def _get_dct1_plan(n):
    """Cache an in-place FFTW REDFT00 (DCT-I) plan for array length n."""
    if n not in _plan_cache:
        arr = pyfftw.empty_aligned(n, dtype='float64')
        plan = pyfftw.FFTW(arr, arr, direction='FFTW_REDFT00',
                            flags=('FFTW_ESTIMATE',))
        _plan_cache[n] = (arr, plan)
    return _plan_cache[n]


def clenshaw_curtis_fftw(fx, N, a, b):
    """
    """
    # Chebyshev coefficients via pyfftw (in-place), replacing scipy's dct
    arr, plan = _get_dct1_plan(N + 1)
    arr[:] = fx
    plan()                 # transform executes in-place: arr now holds the DCT-I
    c = arr / N
    
    return integral_from_c(c, N, a, b)