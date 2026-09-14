###
# Main class for the integration of polynomials
# and spherical Bessel functions
##

import numpy as np
from scipy.special import eval_legendre, spherical_jn
from numba import jit, vectorize, set_num_threads
from joblib import Parallel, delayed

#from integration import clenshaw_curtis_any_interval #filon_sin_quad, filon_cos_quad
from interpolation import interp_jl
from single_bessel_recurrence import precompute_dI_eq33, hankel_transform_multi_l_eq33

class PolyBessel:
    def __init__(self, xs, ls, freqs, interp_spacing=0.03, n_proc=4, x_pad=0.7):
        """
        """
        self.xs = xs
        self.ls = ls
        self.freqs = freqs
        
        self.x_min = np.min(xs)
        self.x_max = np.max(xs)
        self.l_min = np.min(ls)
        self.l_max = np.max(ls)
        self.freq_min = np.min(freqs)
        self.freq_max = np.max(freqs)

        self.interp_spacing = interp_spacing
        self.x_pad = x_pad
        
        self.n_proc = n_proc
        set_num_threads(n_proc)
        # Creating reference arrays
        # skipping for now...
        #self.pre_compute_jl()
        #self.pre_compute_trig_int()
        #self.pre_compute_Pl()
        self.dI0, self.dI1 = precompute_dI_eq33(self.xs, self.freqs, self.ls)


    ### Table Functions ###
    
    def pre_compute_jl(self):
        """
        """
        lower_lim = self.freq_min * self.x_min * (1-self.x_pad)
        upper_lim = self.freq_max * self.x_max * (1+self.x_pad)

        self.npts_sb = int((upper_lim-lower_lim)/self.interp_spacing)
        
        self.x_interp = np.geomspace(lower_lim, upper_lim, self.npts_sb)
        
        self.jl_table = Parallel(n_jobs=self.nproc)(delayed(spherical_jn)(l, self.x_interp) for l in range(self.l_max+2))
        self.jl_table = np.array(self.jl_table)
        
        return

    def pre_compute_Pl(self):
        """
        WILL BE USEFUL WHEN CHANGING ENDPOINTS
        """
        #self.leg_x, self.leg_w = np.polynomial.legendre.leggauss(self.npts_leg)
        #self.Pl_interp = Parallel(n_jobs=nproc)(delayed(eval_legendre)(l, self.leg_x) for l in range(self.lmax+self.l_pad+1))

        # Chebyshev-Lobatto points
        N = self.npts_leg
        j = np.arange(N + 1)
        self.leg_x = np.cos(np.pi * j / N)
        self.Pl_at_l_max = eval_legendre(self.l_max, self.leg_x)
        
        return

    
    
    ### Integration Functions ###
    def integrate_single_bessel(self, fx):
        """
        Compute the integral \int_a^b f(x) j_\ell(\omega x) dx
        for omega in frequencies array and l values in list.
        The endpoints are determined by the endpoints of the x array.
        
        The first evaluation runs table computation scripts and compilation
        of numba functions

        input: fx array of same shape as x

        output: array of shape (len(l), len(freqs))
        """
            
        
        return hankel_transform_multi_l_eq33(self.xs, fx, self.freqs, self.ls, self.dI0, self.dI1)


    
    '''
    def K_rec_sph_terms(self, n, l, x, k, kprime):
        """
        Replace with interpolated spherical_jn
        """

        out = kprime * spherical_jn(l-1, k*x) * spherical_jn(l, kprime*x)
        out += k * spherical_jn(l, k*x) * spherical_jn(l-1, kprime*x)
        out *= -x**n
        out += (2-n) * x**(n-1) * spherical_jn(l-1, k*x) * spherical_jn(l-1, kprime*x)
        out *= 0.5/k/kprime
    
        return out

    def inhomog_term(self, n, l, k, kprime, K_integrals):
        """
        """
    
        out = (n-2) * (n+2*l-3) * K_integrals[l-1][n-2]
        out *= 0.5/k/kprime
        out += (self.K_rec_sph_terms(n, l, xmax, k, kprime) - self.K_rec_sph_terms(n, l, xmin, k, kprime))
        
        return out
    
    @vectorize
    def compute_K_rec(self, k, kprime):
        """
        Main recurrence function for Kl
        Starts at lmax -> 0
        Needs K integrals for all l at n = 0, 1 and for all n at l=lmax
        
        TO-DO: Decide what is cached and what isn't
        """
        K_integrals = np.zeros((lmax+1, nmax+1))
        b = (k**2+kprime**2)/2/k/kprime
        
        for nj in range(3, nmax+1):
            for li in range(lmax, -1, -1):
                inhom = self.inhomog_term(nj, li, k, kprime, K_integrals)
                K_integrals[li-1][nj] = (K_integrals[li][nj] - inhom) / b

        return
    '''