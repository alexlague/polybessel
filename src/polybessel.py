###
# Main class for the integration of polynomials
# and spherical Bessel functions
##

import numpy as np
from scipy.special import eval_legendre, spherical_jn
from numba import jit, vectorize
from joblib import Parallel, delayed

from integration import clenshaw_curtis_any_interval #filon_sin_quad, filon_cos_quad
from interpolation import interp_jl

class PolyBessel:
    def __init__(self, x, l, freqs, interp_spacing=0.03, nproc=4, xpad=0.7, M=20):
        self.lmax = lmax
        self.N = N
        self.k_array = k_array
        self.kp_array = kp_array
        self.xmin = xmin
        self.xmax = xmax
        self.interp_spacing = interp_spacing
        #self.npts_leg = npts_leg
        self.nproc = nproc
        self.xpad = xpad

        self.kmin = np.min([np.min(self.k_array), np.min(self.kp_array)])
        self.kmax = np.max([np.max(self.k_array), np.max(self.kp_array)])
        
        # Creating reference arrays
        self.pre_compute_jl()
        #self.pre_compute_trig_int()
        #self.pre_compute_Pl()


    ### Table Functions ###
    
    def pre_compute_jl(self):
        """
        """
        lower_lim = self.kmin * self.xmin * (1-self.xpad)
        upper_lim = self.kmax * self.xmax * (1+self.xpad)

        self.npts_sb = int((upper_lim-lower_lim)/self.interp_spacing)
        
        self.xinterp = np.geomspace(lower_lim, upper_lim, self.npts_sb)
        
        self.jl_table = Parallel(n_jobs=self.nproc)(delayed(spherical_jn)(l, self.xinterp) for l in range(self.lmax+2))
        self.jl_table = np.array(self.jl_table)
        
        return

    def pre_compute_Pl(self):
        """
        MAY BE NEEDED ONLY FOR LMAX
        """
        #self.leg_x, self.leg_w = np.polynomial.legendre.leggauss(self.npts_leg)
        #self.Pl_interp = Parallel(n_jobs=nproc)(delayed(eval_legendre)(l, self.leg_x) for l in range(self.lmax+self.l_pad+1))

        # Chebyshev-Lobatto points
        N = self.npts_leg
        j = np.arange(N + 1)
        self.leg_x = np.cos(np.pi * j / N)
        self.Pl_at_lmax = eval_legendre(self.lmax, self.leg_x)
        
        return

    def pre_compute_trig_int(self):
        """
        For the integrals of x^n sin(ax) or x^n cos(ax)
        CAN BE DONE ANALYTICALLY
        """
        
        self.trig_alpha_array = np.geomspace(0.2*self.kmin, 5*self.kmax, 100_000) # ADD THIS AS PARAMETER
        a = self.xmin
        b = self.xmax
        o = 300
        omega = self.trig_alpha_array
        
        self.sin_int_interp = Parallel(n_jobs=self.nproc)(delayed(filon_sin_quad)(n, a, b, omega, o) for n in range(self.N+1))
        self.cos_int_interp = Parallel(n_jobs=self.nproc)(delayed(filon_cos_quad)(n, a, b, omega, o) for n in range(self.N+1))

        self.sin_int_interp_neg1 = filon_sin_quad(-1, a, b, omega, o)
        self.cos_int_interp_neg1 = filon_cos_quad(-1, a, b, omega, o)
        
        return

    ### Interpolation Functions ###

    #MOVED
    
    ### Integration Functions ###
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
    '''
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