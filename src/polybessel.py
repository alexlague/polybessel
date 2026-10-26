###
# Main class for the integration of polynomials
# and spherical Bessel functions
##

import numpy as np
from scipy.special import eval_legendre, spherical_jn, jv
from numba import jit, vectorize, set_num_threads
from joblib import Parallel, delayed
import math

from integration import weights_clenshaw_curtis, nodes_clenshaw_curtis #clenshaw_curtis_any_interval #filon_sin_quad, filon_cos_quad
#from interpolation import Cheb_coeffs_dct
from interpolation import bessel_switch, Cheb_coeffs_dct, cheb_fit_adaptive, cheb_deriv_coeffs
#from single_bessel_recurrence import precompute_dI_eq33, hankel_transform_multi_l_eq33

from main_integration import main_integration_Bessel_single as MIBS
from main_integration import main_integration_Bessel_single_many as MIBS_many
from main_integration import main_integration_Bessel_product as MIBP
from main_integration import main_integration_Bessel_product_many as MIBP_many
from main_integration import main_integration_Bessel_product_many_sym as MIBP_many_sym

class PolyBessel:
    def __init__(self, max_order=100, spherical_bessel=False, n_proc=1, n_cheb_max=2048, n_cheb_min=16, n_interp=4096, tol=1e-6):
        """
        """
        #xs, ls, freqs, interp_spacing=0.03, n_proc=4, x_pad=0.7):
        #self.xs = xs
        #self.ls = ls
        #self.freqs = freqs
        
        #self.x_min = np.min(xs)
        #self.x_max = np.max(xs)
        #self.l_min = np.min(ls)
        #if spherical_bessel:
        #    self.l_max = max_order
        #else:
        self.nu_max = max_order

        self.n_cheb_max = n_cheb_max
        self.n_cheb_min = n_cheb_min
        self.n_interp = n_interp
        self.tol = tol
        #self.freq_min = np.min(freqs)
        #self.freq_max = np.max(freqs)
        self.spherical_bessel = spherical_bessel

        #self.interp_spacing = interp_spacing
        #self.x_pad = x_pad
        
        self.n_proc = n_proc
        set_num_threads(n_proc)
        # Creating reference arrays
        # skipping for now...
        #self.pre_compute_jl()
        #self.pre_compute_trig_int()
        #self.pre_compute_Pl()
        #self.dI0, self.dI1 = precompute_dI_eq33(self.xs, self.freqs, self.ls)
    

        ### Table Functions ###
        def make_w_levels(ncheb, nmin):
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
    
        def make_Bessel_table(nu_max, n_interp):
            """
            """
            Jv_int_table = np.zeros((nu_max+2, n_interp)) # need one more for the derivative
            
            # half integer offset for spherical Bessel
            off = 0.5 if self.spherical_bessel else 0.0
            
            for nui in range(nu_max+2):
                nui += off
                xnu = np.linspace(0.0, bessel_switch(nui), n_interp)
                Jv_int_table[int(nui), :] = jv(nui, xnu)
            
            return Jv_int_table

        self.w_levels = make_w_levels(self.n_cheb_max, self.n_cheb_min)
        self.Jv_int_table = make_Bessel_table(self.nu_max, self.n_interp)


    ### Integration Functions ###

    def compute_individual_single_bessel(self, f, nu, omega, a, b):
        """
        Integral of f(x) B_nu(omega x) on [a, b], with B = J (cylindrical) or j (spherical).
        """
        if nu < 0:
            raise ValueError("Negative orders not supported.")
        if nu == 0 and not self.spherical_bessel:
            raise ValueError("Order 0 not yet supported for non-spherical Bessel functions.")
        if nu > self.nu_max:
            raise ValueError("Orders above max_order=%d not supported." % self.nu_max)
        if omega <= 0:
            raise ValueError("Frequency must be positive.")

        if self.spherical_bessel:
            nu = nu + 0.5
            k = np.sqrt(np.pi/2/omega)  # j_n(wx) = sqrt(pi/(2wx)) J_{n+1/2}(wx)
            g = lambda x: k*f(x)/np.sqrt(x)
        else:
            g = f
        c = cheb_fit_adaptive(g, a, b)
        dc = cheb_deriv_coeffs(c, a, b)
        ddc = cheb_deriv_coeffs(dc, a, b)
        u = nodes_clenshaw_curtis(self.n_cheb_max, -1, 1)

        return MIBS(c, dc, ddc, u, self.w_levels, self.Jv_int_table, float(nu), float(omega),
                    float(a), float(b), self.n_cheb_max, self.tol)

    def compute_many_single_bessel(self, f, nu_list, omega_list, a, b):
        """
        Integral of f(x) B_nu(omega x) on [a, b] for every combination of the inputs,
        with B = J (cylindrical) or j (spherical).
        Returns an array of shape (len(nu_list), len(omega_list)).
        """
        nus = np.atleast_1d(np.asarray(nu_list, dtype=np.float64))
        oms = np.atleast_1d(np.asarray(omega_list, dtype=np.float64))

        if nus.min() < 0:
            raise ValueError("Negative orders not supported.")
        if nus.min() == 0 and not self.spherical_bessel:
            raise ValueError("Order 0 not yet supported for non-spherical Bessel functions.")
        if nus.max() > self.nu_max:
            raise ValueError("Orders above max_order=%d not supported." % self.nu_max)
        if oms.min() <= 0:
            raise ValueError("Frequencies must be positive.")

        if self.spherical_bessel:
            # fit f/sqrt(x) once (independent of omega), apply sqrt(pi/(2 omega)) afterwards
            nus = nus + 0.5
            g = lambda x: f(x)/np.sqrt(x)
        else:
            g = f
        c = cheb_fit_adaptive(g, a, b)
        dc = cheb_deriv_coeffs(c, a, b)
        ddc = cheb_deriv_coeffs(dc, a, b)
        u = nodes_clenshaw_curtis(self.n_cheb_max, -1, 1)

        out = MIBS_many(c, dc, ddc, u, self.w_levels, self.Jv_int_table, nus, oms,
                        float(a), float(b), self.n_cheb_max, self.tol)

        if self.spherical_bessel:
            out *= np.sqrt(np.pi/2/oms)[None, :]

        return out
    
    def compute_individual_double_bessel(self, f, nu1, omega1, nu2, omega2, a, b):
        """
        """
        if min(nu1, nu2) < 0:
            raise ValueError("Negative orders not supported.")
        if min(nu1, nu2) == 0 and not self.spherical_bessel:
            raise ValueError("Order 0 not yet supported for non-spherical Bessel functions.")

        if self.spherical_bessel:
            nu1 = nu1+0.5
            nu2 = nu2+0.5
            k = np.pi/2/np.sqrt(omega1*omega2)  # j_n(wx) = sqrt(pi/(2wx)) J_{n+1/2}(wx)
            g = lambda x: k*f(x)/x
        else:
            g = f
        c = cheb_fit_adaptive(g, a, b)
        dc = cheb_deriv_coeffs(c, a, b)
        u = nodes_clenshaw_curtis(self.n_cheb_max, -1, 1)

        return MIBP(c, dc, u, self.w_levels, self.Jv_int_table, nu1, omega1, nu2, omega2, a, b, ncheb=self.n_cheb_max, tol=self.tol)

    def compute_many_double_bessel(self, f, nu1_list, omega1_list, nu2_list, omega2_list, a, b, use_symmetry=True):
        """
        Integral of f(x) B_nu1(omega1 x) B_nu2(omega2 x) on [a, b] for every combination
        of the inputs, with B = J (cylindrical) or j (spherical).
        Returns an array of shape (len(nu1_list), len(omega1_list), len(nu2_list), len(omega2_list)).

        use_symmetry: if nu1_list == nu2_list and omega1_list == omega2_list, integrate only
        one of each pair (nu1, om1, nu2, om2) / (nu2, om2, nu1, om1) and mirror the result
        (about half the work).

        ignore_even / ignore_odd: if True, combinations with nu1 + nu2 even / odd are not
        integrated and set to 0 (e.g. the parity rule of a 3j symbol with m1 = m2 = m3 = 0
        at fixed L). Parity of the orders as given, before any spherical shift.
        Setting both to True raises a ValueError, as nothing would be computed.
        """
        nu1s = np.atleast_1d(np.asarray(nu1_list, dtype=np.float64))
        nu2s = np.atleast_1d(np.asarray(nu2_list, dtype=np.float64))
        om1s = np.atleast_1d(np.asarray(omega1_list, dtype=np.float64))
        om2s = np.atleast_1d(np.asarray(omega2_list, dtype=np.float64))

        if min(nu1s.min(), nu2s.min()) < 0:
            raise ValueError("Negative orders not supported.")
        if min(nu1s.min(), nu2s.min()) == 0 and not self.spherical_bessel:
            raise ValueError("Order 0 not yet supported for non-spherical Bessel functions.")
        if max(nu1s.max(), nu2s.max()) > self.nu_max:
            raise ValueError("Orders above max_order=%d not supported." % self.nu_max)
        if min(om1s.min(), om2s.min()) <= 0:
            raise ValueError("Frequencies must be positive.")

        if self.spherical_bessel:
            # j_n(wx) = sqrt(pi/(2wx)) J_{n+1/2}(wx): fit f/x once (independent of omega)
            # and apply the omega-dependent factor pi/(2 sqrt(w1 w2)) to the results
            nu1s = nu1s + 0.5
            nu2s = nu2s + 0.5
            g = lambda x: f(x)/x
        else:
            g = f
        c = cheb_fit_adaptive(g, a, b)
        dc = cheb_deriv_coeffs(c, a, b)
        u = nodes_clenshaw_curtis(self.n_cheb_max, -1, 1)

        symmetric = use_symmetry and np.array_equal(nu1s, nu2s) and np.array_equal(om1s, om2s)
        if symmetric:
            out = MIBP_many_sym(c, dc, u, self.w_levels, self.Jv_int_table, nu1s, om1s,
                                float(a), float(b), self.n_cheb_max, self.tol)
        else:
            out = MIBP_many(c, dc, u, self.w_levels, self.Jv_int_table, nu1s, om1s, nu2s, om2s,
                            float(a), float(b), self.n_cheb_max, self.tol)

        if self.spherical_bessel:
            out *= (np.pi/2) / np.sqrt(om1s[None, :, None, None] * om2s[None, None, None, :]) #scaled after the integrals instead

        return out
    
    '''
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