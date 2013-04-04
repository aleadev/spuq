from __future__ import division
from abc import ABCMeta, abstractmethod, abstractproperty
import spuq.polyquad.polynomials.StochasticHermitePolynomials
import numpy as np
import itertools as it
from scipy.misc import factorial
from scipy.integrate import quad
from spuq.polyquad.polynomials import StochasticHermitePolynomials

class Covariance:
    __metaclass__ = ABCMeta
    
    def __init__(self, isisotropic=False, ishomogeneous=False):
        self._isisotropic = isisotropic
        self._ishomgeneous = ishomogeneous

    @property
    def isotropic(self):
        return self._isisotropic
    
    @property
    def homogeneous(self):
        return self._ishomogeneous
    
    def __call__(self, x, y):
        return self.evaluate(x, y)

    @abstractmethod
    def evaluate(self, r):
        raise NotImplementedError
    

class GaussianCovariance(Covariance):
    def __init__(self, sigma, a):
        super(True, True)
        self.sigma = sigma
        self.a = a
        
    def evaluate(self, x, y):
        r = ((x - y) ** 2).sum(axis=1)
        return self.sigma ** 2 * np.exp(-r / self.a ** 2)
    

class ExponentialCovariance(Covariance):
    def __init__(self, sigma, a):
        super(True, True)
        self.sigma = sigma
        self.a = a
        
    def evaluate(self, x, y):
        r = np.sqrt(((x - y) ** 2).sum(axis=1))
        return self.sigma ** 2 * np.exp(-r / self.a)

    
class TransformedCovariance(Covariance):
    def __init__(self, phi, KL, N):
        self.phi = phi
        self.N = N
        self._cov_gamma = self.eval_transformed_covariance(phi, KL.cov, KL.basis, N)
        self._r_alpha = self.eval_gpc_coefficients(KL, N)
        
    def eval_transformed_covariance(self, phi, cov_r, basis, N):
        # EZ (3.59)
        def phi_integrand(x, i, phi):
            return phi(x)*StochasticHermitePolynomials.eval(i, x)*np.exp(-x**2/2) / (np.sqrt(2*np.pi)*factorial(i))
        phii = [quad(phi_integrand, -np.Inf, np.Inf, args=(i, phi)) for i in range(N)]
        
        c4dof = basis.get_dof_coordinates()
        N = c4dof.shape[0]
        cov_gamma = np.ndarray(N, N)
        # find roots of polynomials for each pair of coordinates
        for i, j in it.product(range(N), repeat=2):
            c = [phii[0] - cov_r] + [factorial(i)*phii[i]**2 for i in range(1,N+1)]
            r = np.roots(c)
            cov_gamma[i,j] = r[0] 
        return cov_gamma, phii
        
    def eval_gpc_coeficents(self, alphas, KL, phii=None):
        # EZ (3.62)
        def binom(a):
            return factorial(np.sum(a))/np.prod(map(lambda x: float(factorial(x)), a))
        r = [binom(a)*phii[np.sum(a)]*KL.g**a for a in alphas]
        return r

    
class InterpolatedCovariance(Covariance):
    def __init__(self):
        pass

    
class LognormalTransformedCovariance(Covariance):
    def __init__(self):
        pass
        
