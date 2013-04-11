from __future__ import division
from abc import ABCMeta, abstractmethod, abstractproperty
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
        super(self.__class__, self).__init__(True, True)
        self.sigma = sigma
        self.a = a

    def evaluate(self, x, y):
        r = ((x - y) ** 2).T.sum(axis=0).T
        return self.sigma ** 2 * np.exp(-r / self.a ** 2)


class ExponentialCovariance(Covariance):
    def __init__(self, sigma, a):
        super(self.__class__, self).__init__(True, True)
        self.sigma = sigma
        self.a = a

    def evaluate(self, x, y):
        r = np.sqrt(((x - y) ** 2).T.sum(axis=0).T)
        return self.sigma ** 2 * np.exp(-r / self.a)


class TransformedCovariance(Covariance):
    def __init__(self, I, cov_r, phi, N):
        assert N <= KL.M
        self.I = I
        self.cov_r = cov_r
        self.phi = phi
        self.N = N
        self._phii = self.prepare_coefficients(phi, cov_r, N)

    def _prepare_coefficients(self, phi, cov_r, N):
        # EZ (3.55)
        def phi_integrand(x, i, phi, Hpoly):
            return phi(x) * Hpoly.eval(i, x) * np.exp(-x ** 2 / 2) / (np.sqrt(2 * np.pi) * factorial(i))
        Hpoly = StochasticHermitePolynomials()
        phii = [quad(phi_integrand, -np.Inf, np.Inf, args=(i, phi, Hpoly)) for i in range(N)]
        return phii

    def evaluate(self, x, y):
        assert x.shape[0] == y.shape[0]
        J = x.shape[0]
        cov_gamma = np.ndarray((J, J))
        # find roots of polynomials for each pair of coordinates
        for j, k in it.product(range(J), repeat=2):
            # EZ (3.59), evaluation of polynomial roots
            c = [phii[0] - cov_r(x[j], y[k])] + [factorial(i) * self.phii[i] ** 2 for i in range(1, self.N)]
            r = np.roots(c)
            
            # TODO: check existence/uniqueness of result
            
            print "RRRR", r, J
            
            cov_gamma[j, k] = r[0]
        return cov_gamma


class InterpolatedCovariance(Covariance):
    def __init__(self, cov_gamma, phi):
        pass

    def _prepare_interpolation(self):
        # evaluate phi(cov_gamma) at N points in [-1,1]

class LognormalTransformedCovariance(Covariance):
    def __init__(self, cov_r, mu, sigma):
        pass

    def evaluate(self, x, y):
        # analytisch -> Philipp Wähnert, Alex, Mike, Elmar
        pass
    