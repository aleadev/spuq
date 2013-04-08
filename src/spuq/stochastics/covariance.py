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
    def __init__(self, I, phi, KL, N):
        assert N <= KL.M
        self.phi = phi
        self.N = N
        self.I = I
        self._cov_gamma, phii = self.prepare_transformed_covariance(phi, KL.cov, KL.basis, N)
        self._r_alpha = self.eval_gpc_coefficients(I, KL, phii)

    def prepare_transformed_covariance(self, phi, cov_r, basis, N):
        # EZ (3.55)
        def phi_integrand(x, i, phi, Hpoly):
            return phi(x) * Hpoly.eval(i, x) * np.exp(-x ** 2 / 2) / (np.sqrt(2 * np.pi) * factorial(i))
        Hpoly = StochasticHermitePolynomials()
        # TODO: quadrature does not seem to be finite on \R with this integrand
#        phii = [quad(phi_integrand, -np.Inf, np.Inf, args=(i, phi, Hpoly)) for i in range(N)]
        phii = [quad(phi_integrand, -1, 1, args=(i, phi, Hpoly))[0] for i in range(N)]

        print "XXXXXX", phii

        c4dof = basis.get_dof_coordinates()
        J = c4dof.shape[0]
        cov_gamma = np.ndarray((J, J))
        # find roots of polynomials for each pair of coordinates
        for i, j in it.product(range(J), repeat=2):
            # EZ (3.59), evaluation of polynomial roots
            c = [phii[0] - cov_r(c4dof[i], c4dof[j])] + [factorial(i) * phii[i] ** 2 for i in range(1, N)]
            r = np.roots(c)
            # TODO: check existence/uniqueness of result
            
            print "RRRR", r, J
            
            cov_gamma[i, j] = r[0]
        return cov_gamma, phii

    def eval_gpc_coefficients(self, alphas, KL, phii, project_onto_basis=None):
        # evaluate pce coefficients from (optionally projected) KL expansion
        def binom(a):
            if a.order == 0: return 0
            return factorial(a.order) / np.prod(map(lambda x: float(factorial(x)), a.as_array))
        Balphas = [(binom(a), a) for a in alphas]
        if project_onto_basis is not None:
            g = [project_onto_basis.project_onto(KL.gi) for gi in KL.g]
        else:
            g = KL.g
        # EZ (3.68)
        r = [lambda x: A * phii[a.order] * g(x) ** a for A, a in Balphas]
        return r
        # TODO: PCE/GPC class?!

    def evaluate(self, x, y):
        r = np.sqrt(((x - y) ** 2).T.sum(axis=0).T)
        # TODO: what needs to be done here? interpolation?
        pass


class InterpolatedCovariance(Covariance):
    def __init__(self):
        pass


class LognormalTransformedCovariance(Covariance):
    def __init__(self):
        pass

