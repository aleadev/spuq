from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import scipy

import spuq.polynomials._polynomials as _p

class PolynomialFamily(object):
    """abstract base for polynomials"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self.structcoeffs = np.empty((0, 0, 0))

    @abstractmethod
    def recurrence_coefficients(self, n):
        return NotImplemented

    @abstractmethod
    def get_structure_coefficient(self, a, b, c):
        """return specific structure coefficient"""
        return NotImplemented

    def eval(self, n,  x):
        """Evaluate polynomial of degree n at points x"""
        return _p.compute_poly(self.recurrence_coefficients, n, x)[-1]

    def get_coefficients(self, n):
        """return coefficients of polynomial"""
        l = self.eval(n,  poly1d([1, 0]))
        return l.coeffs[::-1]

    def get_structure_coefficients(self, n):
        """return structure coefficients of indices up to n"""
        if n > self.structcoeffs.shape[0]:
            j = self.structcoeffs[0]
            self.structcoeffs.resize((n, n, n))
            for a in xrange(n):
                for b in xrange(n):
                    for c in xrange(n):
                        self.structcoeffs[a, b, c] = \
                            self.get_structure_coefficient(a, b, c)
        return self.structcoeffs[0:n, 0:n, 0:n]

    @abstractmethod
    def norm(self, n):
        """returns norm of polynomial"""
        return NotImplemented

    def is_normalised(self):
        """return True if polynomials are normalised"""
        return False


class NormalisedPolynomialFamily(PolynomialFamily):
    """Wrapper that transforms an unnormalised family of orthogonal
    polynomials into normalised ones"""

    def __init__(self, family):
        self._family = family
        self.recurrence_coefficients = \
            _p.normalise_rc(_family.recurrence_coefficients)

    def norm(self, n):
        """Return the norm of the `n`-th polynomial."""
        return 1.0

    def is_normalised(self):
        """Return True if polynomials are normalised."""
        return True


def normalise(family):
    return NormalisedPolynomialFamily(family)


class LegendrePolynomials(PolynomialFamily):

    def recurrence_coefficients(self, n):
        return _p.rc_legendre(n)

    def norm(self, n, sqrt=True):
        """returns norm of polynomial"""
        return _p.sqnorm_legendre(n)

    def get_structure_coefficient(self, a, b, c):
        return NotImplemented


class StochasticHermitePolynomials(PolynomialFamily):

    def recurrence_coefficients(self, n):
        return _p.rc_stoch_hermite(n)

    def norm(self, n):
        """returns norm of polynomial"""
        return _p.sqnorm_stoch_hermite(n)

    def get_structure_coefficient(self, a, b, c):
        return _p.stc_stoch_hermite(a, b, c)




