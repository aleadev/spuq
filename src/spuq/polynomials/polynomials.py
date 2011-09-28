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
        if n == 0:
            # ensure return value has the right format
            return 0 * x + 1
        else:
            h1, h0 = 1, 0
            for i in xrange(0, n):
                (a, b, c) = self.recurrence_coefficients(i)
                h1, h0 = (a + b * x) * h1 - c * h0, h1
            return h1

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


class OrthogonalPolynomialFamily(PolynomialFamily):
    """Base class for """
    @abstractmethod
    def norm(self, n):
        """returns norm of polynomial"""
        return NotImplemented

    def is_normalised(self):
        """return True if polynomials are normalised"""
        return False


class NormalisedPolynomialFamily(OrthogonalPolynomialFamily):
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


class LegendrePolynomials(OrthogonalPolynomialFamily):

    def recurrence_coefficients(self, n):
        return _p.rc_legendre(n)

    def norm(self, n, sqrt=True):
        """returns norm of polynomial"""
        return _p.sqnorm_legendre(n)

    def get_structure_coefficient(self, a, b, c):
        return NotImplemented


class StochasticHermitePolynomials(OrthogonalPolynomialFamily):

    def recurrence_coefficients(self, n):
        return _p.rc_stoch_hermite(n)

    def norm(self, n):
        """returns norm of polynomial"""
        return _p.sqnorm_stoch_hermite(n)

    def get_structure_coefficient(self, a, b, c):
        n = max((a, b, c))
        s = a + b + c
        if bool(s % 2) or a <= b + c or b <= a + c or c <= a + b:
            c = 0
        else:
            s /= 2
            fac = scipy.factorial
            c = (fac(s - a) * fac(s - b) * fac(s - c) /
                 (fac(a) * fac(b) * fac(c)))




