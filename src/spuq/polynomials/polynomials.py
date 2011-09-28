from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy import factorial


class PolynomialFamily(object):
    """abstract base for polynomials"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self.structcoeffs = np.empty((0, 0, 0))

    @abstractmethod
    def eval(self, n,  x):
        """evaluate polynomial of degree n at points x"""
        return NotImplemented

    @abstractmethod
    def get_structure_coefficient(self, a, b, c):
        """return specific structure coefficient"""
        return NotImplemented

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

    def get_measure(self):
        """Return the measure underlying the scalar product"""
        return NotImplemented


class Measure(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def support(self):
        """Return an interval containing the support"""
        pass

    @abstractproperty
    def is_discrete(self):
        pass


class NormalisedPolynomialFamily(OrthogonalPolynomialFamily):
    """Wrapper that transforms an unnormalised family of orthogonal
    polynomials into normalised ones"""

    def __init__(self, family):
        self._family = family


def normalise(family):
    return NormalisedPolynomialFamily(family)


class LegendrePolynomials(OrthogonalPolynomialFamily):

    def recurrence_coefficients(self, n):
        pass

    def eval(self, n,  x):
        if n == 0:
            return 0 * x + 1        # ensure return value has the right format
        elif n == 1:
            return x
        else:
            h0 = 1
            h1 = x
            for i in xrange(2, n + 1):
                h1, h0 = (2 * i - 1.0) * x * h1 / i - (i - 1.0) * h0 / i, h1
            return h1

    def norm(self, n):
        """returns norm of polynomial"""
        return NotImplemented

    def get_structure_coefficient(self, a, b, c):
        return NotImplemented


class StochasticHermitePolynomials(OrthogonalPolynomialFamily):

    def eval(self, n,  x):
        if n == 0:
            return 0 * x + 1        # ensure return value has the right format
        elif n == 1:
            return x
        else:
            h0 = 1
            h1 = x
            for i in xrange(2, n + 1):
                h1, h0 = x * h1 - (i - 1) * h0, h1
            return h1

    def norm(self, n):
        """returns norm of polynomial"""
        return NotImplemented

    def get_structure_coefficient(self, a, b, c):
        n = max((a, b, c))
        s = a + b + c
        if bool(s % 2) or a <= b + c or b <= a + c or c <= a + b:
            c = 0
        else:
            s /= 2
            c = (factorial(s - a) * factorial(s - b) * factorial(s - c) /
                 (factorial(a) * factorial(b) * factorial(c)))




