from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import scipy

import spuq.polyquad._polynomials as _p

class PolynomialFamily(object):
    """Abstract base for families of (orthogonal) polynomials"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def recurrence_coefficients(self, n):
        return NotImplemented

    @abstractmethod
    def get_structure_coefficient(self, a, b, c):
        """Return specific structure coefficient"""
        return NotImplemented

    def eval(self, n,  x):
        """Evaluate polynomial of degree ``n`` at points ``x``"""
        return _p.compute_poly(self.recurrence_coefficients, n, x)[-1]

    def get_coefficients(self, n):
        """Return coefficients of the polynomial with degree ``n`` of
        the family."""
        l = self.eval(n,  poly1d([1, 0]))
        return l.coeffs[::-1]

    def get_structure_coefficients(self, n):
        """Return structure coefficients of indices up to ``n``"""
        
        structcoeffs = getattr(self, "_structcoeffs", np.empty((0, 0, 0)))

        if n > structcoeffs.shape[0]:
            structcoeffs = np.array( 
                [[[self.get_structure_coefficient(a, b, c)
                   for a in xrange(n)]
                  for b in xrange(n)]
                 for c in xrange(n)])

        return structcoeffs[0:n, 0:n, 0:n]

    @abstractmethod
    def norm(self, n, sqrt=True):
        """Return norm of the ``n``-th degree polynomial."""
        return NotImplemented

    @property
    def normalised(self):
        """True if polynomials are normalised."""
        return False


class NormalisedPolynomialFamily(PolynomialFamily):
    """Wrapper that transforms an unnormalised family of orthogonal
    polynomials into normalised ones"""

    def __init__(self, family):
        self._family = family
        self.recurrence_coefficients = \
            _p.normalise_rc(_family.recurrence_coefficients)

    def norm(self, n, sqrt=True):
        """Return the norm of the `n`-th polynomial."""
        return 1.0

    def is_normalised(self):
        """Return True if polynomials are normalised."""
        return True


def normalise(family):
    return NormalisedPolynomialFamily(family)


class LegendrePolynomials(PolynomialFamily):

    def __init__(self, a=-1.0, b=1.0, normalised=False):
        self._a = a
        self._b = b
        self._normalised = normalised

    def recurrence_coefficients(self, n):
        return _p.rc_legendre(n)

    def norm(self, n, sqrt=True):
        """Returns the norm of polynomial"""
        if self._normalised:
            return 1
        return _p.sqnorm_legendre(n)

    @property
    def normalised(self):
        return self._normalised

    def get_structure_coefficient(self, a, b, c):
        return NotImplemented


class StochasticHermitePolynomials(PolynomialFamily):

    def __init__(self, mu=0.0, sigma=1.0, normalised=False):
        # currently nothing else is supported (coming soon however)
        assert mu == 0.0
        assert sigma == 1.0
        assert normalised == False

    def recurrence_coefficients(self, n):
        return _p.rc_stoch_hermite(n)

    def norm(self, n):
        """returns norm of polynomial"""
        return _p.sqnorm_stoch_hermite(n)

    def get_structure_coefficient(self, a, b, c):
        return _p.stc_stoch_hermite(a, b, c)
