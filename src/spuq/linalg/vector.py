from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from spuq.linalg.basis import Basis, CanonicalBasis, check_basis
from spuq.utils import strclass
from spuq.utils.type_check import takes, returns, anything, optional, list_of


class Vector(object):
    """Abstract base class for vectors which consist of a coefficient
    vector and an associated basis"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def basis(self):
        """Return basis of this vector"""
        return NotImplemented

    @abstractproperty
    def coeffs(self):
        """Return cofficients of this vector w.r.t. the basis"""
        return NotImplemented

    @abstractmethod
    def as_array(self):
        return NotImplemented

    @abstractmethod
    def __add__(self, other):
        """Compute the sum of two vectors."""
        return NotImplemented

    def __sub__(self, other):
        """Compute the difference between two vectors."""
        return self + (-1.0 * other)

    @abstractmethod
    def __mul__(self, other):
        """Compute the product of this vector with a scalar."""
        return NotImplemented

    def __rmul__(self, other):
        """Compute the product of this vector with a scalar."""
        return self.__mul__(other)

    @abstractmethod
    def __eq__(self, other):
        """Compare vectors for equality.

        Note that vectors are only considered equal when they have
        exactly the same type.
        """
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<%s basis=%s, coeffs=%s>" % \
               (strclass(self.__class__), self.basis, self.coeffs)


class FlatVector(Vector):
    """A vector classed based on the numpy array"""

    @takes(anything, (np.ndarray, list_of((int,float))), optional(Basis))
    def __init__(self, coeffs, basis=None):
        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(coeffs, dtype=float)
        if basis is None:
            basis = CanonicalBasis(coeffs.shape[0])
        assert(basis.dim == coeffs.shape[0])
        self._coeffs = coeffs
        self._basis = basis

    @property
    def basis(self):
        return self._basis

    @property
    def coeffs(self):
        return self._coeffs

    def as_array(self):
        return self.coeffs

    def _create_copy(self, coeffs):
        return self.__class__(coeffs, self.basis)

    @takes(anything, "FlatVector")
    def __add__(self, other):
        check_basis(self.basis, other.basis)
        return self._create_copy(self.coeffs + other.coeffs)

    @takes(anything, "FlatVector")
    def __sub__(self, other):
        check_basis(self.basis, other.basis)
        return self._create_copy(self.coeffs - other.coeffs)

    @takes(anything, (float, int))
    def __mul__(self, other):
        return self._create_copy(other * self.coeffs)

    def __eq__(self, other):
        """Compare vectors for equality.

        Note that vectors are only considered equal when they have
        exactly the same type.
        """
        return (type(self) == type(other) and
                self._basis == other._basis and
                (self.coeffs == other.coeffs).all())

