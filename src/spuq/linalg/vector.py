from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from spuq.linalg.basis import Basis, EuclideanBasis


class Vector(object):
    """Abstract base class for vectors"""
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

    def __add__(self, other):
        """Compute sum of vectors"""
        return NotImplemented

    def __sub__(self, other):
        """Compute difference of vectors"""
        return NotImplemented

    def __mul__(self, other):
        """Compute product with a scalar"""
        return NotImplemented

    def __rmul__(self, other):
        """Compute product of scalar and vectors"""
        return self.__mul__(other)


class FlatVector(Vector):
    """A vector classed based on the numpy array"""
    def __init__(self, coeffs, basis=None):
        if basis is None:
            basis = EuclideanBasis(coeffs.shape[0])
        assert(isinstance(coeffs, np.ndarray))
        assert(isinstance(basis, Basis))
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

    def __add__(self, other):
        assert(isinstance(other, FlatVector))
        assert(self.basis == other.basis)
        return FlatVector(self.coeffs + other.coeffs, self.basis)

    def __sub__(self, other):
        assert(isinstance(other, FlatVector))
        assert(self.basis == other.basis)
        return FlatVector(self.coeffs - other.coeffs)

    def __mul__(self, other):
        assert(np.isscalar(other))
        return FlatVector(other * self.coeffs)

    def __eq__(self, other):
        if not isinstance(other, FlatVector):
            return False
        return (self._basis == other._basis and
                (self.coeffs == other.coeffs).all())

    def __repr__(self):
        return "FlatVector(" + str(self.coeffs) + ")"
