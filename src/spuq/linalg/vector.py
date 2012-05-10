"""This module supplies a base class for vectors and a basis
implementation for flat vectors, i.e. vectors that have no structure,
but consist just of a bunch of numbers. The base class Vector just
satisfies the vector space axioms and supplies some default
implementations.

Note for derived classes: It is sufficient to override the methods
copy, __imul__, __iadd__, and either __isub__ or __neg__. The
operations __mul__, __rmul__, __add__, __radd__ and so on are then
defined automatically.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from numbers import Number as Scalar

import numpy as np

from spuq.linalg.basis import Basis, CanonicalBasis, check_basis
from spuq.utils import strclass, with_equality
from spuq.utils.type_check import takes, returns, anything, optional, list_of
from spuq.math_utils.math_object import MathObject

__all__ = ["Scalar", "Vector", "FlatVector", "inner"]


class Transposable(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def transpose(self):
        pass


class Flat(object):
    __metaclass__ = ABCMeta
    @abstractproperty
    def coeffs(self):
        pass

    def as_array(self):
        return self.coeffs


class Vector(MathObject):
    """Abstract base class for vectors which consist of a coefficient
    vector and an associated basis"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def basis(self):  # pragma: no cover
        """Return basis of this vector"""
        raise NotImplementedError

    @abstractmethod
    def copy(self):  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def flatten(self):
        raise NotImplementedError

    def transpose(self):
        dual_coeffs = vec.basis.Gramian * vec.coeffs
        return vec.__class__(dual_coeffs, vec.basis.dual())

    @abstractmethod
    def __eq__(self, other):  # pragma: no cover
        """Test whether vectors are equal."""
        return NotImplemented

    @abstractmethod
    def __neg__(self, other):  # pragma: no cover
        """Compute the negative of this vector."""
        return NotImplemented

    @abstractmethod
    def __iadd__(self, other):  # pragma: no cover
        """Add another vector to this one."""
        return NotImplemented

    @abstractmethod
    def __imul__(self, other):  # pragma: no cover
        """Multiply this vector with a scalar."""
        return NotImplemented

    def __inner__(self, other):  # pragma: no cover
        """Scalar product of this vector with another vector."""
        return NotImplemented

    def __rinner__(self, other):  # pragma: no cover
        """Scalar product of this vector with another vector (reverse)."""
        return NotImplemented

    def __add__(self, other):
        """Add two vectors."""
        return self.copy().__iadd__(other)

    def __radd__(self, other):  # pragma: no cover
        """This happens only when other is not a vector."""
        if isinstance(other, Scalar) and other==0:
            return self
        return NotImplemented

    def __sub__(self, other):
        """Subtract two vectors."""
        if hasattr(self, "__isub__"):
            return self.copy().__isub__(other)
        return self +(-other)

    def __rsub__(self, other):  # pragma: no cover
        """This happens only when other is not a vector."""
        return NotImplemented

    def __mul__(self, other):
        """Multiply a vector with a scalar from the right."""
        if isinstance(other, Scalar):
            return self.copy().__imul__(other)
        return NotImplemented

    def __rmul__(self, other):
        """Multiply a vector with a scalar from the left."""
        if isinstance(other, Scalar):
            return self.copy().__imul__(other)
        return NotImplemented

    def __repr__(self):
        return "<%s basis=%s, coeffs=%s>" % \
               (strclass(self.__class__), self.basis, self.coeffs)


class FlatVector(Vector, Flat):
    """A vector classed based on the numpy array"""

    @takes(anything, (np.ndarray, list_of(Scalar)), optional(Basis))
    def __init__(self, coeffs, basis=None):
        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(coeffs, dtype=float)
        if basis is None:
            basis = CanonicalBasis(coeffs.shape[0])
        assert(basis.dim == coeffs.shape[0])
        assert(coeffs.ndim == 1)
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

    def __inner__(self, other):
        check_basis(self.basis, other.basis)
        # TODO: needs to check the gramian too
        return np.dot(self.coeffs, other.coeffs)

    def __rinner__(self, other):
        return self.__inner__(other)

    def copy(self):
        return self._create_copy(self.coeffs.copy())

    def _create_copy(self, coeffs):
        """Creates a copy of this vector with new coefficients, but
        the same class and basis."""
        return self.__class__(coeffs, self.basis)

    def __eq__(self, other):
        """Compare vectors for equality.

        Note that vectors are only considered equal when they have
        exactly the same type."""
        return (type(self) == type(other) and
                self._basis == other._basis and
                (self._coeffs == other._coeffs).all())

    def flatten(self):
        return self

    @takes(anything)
    def __neg__(self):
        return self._create_copy(-self._coeffs)

    @takes(anything, "FlatVector")
    def __iadd__(self, other):
        check_basis(self.basis, other.basis)
        self._coeffs += other._coeffs
        return self

    @takes(anything, "FlatVector")
    def __isub__(self, other):
        check_basis(self.basis, other.basis)
        self._coeffs -= other.coeffs
        return self

    @takes(anything, Scalar)
    def __imul__(self, other):
        self._coeffs *= other
        return self


@takes(Vector, Vector)
def inner(vec1, vec2):
    res = vec1.__inner__(vec2)
    if res is not NotImplemented:
        return res
    res = vec2.__rinner__(vec1)
    if res is not NotImplemented:
        return res
    if isinstance(vec1, Transposable):
        return vec1.transpose() * vec2
    if isinstance(vec2, Transposable):
        return vec2.transpose() * vec1
    raise NotImplementedError(str(vec1) + "/" + str(vec2))
