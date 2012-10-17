from abc import abstractproperty, abstractmethod

from spuq.utils import strclass, with_equality
from spuq.utils.decorators import copydocs
from spuq.math_utils.math_object import MathObject


class BasisMismatchError(ValueError):
    pass


def check_basis(basis1, basis2, descr1="basis1", descr2="basis2"):
    """Throw if the bases do not match"""
    if basis1 != basis2:
        raise BasisMismatchError("Basis don't match: %s=%s, %s=%s" % 
                                 (descr1, str(basis1), descr2, str(basis2)))


@with_equality
class Basis(MathObject):
    """Abstract base class for basis objects"""

    def __init__(self, dual=False):
        _dual = True

    @abstractproperty
    def dim(self):  # pragma: no cover
        """The dimension of this basis."""
        return NotImplemented

    @property
    def is_dual(self):
        return self._dual

    def dual(self):  # pragma: no cover
        dual_basis = self.copy()
        dual_basis._dual = True
        raise dual_basis

    @abstractproperty
    def gramian(self):  # pragma: no cover
        """The Gramian as a LinearOperator (not necessarily a matrix)"""
        raise NotImplementedError

    def __repr__(self):
        return ("<%s dim=%s>" % 
                (strclass(self.__class__), self.dim))


@copydocs
class CanonicalBasis(Basis):
    def __init__(self, dim):
        Basis.__init__(self)
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def copy(self):
        return self.__cls__(self._dim)

    @property
    def gramian(self):
        return 1


class FunctionBasis(Basis):

    @abstractmethod
    def eval(self, x):
        """Evaluate the basis functions at point x where x has length domain_dim."""
        raise NotImplementedError

    @abstractproperty
    def domain_dim(self):
        """The dimension of the domain the functions are defined upon."""
        raise NotImplementedError


# class SubBasis, IndexedBasis

#class TensorProductBasis(Basis):
#    def eval_at(self, vector):
#        # for b in bases:
#        # side question: what about tensor product bases?
#        # Is "vector" a tuple then?
#        return NotImplemented
#
# Maybe multiple classes and a generator method for tensor products?
