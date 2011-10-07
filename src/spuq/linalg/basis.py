from abc import ABCMeta, abstractproperty, abstractmethod

from spuq.utils import strclass
from spuq.utils.decorators import copydocs


class BasisMismatchError(ValueError):
    pass


def check_basis(basis1, basis2, descr1="basis1", descr2="basis2"):
    """Throw if the bases do not match"""
    if basis1 != basis2:
        raise BasisMismatchError("Basis don't match: %s=%s, %s=%s" %
                                 (descr1, str(basis1), descr2, str(basis2)))


class Basis(object):
    """Abstract base class for basis objects"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def dim(self):
        """The dimension of this basis."""
        return NotImplemented

    @abstractmethod
    def __eq__(self, other):
        """Compare two basis objects."""
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<%s dim=%s>" % \
               (strclass(self.__class__), self.dim)


@copydocs
class CanonicalBasis(Basis):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def __eq__(self, other):
        # Note: classes must match exactly, otherwise it is a
        # *different* basis
        return (type(self) == type(other) and
                self.dim == other.dim)


class FunctionBasis(Basis):

    @abstractproperty
    def gramian(self):
        """The Gramian as a LinearOperator (not necessarily a matrix)"""
        return NotImplemented


# class SubBasis, IndexedBasis


# NOTE: not directly supported by fenics classes - why do we need it?
#    @abstractmethod
#    def eval_at(self, vector):
#        """Evaluated the basis functions at the specified points"""
#        return NotImplemented
#class TensorProductBasis(Basis):
#    def eval_at(self, vector):
#        # for b in bases:
#        # side question: what about tensor product bases?
#        # Is "vector" a tuple then?
#        return NotImplemented
#
# Maybe multiple classes and a generator method for tensor products?
