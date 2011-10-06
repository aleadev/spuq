from abc import ABCMeta, abstractproperty, abstractmethod

from spuq.utils.decorators import copydocs

class Basis(object):
    """Abstract base class for basis objects"""
    __metaclass__ = ABCMeta

    @abstractproperty
    def dim(self):
        """The dimension of this basis."""
        return NotImplemented


#@copydocs
class EuclideanBasis(Basis):
    def __init__(self, dim):
        self._dim = dim
    
    @property
    def dim(self):
        return self._dim

    def __eq__(self, other):
        # Note: classes must match exactly, otherwise it is a
        # *different* basis
        return (self.__class__ == other.__class__ and
                self.dim == other.dim)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "%s(%s)" % (self.__class__, self.dim)


class FunctionBasis(Basis):

    @abstractproperty
    def gramian(self):
        """The Gramian as a LinearOperator (not necessarily a matrix)"""
        return NotImplemented





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
