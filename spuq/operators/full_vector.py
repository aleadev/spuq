import numpy as np
   

class Vector(object):
    """Abstract base class for vectors"""

    def dim(self):
        """Return dimension of this vector"""
        return NotImplemented

    def __mul__(self,other):
        """Compute product with a scalar"""
        return NotImplemented

    def __add__(self,other):
        """Compute sum of vectors"""
        return NotImplemented

    def __sub__(self,other):
        """Compute difference of vectors"""
        return NotImplemented

    def __mul__(self,other):
        """Compute product of scalar and vectors"""
        return NotImplemented

    def __rmul__(self,other):
        """Compute product of scalar and vectors"""
        return self.__mul__(other)

    def asvector(self):
        return NotImplemented

class FullVector(Vector):
    """A vector classed based on the numpy array"""
    def __init__(self,vec):
        assert( isinstance( vec, np.ndarray ) )
        self.vec = vec
        from spuq.bases.basis import EuclideanBasis
        self._basis = EuclideanBasis(vec.shape[0])

    def dim(self):
        """Return dimension of this vector"""
        return self.vec.shape[0]

    def basis(self):
        return self._basis

    def as_vector(self):
        return self.vec
    
    def __add__(self, other):
        assert( isinstance(other, FullVector) )
        assert( self.basis()==other.basis() )
        return FullVector(self.vec + other.vec)

    def __sub__(self, other):
        assert( isinstance(other, FullVector) )
        assert( self.basis()==other.basis() )
        return FullVector(self.vec - other.vec)

    def __mul__(self, other):
        assert( np.isscalar(other) )
        return FullVector(other*self.vec)

    def __eq__(self, other):
        if not isinstance(other, FullVector):
            return False
        return self._basis==other._basis and (self.vec==other.vec).all()
    
    
    def __repr__(self):
        return "FullVector(" + str(self.vec) + ")"

