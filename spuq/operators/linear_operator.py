import numpy as np
from numpy import array, ndarray, zeros, dot

class LinearOperator(object):
    """Abstract base class for linear operators mapping elements from 
    some domain into the codomain
    """

    def apply(self, vec):
        "Apply operator to vec which should be in the domain of op"
        return NotImplemented
    
    def can_transpose(self):
        "Return whether the operator can transpose itself"
        return NotImplemented

    def transpose(self):
        """Transpose the operator; 
        need not be implemented"""
        return NotImplemented

    def invert(self):
        """Return an operator that is the inverse of this operator; 
        may not be implemented"""
        return NotImplemented


    def domain_dim(self):
        "Returns the dimension of the domain"
        return self.domain_basis().dim()

    def codomain_dim(self):
        "Returns the dimension of the codomain"
        return self.codomain_basis().dim()

    def as_matrix(self):
        return NotImplemented

    def __mul__(self, other):
        from spuq.operators.composed_operator import ComposedLinearOperator
        from spuq.operators.summed_operator import SummedLinearOperator
        if isinstance(other, LinearOperator):
            #return ComposedLinearOperator( operators=(other, self) )
            return ComposedLinearOperator( other, self )
        elif ( np.isscalar(other) ):
            return SummedLinearOperator( operators=(self,), factors=(other,) ) 
        else:
            return self.apply( other )

    def __rmul__(self, other):
        assert( np.isscalar(other) )
        return self.__mul__(other)

    def __add__(self, other):
        from spuq.operators.summed_operator import SummedLinearOperator
        return SummedLinearOperator( operators=(self,other) )

    def __sub__(self, other):
        from spuq.operators.summed_operator import SummedLinearOperator
        return SummedLinearOperator( operators=(self,other), factors=(1,-1) ) 

    def __call__(self, arg):
        assert( self.domain_basis() == arg.basis() )
        return self.apply(arg)


class AbstractLinearOperator(LinearOperator):
    """Base class for linear operators implementing some of the base
    functionality
    """
    
    def __init__(self, domain_basis, codomain_basis):
        self._domain_basis=domain_basis
        self._codomain_basis=codomain_basis

    def domain_basis(self):
        "Returns the basis of the domain"
        return self._domain_basis
    
    def codomain_basis(self):
        "Returns the basis of the codomain"
        return self._codomain_basis

 

class FullLinearOperator(AbstractLinearOperator):
    def __init__(self, arr, domain_basis=None, codomain_basis=None):
        from spuq.bases.basis import EuclideanBasis
        assert( isinstance(arr, ndarray) )
        if domain_basis is None:
            domain_basis = EuclideanBasis(arr.shape[1])
        if codomain_basis is None:
            codomain_basis = EuclideanBasis(arr.shape[0])
            
        self._arr = arr
        AbstractLinearOperator.__init__(self, domain_basis, codomain_basis)

    def apply(self, vec):
        "Apply operator to vec which should be in the domain of op"
        return FullVector( dot(self._arr, vec.vec) )

    def as_matrix(self):
        return np.asmatrix(self._arr)

    def transpose(self):
        return FullLinearOperator( self._arr.T, 
                                   self.codomain_basis(), 
                                   self.domain_basis() )
    

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
        assert( isinstance( vec, ndarray ) )
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
    
    
    def __repr__(self):
        return "FullVector(" + str(self.vec) + ")"



    
