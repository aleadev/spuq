import numpy as np
from numpy import array, ndarray, zeros, dot

from spuq.bases.basis import EuclideanBasis
from spuq.operators.full_vector import FullVector


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
    



    
