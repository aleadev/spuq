import numpy as np
from numpy import array, ndarray, zeros, dot

from spuq.bases.basis import EuclideanBasis
from spuq.operators.full_vector import FullVector
from spuq.operators.composed_operator import ComposedLinearOperator
from spuq.operators.summed_operator import SummedLinearOperator


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
        return SummedLinearOperator( operators=(self,other) )

    def __sub__(self, other):
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
    



    
from spuq.operators.linear_operator import LinearOperator

class ComposedLinearOperator(LinearOperator):
    """Wrapper class for linear operators that are composed of other
    linear operators
    """
    
    def __init__(self, op1, op2, trans=None, inv=None, invtrans=None):
        """Takes two operators and returns the composition of those operators"""
        assert( op1.codomain_basis() == op2.domain_basis() )
        self.op1 = op1
        self.op2 = op2
        self.trans = None
        self.inv = None
        self.invtrans = None

    def domain_basis(self):
        "Returns the basis of the domain"
        return self.op1.domain_basis()

    def codomain_basis(self):
        "Returns the basis of the codomain"
        return self.op2.codomain_basis()

    def apply(self, vec):
        "Apply operator to vec which should be in the domain of op"
        r = self.op1.apply( vec )
        r = self.op2.apply( r )
        return r
    
    def can_transpose(self):
        "Return whether the operator can transpose itself"
        if self.trans:
            return True
        else:
            return self.op1.can_transpose() and self.op2.can_transpose()

    def is_invertible(self):
        "Return whether the operator is invertible"
        if self.inv:
            return True
        else:
            return self.op1.is_invertible() and self.op2.is_invertible()

    def transpose(self):
        """Transpose the operator"""
        if self.trans:
            return self.trans
        else:
            return ComposedLinearOperator(
                self.op2.transpose(),
                self.op1.transpose(),
                trans=self,
                inv=self.invtrans,
                invtrans=self.inv)

    def invert(self):
        """Return an operator that is the inverse of this operator"""
        if self.inv:
            return self.inv
        else:
            return ComposedLinearOperator(
                self.op2.invert(),
                self.op1.invert(),
                inv = self,
                trans = self.invtrans,
                invtrans = self.trans)

    def as_matrix(self):
        return self.op2.as_matrix() * self.op1.as_matrix()

from spuq.operators.linear_operator import LinearOperator

class SummedLinearOperator(LinearOperator):
    """Wrapper class for linear operators adding two operators
    """
    
    def __init__(self, operators, factors=None, \
                     trans=None, inv=None, invtrans=None):
        """Takes two operators and returns the sum of those operators"""
        op1=operators[0];
        for op2 in operators:
            assert( op1.domain_basis() == op2.domain_basis() )
            assert( op1.codomain_basis() == op2.codomain_basis() )
        self.operators = operators
        self.factors = factors
        self.trans = None
        self.inv = None
        self.invtrans = None

    def domain_basis(self):
        "Returns the basis of the domain"
        return self.operators[0].domain_basis()

    def codomain_basis(self):
        "Returns the basis of the codomain"
        return self.operators[0].codomain_basis()

    def apply(self, vec):
        "Apply operator to vec which should be in the domain of op"
        # TODO: implement zero vector
        r=None
        for i, op in enumerate(self.operators):
            r1 = op.apply( vec )
            if self.factors and self.factors[i]!=1.0:
                r1=self.factors[i]*r1
            if r:
                r=r+r1
            else:
                r=r1
        return r
    
    def can_transpose(self):
        "Return whether the operator can transpose itself"
        if self.trans:
            return True
        else:
            return all( map( lambda op: op.can_transpose(), self.operators ) )

    def is_invertible(self):
        "Return whether the operator is invertible"
        if self.inv:
            return True
        else:
            return False

    def transpose(self):
        """Transpose the operator"""
        # TODO: should go into AbstractLinOp, here only create_transpose
        if self.trans:
            return self.trans
        else:
            return SummedLinearOperator(
                map( lambda op: op.transpose(), self.operators ),
                self.factors,
                trans=self,
                inv=self.invtrans,
                invtrans=self.inv)

    def invert(self):
        """Return an operator that is the inverse of this operator"""
        if self.inv:
            return self.inv
        else:
            # Cannot do this, the inverse of a sum is not the sum of the inverses
            # throw exeception?
            # TODO: should go if only 1 operators
            return None

    def as_matrix(self):
        return sum( map( lambda op: op.as_matrix(), self.operators ) )
class TensorOperator(LinearOperator):
    pass
class MatrixOperator(LinearOperator):
    "A linear operator implemented as a matrix"
    pass
    
from spuq.operators.linear_operator import LinearOperator
from spuq.operators.linear_operator import AbstractLinearOperator

class ReindexOperator(AbstractLinearOperator):
    def __init__( self, index_map, domain_basis, codomain_basis ):
        AbstractLinearOperator( self, domain_basis, codomain_basis )
        self.index_map = index_map
        
    def apply( ):
        pass

    def transpose():
        pass

    def invert():
        # is size(domain_basis)==size(codomain_basis) && index_map is full
        pass

    
