import numpy as np
from numpy import array, ndarray, zeros, dot
from spuq.bases.basis import *

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
            return ComposedLinearOperator( other, self )
        else:
            return self.apply( other )

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


class ComposedLinearOperator(LinearOperator):
    """Wrapper class for linear operators mapping elements from 
    some domain into the codomain
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
        return self.op1.can_transpose() and self.op2.can_transpose()

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

    def asvector(self):
        return NotImplemented

class FullVector(Vector):
    """A vector classed based on the numpy array"""
    def __init__(self,vec):
        assert( isinstance( vec, ndarray ) )
        self.vec = vec
        self._basis = EuclideanBasis(vec.shape[0])

    def dim(self):
        """Return dimension of this vector"""
        return self.vec.shape[0]

    def basis(self):
        return self._basis

    def as_vector(self):
        return self.vec
    
    def __repr__(self):
        return "FullVector(" + str(self.vec) + ")"



if __name__=="__main__":
    from numpy.random import rand, randn
    A = FullLinearOperator( 1 + rand(3, 5) )
    B = FullLinearOperator( 1 + rand(7, 3) )
    print A.domain_dim(), A.codomain_dim()
    print B.domain_dim(), B.codomain_dim()

    x = FullVector( rand( 5,1 ) )
    print x

    C = B * A
    print C.domain_dim(), C.codomain_dim()

    print B(A(x))
    print (B * A)(x)

    print B * A * x
    print B * (A * x)
    print (B * A) * x

    print (B*A).as_matrix() * x.as_vector()
    print B.as_matrix() * (A.as_matrix() * x.as_vector())

    CT=C.transpose()

