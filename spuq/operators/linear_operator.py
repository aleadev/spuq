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
            return ComposedLinearOperator( op1=other, op2=self )
        elif ( np.isscalar(other) ):
            # TODO: OUCH, not efficient (need better sum operator)
            return SumLinearOperator( op1=self, op2=self, a1=other, a2=0.0 ) 
        else:
            return self.apply( other )

    def __rmul__(self, other):
        assert( np.isscalar(other) )
        return self.__mul__(other)

    def __add__(self, other):
        return SumLinearOperator( op1=self, op2=other )

    def __sub__(self, other):
        return SumLinearOperator( op1=self, op2=other, a2=-1.0 )

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

class SumLinearOperator(LinearOperator):
    """Wrapper class for linear operators adding two operators
    """
    
    def __init__(self, op1, op2, a1=1.0, a2=1.0, \
                     trans=None, inv=None, invtrans=None):
        """Takes two operators and returns the sum of those operators"""
        assert( op1.domain_basis() == op2.domain_basis() )
        assert( op1.codomain_basis() == op2.codomain_basis() )
        self.op1 = op1
        self.op2 = op2
        self.a1 = a1
        self.a2 = a2
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
        r1 = self.op1.apply( vec )
        if self.a1!=1.0:
            r1=self.a1*r1
        r2 = self.op2.apply( vec )
        if self.a2!=1.0:
            r2=self.a2*r2
        return r1+r2
    
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
            return False

    def transpose(self):
        """Transpose the operator"""
        if self.trans:
            return self.trans
        else:
            return SumLinearOperator(
                self.op1.transpose(),
                self.op2.transpose(),
                trans=self,
                inv=self.invtrans,
                invtrans=self.inv)

    def invert(self):
        """Return an operator that is the inverse of this operator"""
        if self.inv:
            return self.inv
        else:
            return SumLinearOperator(
                self.op2.invert(),
                self.op1.invert(),
                inv = self,
                trans = self.invtrans,
                invtrans = self.trans)

    def as_matrix(self):
        return self.op1.as_matrix() + self.op2.as_matrix()
 

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



if __name__=="__main__":
    from numpy.random import rand, randn
    A = FullLinearOperator( 1 + rand(3, 5) )
    B = FullLinearOperator( 1 + rand(7, 3) )
    print A.domain_dim(), A.codomain_dim()
    print B.domain_dim(), B.codomain_dim()

    x = FullVector( rand( 5,1 ) )
    print x

    # operators can be multiplied
    C = B * A
    print C.domain_dim(), C.codomain_dim()

    # operator composition can be performed in a number of ways
    print B(A(x))
    print (B * A)(x)

    print B * A * x
    print B * (A * x)
    print (B * A) * x

    # similar as above, only as matrices
    print (B*A).as_matrix() * x.as_vector()
    print B.as_matrix() * (A.as_matrix() * x.as_vector())

    # you can transpose (composed) operators
    AT=A.transpose()
    BT=B.transpose()
    CT=C.transpose()

    y = FullVector( rand( CT.domain_dim(),1 ) )
    print CT*y
    print AT*(BT*y)
    
    # can add and subtract operators
    print (B * (A+A))*x
    print C*x+C*x
    print (C-C)*x
    print C*x-C*x

    # you can pre- and post-multiply vectors with scalars
    print 3*x-x*3

    # you can multiply operators with scalars or vectors with scalars
    print (3*C)*x
    print (C*3)*x
    print 3*(C*x)
    
    
