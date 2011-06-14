import numpy;



class RandomField(object):
    def __init__(self):
        # maybe this should go into TensorProductBasis
        self.Phi=SpatialBasis();
        self.Psi=StochasticBasis();
        # maybe this should go into derived classes
        self.coeffs=GMatrix()
    def __add__(self,other):
        # add only if same spatial and stochastic basis are used
        # or add scalar, or spatial function
        return NotImplemented
    def __mult__(self,other):
        # mult by other field
        # mult by scalar
        return NotImplemented
        
class SeparatedRandomField(RandomField):
    # e.g. for KL representations
    pass

class FullRandomField(RandomField):
    # e.g. for full (PCE,gPC) representations
    pass



class FunctionalBasis(object):
    def getGramian(self):
        # return a LinearOperator, not necessarily a matrix
        return NotImplemented
    def evalAt(self,vector):
        return NotImplemented
    def integrateOver(self,function):
        # do we need that? What functions are general enough to be
        # included in a Basis base blass
        return NotImplemented
    # projection (oblige, orthogonal), 

class TensorProductBasis(Basis):
    def evalAt(self,vector):
        # for b in bases: 
        # side question: what about tensor product bases? Is "vector" a tuple then?
        return NotImplemented

class Algebra(object):
    # mult, identity
    pass

class StochasticBasis(Basis,Algebra):
    pass
    

class PolynomialBasis(StochasticBasis):
    def __init__(self):
        pass
        # describe independent variables by pdf, cdf, icdf, 
        # structure coefficients, mean, var, etc
        # describe multiindex sets
        # describe normalisation
        
class GPCBasis(PolynomialBasis):
    # ideas I have
    pass

class PCEBasis(PolynomialBasis):
    # we already have that (normalised or not?)
    pass

class WaveletBasis(StochasticBasis):
    # see LeMaitre
    pass

class SFEMBasis(StochasticBasis):
    # as in Babuska
    pass


class SpatialBasis(Basis):
    pass
    
class LagrangeBasis(SpatialBasis):
    pass


class KL(object):
    pass

class PCE(object):
    pass

class KL_PCE(object):
    pass

    

        
            
class GMatrix(object):
    pass
    
class GVector(object):
    def __add__(self,other):
        #return GVector()
        return NotImplemented
    def __mul__(self,other):
        return NotImplemented
    
class Tensor(GVector):
    def __init__(self):
        self.x=1
    def __add__(self,other):
        t=Tensor()
        t.x=self.x+other.x
        return t
    def __repr__(self):
        return str(self.x)

class LinearOperator(object):
    def range_basis(self):
        "Returns the dimension of the range of Op"
        return NotImplemented
    def dim_range(self):
        "Returns the dimension of the range of Op"
        return NotImplemented
    def dim_domain(self):
        "Returns the dimension of the domain of Op"
        return NotImplemented
    def __call__(self, arg):
        return NotImplemented
    def apply(self):
        pass
    def transpose(self):
        pass
    def invert(self):
        pass
    pass

class ComposedOpertor(LinearOperator):
    def __init__(op1,op2):
        assert( op1.range_basis()==op2.domain_basis() );
        self.op1 = op1
        self.op2 = op2
    
class MatrixOperator(LinearOperator):
    "A linear operator implemented as a matrix"
    pass

class TensorProductOperator(LinearOperator):
    pass
    
def genpcg( operator, rhs, epsilon=1e-4):
    assert( isinstance( operator, LinearOperator ) )
    return rhs



class ProbabilityDistribution(object):
    def pdf( self, x ): pass
    def cdf( self, x ): pass
    def invcdf( self, x ): pass
    def mean( self ): pass
    def var( self ): pass
    def skew( self ): pass
    def excess( self ): pass
    def median( self ): pass

class UniformDistribution(ProbabilityDistribution):
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def mean( self ): return self.a+self.b
    def var( self ): pass
    def skew( self ): pass
    def excess( self ): pass
    def median( self ): pass
    def pdf( self, x ): pass
    def cdf( self, x ): pass
    def invcdf( self, x ): pass
    def __repr__(self):
        return "U["+str(self.a)+","+str(self.b)+"]"

class ShiftedDistribution(ProbabilityDistribution):
    def __init__(self,dist,delta):
        self.dist=dist
        self.delta=delta
    def mean(self):
        return dist.mean()+delta
    def var(self):
        return dist.var()
    def __repr__(self):
        return self.dist.__repr__()+"+"+str(self.delta)
    

def shift( dist, delta ):
    try:
        return dist.shift( delta )
    except AttributeError:
        return ShiftedDistribution( dist, delta )
   
def main():
    t=Tensor()
    s=Tensor()
    print t+s
    genpcg( MatrixOperator(), Tensor() )
    print None
    x=numpy.array([1, 2])
    y=numpy.array([1, 2.3])
    x=x+y
    print x
    from numpy import vstack, hstack
    print vstack((x,y))
    print hstack((x,y))
    b=StochasticBasis()
    print b
    print isinstance(b,Algebra)
    print isinstance(b,Basis)
    print isinstance(b,LinearOperator)
    u=UniformDistribution(3,5)
    s=shift(u,2.3)
    print u, s
    

main()
