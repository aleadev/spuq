from numpy import poly1d, ones
from spuq.linalg.function import GenericFunction

class FunctionScalar(GenericFunction):
    def __init__(self, const=1, domain_dim=1, codomain_dim=1):
        self.const = const
        
    def eval(self, *x):
        return self.const*ones((self.codomain_dim,1))
    
    def diff(self):
        return FunctionScalar(const=0, domain_dim=self.domain_dim, codomain_dim=self.domain_dim*self.codomain_dim) 
    
class Poly1dFunction(GenericFunction):
    """numpy poly1d GenericFunction wrapper"""
    
    def __init__(self, f=None, coeffs=None):
        if f:
            assert isinstance(f, poly1d)
            self.f = f
        else:
            assert coeffs
            self.f = poly1d(coeffs)
    
    def eval(self, *x):
        return self.f(*x)
    
    def diff(self):
        return Poly1dFunction(self.f.deriv(1))

    # TODO: discuss if specialisation of operations between poly1d instances is required