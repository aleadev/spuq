from numpy import poly1d
from spuq.linalg.function import GenericFunction
    
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
