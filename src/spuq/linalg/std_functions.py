import numpy as np

from spuq.linalg.function import GenericFunction, Differentiable
from spuq.utils.type_check import takes, anything

class Poly1dFunction(GenericFunction, Differentiable):
    """numpy poly1d GenericFunction wrapper"""

    @takes(anything, np.poly1d)
    def __init__(self, poly):
        GenericFunction.__init__(self, domain_dim=1, codomain_dim=1)
        self._poly = poly

    @classmethod
    #@takes(anything, np.ndarray)
    def from_coeffs(cls, coeffs):
        poly = np.poly1d(coeffs)
        return cls(poly)

    def eval(self, *x):
        return self._poly(*x)

    def diff(self):
        return Poly1dFunction(self._poly.deriv(1))

    # TODO: discuss if specialisation of operations between poly1d instances is required
