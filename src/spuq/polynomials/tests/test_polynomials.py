import numpy as np
from numpy.testing import *

from spuq.polynomials.polynomials import *

class TestLegendre(TestCase):
    def test_legendre(self):
        x = 3.14159
        l = LegendrePolynomials()
        self.assertEquals( l.eval(3, x), 2.5*x**3-1.5*x )


class TestHermite(TestCase):
    def test_hermite(self):
        x = 3.14159
        h = StochasticHermitePolynomials()
        self.assertAlmostEquals( h.eval(3, x), x**3-3*x )


if __name__ == "__main__":
    run_module_suite()
