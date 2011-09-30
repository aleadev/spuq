import numpy as np
from numpy.testing import *

from spuq.polynomials.polynomials import *


class TestPolynomials(TestCase):
    def test_legendre(self):
        """Make sure the Legendre polynomials work."""
        _a = lambda *args: np.array(args, dtype=float)

        x = 3.14159
        p = LegendrePolynomials()
        self.assertEquals(p.eval(0, x), 1)
        self.assertEquals(p.eval(3, x), 2.5 * x ** 3 - 1.5 * x)

    def test_hermite(self):
        """Make sure the Hermite polynomials work."""
        x = 3.14159
        p = StochasticHermitePolynomials()
        self.assertEquals(p.eval(0, x), 1)
        self.assertAlmostEquals(p.eval(3, x), x ** 3 - 3 * x)

    def test_eval_array(self):
        """Make sure the eval functions works for arrays."""
        _a = lambda *args: np.array(args, dtype=float)
        p = LegendrePolynomials()
        x = _a(1, 2, 3)
        assert_array_equal(p.eval(0, x), _a(1, 1, 1))
        assert_array_equal(p.eval(1, x), _a(1, 2, 3))
        assert_array_equal(p.eval(2, x), _a(1, 5.5, 13))

    def test_eval_poly(self):
        """Make sure the eval functions works for polynomials."""
        p = LegendrePolynomials()
        x = np.poly1d([1, 0])
        x2 = np.poly1d([1, 0, 0])
        self.assertEquals(p.eval(0, x2), np.poly1d([1]))
        self.assertEquals(p.eval(1, x2), x ** 2)
        self.assertEquals(p.eval(3, x2), 2.5 * x ** 6 - 1.5 * x ** 2)
