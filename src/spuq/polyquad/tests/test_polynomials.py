import numpy as np

from spuq.utils.testing import *
from spuq.polyquad.polynomials import *


class TestPolynomials(TestCase):

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
        x = np.poly1d([1.0, 0])
        x2 = np.poly1d([1.0, 0, 0])
        assert_is_instance(p.eval(0, x2), np.poly1d)
        assert_equal(p.eval(0, x2).coeffs, [1])
        assert_equal(p.eval(1, x2), x ** 2)
        assert_equal(p.eval(3, x2), 2.5 * x ** 6 - 1.5 * x ** 2)


class TestLegendre(TestCase):

    def test_legendre(self):
        """Make sure the Legendre polynomials work."""
        _a = lambda *args: np.array(args, dtype=float)

        x = 3.14159
        p = LegendrePolynomials()
        assert_equal(p.eval(0, x), 1)
        assert_equal(p.eval(3, x), 2.5 * x ** 3 - 1.5 * x)

    def test_integrate(self):
        a, b = -1, 1
        p = LegendrePolynomials(a, b, False)
        x = np.poly1d([1.0, 0])
        P0 = p.eval(0, x)
        P1 = p.eval(1, x)
        P2 = p.eval(2, x)
        P3 = p.eval(3, x)
        def integ(p):
            q = p.integ()
            return (q(b) - q(a)) / (b - a)
        # test orthogonality
        assert_approx_equal(integ(P1 * P0), 0.0)
        assert_approx_equal(integ(P1 * P2), 0.0)
        assert_approx_equal(integ(P2 * P3), 0.0)

        # test normalisation
        #assert_approx_equal(integ(P0 * P0), 1.0)
        #assert_approx_equal(integ(P1 * P1), 1.0)
        #assert_approx_equal(integ(P3 * P3), 1.0)
    


class TestHermite(TestCase):

    def test_hermite(self):
        """Make sure the Hermite polynomials work."""
        x = 3.14159
        p = StochasticHermitePolynomials()
        assert_equal(p.eval(0, x), 1)
        assert_almost_equal(p.eval(3, x), x ** 3 - 3 * x)
