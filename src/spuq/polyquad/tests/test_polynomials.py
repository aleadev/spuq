import math
import numpy as np

from spuq.utils.testing import *
from spuq.polyquad.polynomials import *


def _check_poly_consistency(p, dist):
    """Check whether the polynomials are truly orthonormal for the
    given distribution"""
    def dist_integrate(dist, f, **kwargs):
        import scipy.integrate as integ
        def foo(x):
            return f(dist.ppf(x))
        return integ.quad(foo, 0, 1, epsabs=1e-5, **kwargs)[0]
    def polyprod(p, i, j):
        return lambda x: p.eval(i, x) * p.eval(j, x)
    assert_approx_equal(dist_integrate(dist, polyprod(p, 0, 0)), 
                        p.norm(0, False))
    assert_approx_equal(dist_integrate(dist, polyprod(p, 3, 3)), 
                        p.norm(3, False))
    assert_approx_equal(dist_integrate(dist, polyprod(p, 4, 4)), 
                        p.norm(4, False))
    assert_almost_equal(dist_integrate(dist, polyprod(p, 0, 1)), 0)
    assert_almost_equal(dist_integrate(dist, polyprod(p, 2, 3)), 0)
    assert_almost_equal(dist_integrate(dist, polyprod(p, 3, 4)), 0)


class TestPolynomials(TestCase):

    def test_eval_array(self):
        """Make sure the eval functions works for arrays."""
        _a = lambda * args: np.array(args, dtype=float)
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
        _a = lambda * args: np.array(args, dtype=float)

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

    @dec.slow
    def test_consistency(self):
        import scipy.stats as stats
        dist = stats.uniform(-1, 1 - -1)
        p = LegendrePolynomials(normalised=False)
        _check_poly_consistency(p, dist)


        p = LegendrePolynomials(a=2, b=5, normalised=False)
        dist = stats.uniform(2, 5 - 2)
        _check_poly_consistency(p, dist)

        p = LegendrePolynomials(a=-2.5, b=-1.2, normalised=True)
        assert_equal(p.norm(4, False), 1)
        dist = stats.uniform(-2.5, -1.2 - -2.5)
        _check_poly_consistency(p, dist)


class TestHermite(TestCase):

    def test_hermite(self):
        """Make sure the Hermite polynomials work."""
        x = 3.14159
        p = StochasticHermitePolynomials()
        assert_equal(p.eval(0, x), 1)
        assert_almost_equal(p.eval(3, x), x ** 3 - 3 * x)
        assert_equal(p.norm(0, False), 1)
        assert_equal(p.norm(1, False), 1)
        assert_equal(p.norm(2, False), 2)
        assert_equal(p.norm(3, False), 6)
        assert_equal(p.norm(4, False), 24)
        assert_equal(p.norm(5, False), 120)

        p = StochasticHermitePolynomials(normalised=True)
        assert_almost_equal(p.eval(3, x), (x ** 3 - 3 * x) / math.sqrt(6.0))
        assert_almost_equal(p.eval(5, x), (x ** 5 - 10 * x ** 3 + 15 * x) /
                            math.sqrt(120.0))
    @dec.slow
    def test_consistency(self):
        import scipy.stats as stats
        dist = stats.norm(0, 1)
        p = StochasticHermitePolynomials(normalised=False)
        _check_poly_consistency(p, dist)

        p = StochasticHermitePolynomials(mu=3, sigma=2, normalised=False)
        dist = stats.norm(3, 2)
        _check_poly_consistency(p, dist)

        p = StochasticHermitePolynomials(mu= -2.5, sigma=1.2, normalised=True)
        assert_equal(p.norm(4, False), 1)
        dist = stats.norm(-2.5, 1.2)
        _check_poly_consistency(p, dist)


if __name__ == "__main__":
    run_module_suite()
