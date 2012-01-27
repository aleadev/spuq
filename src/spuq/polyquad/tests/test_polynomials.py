import math
import numpy as np

from spuq.utils.testing import *
from spuq.polyquad.polynomials import *

_DO_CONSISTENCY_TEST = True

def _check_poly_consistency(p, dist):
    #Check whether the polynomials are truly orthonormal for the given
    #distribution
    def dist_integrate(dist, f, **kwargs):
        from scipy.integrate import quad
        def foo(x):
            return f(dist.ppf(x))
        return quad(foo, 0, 1, epsabs=1e-5, **kwargs)[0]
    
    def polyprod(p, i, j):
        return lambda x: p.eval(i, x) * p.eval(j, x)
    
    assert_almost_equal(dist_integrate(dist, p[1]*p[3]), 0)
    assert_almost_equal(dist_integrate(dist, polyprod(p, 0, 1)), 0)
    assert_almost_equal(dist_integrate(dist, polyprod(p, 2, 3)), 0)
    assert_almost_equal(dist_integrate(dist, polyprod(p, 3, 4)), 0)

    assert_approx_equal(dist_integrate(dist, polyprod(p, 0, 0)), 
                        p.norm(0, False))
    assert_approx_equal(dist_integrate(dist, polyprod(p, 3, 3)), 
                        p.norm(3, False))
    assert_approx_equal(dist_integrate(dist, polyprod(p, 4, 4)), 
                        p.norm(4, False))


def test_eval_array():
    #Make sure the eval functions works for arrays.
    _a = lambda * args: np.array(args, dtype=float)
    p = LegendrePolynomials(normalised=False)
    x = _a(1, 2, 3)
    assert_array_equal(p.eval(0, x), _a(1, 1, 1))
    assert_array_equal(p.eval(1, x), _a(1, 2, 3))
    assert_array_equal(p.eval(2, x), _a(1, 5.5, 13))


def test_eval_poly():
    #Make sure the eval functions works for polynomials.
    p = LegendrePolynomials(normalised=False)
    x = np.poly1d([1.0, 0])
    x2 = np.poly1d([1.0, 0, 0])
    assert_is_instance(p.eval(0, x2), np.poly1d)
    assert_equal(p.eval(0, x2).coeffs, [1])
    assert_equal(p.eval(1, x2), x ** 2)
    #assert_equal(p.eval(3, x2), 2.5 * x ** 6 - 1.5 * x ** 2)


def test_get_coefficients():
    p = LegendrePolynomials(normalised=False)
    assert_array_equal(p.get_coefficients(3), [0, -1.5, 0, 2.5])


def test_norm():
    p = LegendrePolynomials(normalised=False)
    assert_equal(p.norm(3, True), math.sqrt(p.norm(3, False)))
    # should be default
    assert_equal(p.norm(3, True), p.norm(3))
        


def test_legendre():
    x = 3.14159
    p = LegendrePolynomials(normalised=False)
    assert_equal(p.eval(0, x), 1)
    assert_equal(p.eval(3, x), 2.5 * x ** 3 - 1.5 * x)

def test_legendre_integrate():
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

@skip_if(not _DO_CONSISTENCY_TEST)
@slow
def test_legendre_consistency():
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


def test_hermite():
    x = 3.14159
    p = StochasticHermitePolynomials(normalised=False)
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

@skip_if(not _DO_CONSISTENCY_TEST)
@slow
def test_hermite_consistency():
    import scipy.stats as stats
    dist = stats.norm(0, 1)
    p = StochasticHermitePolynomials(normalised=False)
    assert_false(p.normalised)
    _check_poly_consistency(p, dist)

    p = StochasticHermitePolynomials(mu=3, sigma=2, normalised=False)
    dist = stats.norm(3, 2)
    assert_false(p.normalised)
    _check_poly_consistency(p, dist)

    p = StochasticHermitePolynomials(mu=-2.5, sigma=1.2, normalised=True)
    assert_equal(p.norm(4, False), 1)
    dist = stats.norm(-2.5, 1.2)
    assert_true(p.normalised)
    _check_poly_consistency(p, dist)


def test_jacobi():
    x = JacobiPolynomials.x
    # test the standard JacobiPolynomials for alpha=1, beta=1  
    p = JacobiPolynomials(alpha=1.0, beta=1.0, normalised=False)
    assert_array_almost_equal(p.eval(0, x), x**0)
    assert_array_almost_equal(p.eval(1, x), 2 * x)
    assert_array_almost_equal(p.eval(2, x), 15.0 / 4  * x ** 2 - 3.0 / 4)
    assert_array_almost_equal(p.eval(3, x), 7 * x ** 3 - 3 * x)

    # test the standard JacobiPolynomials for alpha=0, beta=2 (unsymmetric), and check the norm  
    p = JacobiPolynomials(alpha=0.0, beta=2.0, normalised=False)
    assert_array_almost_equal(p.eval(1, x), 2 * x - 1)
    assert_array_almost_equal(p.eval(2, x), 3.75  * x ** 2 - 2.5 * x - 0.25)
    assert_array_almost_equal(p.eval(3, x), 
                              7 * x ** 3 - 5.25 * x **2 - 1.5 * x + 0.75)

    assert_equal(p.norm(0, False), 3.0/3.0)
    assert_equal(p.norm(1, False), 3.0/5.0)
    assert_approx_equal(p.norm(2, False), 3.0/7.0)
    assert_approx_equal(p.norm(3, False), 3.0/9.0)
    assert_approx_equal(p.norm(4, False), 3.0/11.0)
    assert_approx_equal(p.norm(5, False), 3.0/13.0)

    # test the normalised JacobiPolynomials for alpha=0, beta=2  
    p = JacobiPolynomials(alpha=0, beta=2, normalised=True)
    assert_array_almost_equal(p.eval(3, x), (7 * x ** 3 - 5.25 * x **2 - 1.5 * x + 0.75) / math.sqrt(1/3.0))

    p = JacobiPolynomials(alpha=0.5, beta=-0.5, normalised=False)
    assert_array_almost_equal(p.eval(0, x), x**0)
    assert_array_almost_equal(p.eval(1, x), x + 0.5)

    p = JacobiPolynomials(alpha=0.0, beta=0.0, normalised=False)
    assert_array_almost_equal(p.eval(0, x), x**0)
    assert_array_almost_equal(p.eval(1, x), x)
    assert_array_almost_equal(p.eval(3, x), 2.5 * x ** 3 - 1.5 * x)

def test_cmp_jacobi_legendre():
    # make the Jacobi polynomials for alpha=0 and beta=0 are the same as 
    # the Legendre polynomials 
    p = JacobiPolynomials(alpha=0, beta=0, a=3, b=7, normalised=True)
    q = LegendrePolynomials(a=3, b=7, normalised=True)
    assert_array_almost_equal( p[0], q[0])
    assert_array_almost_equal( p[1], q[1])
    assert_array_almost_equal( p[2], q[2])
    assert_array_almost_equal( p[3], q[3])
    assert_array_almost_equal( p[4], q[4])

@skip_if(not _DO_CONSISTENCY_TEST)
@slow
def test_jacobi_consistency():
    import scipy.stats as stats
    dist = stats.beta(2, 3, loc=-1, scale=2)
    p = JacobiPolynomials(alpha=2, beta=1, a=-1, b=1, normalised=False)
    assert_false(p.normalised)
    _check_poly_consistency(p, dist)

    dist = stats.beta(2, 1.5, loc=-2, scale=5)
    p = JacobiPolynomials(alpha=0.5, beta=1, a=-2, b=3)
    assert_true(p.normalised)
    _check_poly_consistency(p, dist)


test_main()
