import numpy as np

from spuq.utils.testing import *
from spuq.polyquad._polynomials import *


class TestRecurrences(TestCase):
    def test_legendre(self):
        x = np.poly1d([1, 0])
        P = compute_poly(rc_legendre, 5, x)
        assert_array_equal(P[0].coeffs, [1])
        assert_array_equal(P[1].coeffs, np.array([1, 0]))
        assert_array_equal(P[2].coeffs, np.array([3, 0, - 1]) / 2.0)
        assert_array_equal(P[3].coeffs, np.array([5, 0, - 3, 0]) / 2.0)
        assert_array_equal(P[4].coeffs, np.array([35, 0, - 30, 0, 3]) / 8.0)

    def test_stoch_hermite(self):
        x = np.poly1d([1, 0])
        P = compute_poly(rc_stoch_hermite, 5, x)
        assert_array_equal(P[0].coeffs, [1])
        assert_array_equal(P[1].coeffs, np.array([1, 0]))
        assert_array_equal(P[2].coeffs, np.array([1, 0, - 1]))
        assert_array_equal(P[3].coeffs, np.array([1, 0, - 3, 0]))
        assert_array_equal(P[4].coeffs, np.array([1, 0, - 6, 0, 3]))

    def test_monomials(self):
        x = np.poly1d([1, 0])
        P = compute_poly(rc_monomials, 5, x)
        assert_array_equal(P[4], x**4)

    def test_sqnorm_from_rc(self):
        assert_array_almost_equal(sqnorm_legendre(0), 1)
        h1 = [sqnorm_from_rc(rc_legendre, i) for i in range(7)]
        h2 = [sqnorm_legendre(i) for i in range(7)]
        assert_array_almost_equal(h1, h2)

        assert_array_almost_equal(sqnorm_stoch_hermite(0), 1)
        h1 = [sqnorm_from_rc(rc_stoch_hermite, i) for i in range(7)]
        h2 = [sqnorm_stoch_hermite(i) for i in range(7)]
        assert_array_almost_equal(h1, h2)

    def test_rc3_to_orth(self):
        x = np.poly1d([1, 0])
        rc3_norm_legendre = rc_norm_to_rc3(rc_norm_legendre)
        P1 = compute_poly(rc3_norm_legendre, 4, x)
        P2 = compute_poly(rc_legendre, 4, x)
        hs = [sqnorm_legendre(i) ** 0.5 for i in range(5)]

        assert_array_almost_equal(P1[0] * hs[0], P2[0])
        assert_array_almost_equal(P1[1] * hs[1], P2[1])
        assert_array_almost_equal(P1[2] * hs[2], P2[2])
        assert_array_almost_equal(P1[3] * hs[3], P2[3])
        assert_array_almost_equal(P1[4] * hs[4], P2[4])

    def test_normalise_rc(self):
        rc_1 = rc_norm_to_rc3(rc_norm_legendre)
        rc_2 = normalise_rc(rc_legendre)
        rc_3 = normalise_rc(rc_legendre, sqnorm_legendre)

        assert_array_almost_equal(rc_1(0), rc_2(0))
        assert_array_almost_equal(rc_1(1), rc_2(1))
        assert_array_almost_equal(rc_1(2), rc_2(2))
        assert_array_almost_equal(rc_1(3), rc_2(3))
        assert_array_almost_equal(rc_1(4), rc_2(4))
        assert_array_almost_equal(rc_1(1), rc_3(1))
        assert_array_almost_equal(rc_1(2), rc_3(2))
        assert_array_almost_equal(rc_1(3), rc_3(3))
        assert_array_almost_equal(rc_1(4), rc_3(4))

    def test_rc_shift_scale(self):
        x = np.poly1d([1.0, 0])
        rc_1 = rc_legendre
        rc_2 = rc_shift_scale(rc_legendre, 2, 3)
        P1 = compute_poly(rc_1, 5, (x-2.0)/3.0)
        P2 = compute_poly(rc_2, 5, x)
        assert_array_almost_equal(P1[0], P2[0])
        assert_array_almost_equal(P1[3], P2[3])
        assert_array_almost_equal(P1[5], P2[5])
        P1 = compute_poly(rc_1, 5, x)
        P2 = compute_poly(rc_2, 5, 3*x+2)
        assert_array_almost_equal(P1[0], P2[0])
        assert_array_almost_equal(P1[3], P2[3])
        assert_array_almost_equal(P1[5], P2[5])

    def test_rc_window_trans(self):
        x = np.poly1d([1.0, 0])
        rc_1 = rc_stoch_hermite
        P1 = compute_poly(rc_1, 5, x)
        # just scale
        rc_2 = rc_window_trans(rc_stoch_hermite, (-1, 1), (-2, 2))
        P2 = compute_poly(rc_2, 5, x)
        assert_almost_equal(P2[5](-2), P1[5](-1))
        assert_almost_equal(P2[5](0), P1[5](0))
        assert_almost_equal(P2[5](1), P1[5](0.5))
        assert_almost_equal(P2[5](2), P1[5](1))
        # just shift
        rc_2 = rc_window_trans(rc_stoch_hermite, (-1, 1), (3, 5))
        P2 = compute_poly(rc_2, 5, x)
        assert_almost_equal(P2[5](3), P1[5](-1))
        assert_almost_equal(P2[5](4), P1[5](0))
        assert_almost_equal(P2[5](5), P1[5](1))
        # shift and scale
        rc_2 = rc_window_trans(rc_stoch_hermite, (-1, 1), (2, 6))
        P2 = compute_poly(rc_2, 5, x)
        assert_almost_equal(P2[5](2), P1[5](-1))
        assert_almost_equal(P2[5](4), P1[5](0))
        assert_almost_equal(P2[5](5), P1[5](0.5))
        assert_almost_equal(P2[5](6), P1[5](1))

    def test_eval_clenshaw(self):
        rc = rc_monomials
        x = np.poly1d([1, 0])
        assert_array_equal(eval_clenshaw(rc, [2], x), 2)
        assert_array_equal(eval_clenshaw(rc, [2, 3, 0.5], x), 
                           2 + 3 * x + 0.5 * x ** 2)
        assert_array_equal(eval_clenshaw(rc, [2, 3, 0.5, 4], x), 
                           2 + 3 * x + 0.5 * x ** 2 + 4 * x**3)

        from operator import add, mul
        from itertools import imap
        inner = lambda p, q: reduce(add, imap(mul, p, q))

        rc = rc_legendre
        x = np.poly1d([1, 0])
        p = compute_poly(rc, 4, x)
        assert_array_equal(eval_clenshaw(rc, [3], x), inner(p, [3]))
        assert_array_equal(eval_clenshaw(rc, [3, 5], x), inner(p, [3, 5]))
        assert_array_equal(eval_clenshaw(rc, [3, 5, 7], x), inner(p, [3, 5, 7]))
        assert_array_equal(eval_clenshaw(rc, [3, 5, 7, 9], x), 
                           inner(p, [3, 5, 7, 9]))

    def test_eval_forsythe(self):
        rc = rc_monomials
        x = np.poly1d([1, 0])
        assert_array_equal(eval_forsythe(rc, [2], x), 2)
        assert_array_equal(eval_forsythe(rc, [2, 3, 0.5], x), 
                           2 + 3 * x + 0.5 * x ** 2)
        assert_array_equal(eval_forsythe(rc, [2, 3, 0.5, 4], x), 
                           2 + 3 * x + 0.5 * x ** 2 + 4 * x**3)

        from operator import add, mul
        from itertools import imap
        inner = lambda p, q: reduce(add, imap(mul, p, q))

        rc = rc_legendre
        x = np.poly1d([1, 0])
        p = compute_poly(rc, 4, x)
        assert_array_equal(eval_forsythe(rc, [3], x), inner(p, [3]))
        assert_array_equal(eval_forsythe(rc, [3, 5], x), inner(p, [3, 5]))
        assert_array_equal(eval_forsythe(rc, [3, 5, 7], x), inner(p, [3, 5, 7]))
        assert_array_equal(eval_forsythe(rc, [3, 5, 7, 9], x), 
                           inner(p, [3, 5, 7, 9]))


test_main()
