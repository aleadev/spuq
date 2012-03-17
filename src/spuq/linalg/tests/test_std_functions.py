import numpy as np

from spuq.utils.testing import test_main, assert_equal, assert_array_almost_equal
from spuq.linalg.std_functions import Poly1dFunction

def assert_polyfun_equal(p1, p2):
    x = np.poly1d([1, 0])
    assert_array_almost_equal(p1(x), p2(x))

def test_poly1d_func():
    p4 = np.array([3.0, 4, 5, 6])
    pf = Poly1dFunction(np.poly1d(p4))
    assert_equal(pf(2), 24 + 16 + 10 + 6)
    pf = Poly1dFunction.from_coeffs(p4)
    assert_equal(pf(2), 24 + 16 + 10 + 6)

def test_poly1d_func_deriv():
    p4 = np.array([3.0, 4, 5, 6])
    p3 = np.array([9.0, 8, 5])
    p2 = np.array([18.0, 8])
    pf = Poly1dFunction(np.poly1d(p4))

    assert_polyfun_equal(pf, np.poly1d(p4))
    assert_polyfun_equal(pf.D, np.poly1d(p3))
    assert_polyfun_equal(pf.D.D, np.poly1d(p2))

def test_poly1d_func_ops():
    p1 = Poly1dFunction.from_coeffs([5.0, 4])
    p2 = Poly1dFunction.from_coeffs([2.0, 3, 4])

    assert_equal((p1 * p2)(3), 19 * 31)
    assert_equal((p1 + p2)(3), 19 + 31)


test_main(True)
