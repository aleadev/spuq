import numpy as np

from spuq.utils.testing import *
from spuq.math_utils.multiindex_set import *


def test_init():
    mi = MultiindexSet(np.array([[1, 1], [2, 3]]))
    assert_equal(mi.m, 2)
    assert_equal(mi.p, 5)


def test_create_comp_order():
    mi = createCompleteOrderSet(1,  1)
    assert_equal(mi.m, 1)
    assert_equal(mi.p, 1)
    assert_equal(mi.count, 2)

    mi = createCompleteOrderSet(2,  3)
    assert_equal(mi.m, 2)
    assert_equal(mi.p, 3)
    assert_equal(mi.count, 10)

    mi = createCompleteOrderSet(0,  2)
    assert_equal(mi.m, 0)
    assert_equal(mi.p, 0)
    assert_equal(mi.count, 1)

    mi = createCompleteOrderSet(5,  0)
    assert_equal(mi.m, 5)
    assert_equal(mi.p, 0)
    assert_equal(mi.count, 1)

    mi = createCompleteOrderSet(7,  5)
    assert_equal(mi.count, 792)


def test_index():
    mi = MultiindexSet(np.array([[1, 1], [2, 3], [3, 6], [1, 7]]))
    assert_equal(mi[0], np.array([1, 1]))
    assert_equal(mi[2], np.array([3, 6]))


def test_power():
    mi = MultiindexSet(np.array([[1, 1], [2, 3]]))
    p = mi.power(np.array([5, 7]))
    assert_equal(p, np.array([35, 8575]))


def test_factorial():
    mi = MultiindexSet(np.array([[1, 1], [2, 3]]))
    f = mi.factorial()
    assert_equal(f, np.array([1, 12]))


def test_repr():
    mi = createCompleteOrderSet(2,  3)
    repr = str(mi)
    assert_true(repr.startswith("<MISet m=2, p=3, arr=[[0 0]"))


test_main()
