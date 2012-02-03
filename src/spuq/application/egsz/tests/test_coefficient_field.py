import numpy as np

from spuq.utils.testing import *
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV, ArcsineRV
from spuq.application.egsz.coefficient_field import CoefficientField


def test_init():
    cnf = ConstFunction(1)
    snf = SimpleFunction(np.sin)
    csf = SimpleFunction(np.cos)
    uni = UniformRV()
    a1 = [cnf, snf, csf]
    a2 = [cnf, snf]
    rvs = [uni, NormalRV()]

    cf = CoefficientField(a1, rvs)
    assert_equal(cf._funcs[0], cnf)
    assert_equal(len(cf._rvs), 3)
    assert_equal(cf._rvs[1], uni)
    assert_equal(len(rvs), 2)

    assert_raises(AssertionError, CoefficientField, a2, rvs)
    assert_raises(TypeError, CoefficientField, [1, 2], rvs)
    assert_raises(TypeError, CoefficientField, a1[1], rvs)
    assert_raises(TypeError, CoefficientField, a2, [1, 2])
    assert_raises(TypeError, CoefficientField, a2, rvs[0])


def test_create_iid():
    cnf = ConstFunction(1)
    snf = SimpleFunction(np.sin)
    csf = SimpleFunction(np.cos)
    rv = ArcsineRV()
    a1 = [cnf, snf, csf]

    cf = CoefficientField.createWithIidRVs(a1, rv)
    assert_equal(len(cf._funcs), 3)
    assert_equal(cf._funcs[0], cnf)
    assert_equal(cf._funcs[2], csf)
    a1[1] = csf
    assert_equal(cf._funcs[1], snf)
    assert_equal(len(cf._rvs), 3)
    assert_not_equal(cf._rvs[0], rv)
    assert_equal(cf._rvs[1], rv)
    assert_equal(cf._rvs[2], rv)


def test_coefficients():
    a1 = [ConstFunction(1), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]

    cf = CoefficientField(a1, rvs)
    a2 = [a for (a, _) in cf.coefficients()]
    assert_equal(a2, a1)
    rvs2 = [rv for (_, rv) in cf.coefficients()]
    assert_equal(rvs2[1:], rvs)


def test_len_getitem_repr():
    a1 = [ConstFunction(1), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    cf = CoefficientField(a1, rvs)
    assert_equal(len(cf), 3)
    assert_equal(cf[1], (a1[1], rvs[0]))
    assert_true(str(cf).startswith("<CoefficientField funcs="))
    assert_true(str(cf).endswith(">"))


test_main()

