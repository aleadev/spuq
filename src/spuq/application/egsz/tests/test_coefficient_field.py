import numpy as np
from itertools import count

from spuq.utils.testing import *
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV, ArcsineRV
from spuq.application.egsz.coefficient_field import ListCoefficientField, ParametricCoefficientField


def test_init():
    cnf = ConstFunction(1)
    snf = SimpleFunction(np.sin)
    csf = SimpleFunction(np.cos)
    uni = UniformRV()
    mean = cnf
    a1 = [cnf, snf]
    a2 = [cnf, snf, csf]
    rvs = [uni, NormalRV()]

    cf = ListCoefficientField(cnf, a1, rvs)

    assert_raises(AssertionError, ListCoefficientField, mean, a2, rvs)
    assert_raises(TypeError, ListCoefficientField, mean, cnf, rvs)
    assert_raises(TypeError, ListCoefficientField, mean, a1, [1, 2])
    assert_raises(TypeError, ListCoefficientField, mean, a1, rvs[0])


def test_create_iid():
    cnf = ConstFunction(1)
    snf = SimpleFunction(np.sin)
    csf = SimpleFunction(np.cos)
    mean = cnf
    rv = ArcsineRV()
    a1 = [snf, csf]

    cf = ListCoefficientField.create_with_iid_rvs(mean, a1, rv)
    assert_equal(len(cf), 2)
    assert_equal(mean, cnf)
    assert_equal(len(cf.funcs), 2)
    assert_equal(cf.rvs[0], rv)
    assert_equal(cf.rvs[1], rv)


def test_len_getitem_repr():
    mean = ConstFunction(1)
    a1 = [SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    cf = ListCoefficientField(mean, a1, rvs)
    assert_equal(len(cf), 2)
    assert_equal(cf[0], (a1[0], rvs[0]))
    assert_equal(cf[1], (a1[1], rvs[1]))
    assert_true(str(cf).startswith("<ListCoefficientField mean="))
    assert_true(str(cf).endswith(">"))


def test_parametric():
    cnf = ConstFunction(1)
    snf = SimpleFunction(np.sin)
    csf = SimpleFunction(np.cos)

    def func_func(i):
        return [cnf, snf, csf][i % 3]

    mean = cnf

    urv = UniformRV()
    nrv = NormalRV()

    def rv_func(i):
        return [urv, nrv][i % 2]


    cf = ParametricCoefficientField(mean, func_func, rv_func)
    assert_true(len(cf) > 100000) # infty doesn't work
    assert_equal(cf.mean_func, cnf)
    assert_equal(cf[0], (cnf, urv))
    assert_equal(cf[1], (snf, nrv))
    assert_equal(cf[17], (csf, nrv))


test_main()

