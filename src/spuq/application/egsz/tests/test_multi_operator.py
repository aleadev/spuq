import numpy as np

from spuq.utils.testing import *
from spuq.math_utils.multiindex import Multiindex
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.linalg.vector import FlatVector
from spuq.linalg.operator import DiagonalMatrixOperator
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV

def diag_assemble(func, basis):
    import sys
    x = np.linspace(0, 1, basis.dim)
    diag = np.array([np.abs(func(y)) for y in x])
    return DiagonalMatrixOperator(diag)

def test_init():
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    MultiOperator(coeff_field, diag_assemble)
    assert_raises(TypeError, 3, diag_assemble)
    assert_raises(TypeError, coeff_field, 7)

def test_apply():
    N = 4
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    A = MultiOperator(coeff_field, diag_assemble)
    vec1 = FlatVector(np.random.random(N))
    vec2 = FlatVector(np.random.random(N))

    mis = Multiindex.createCompleteOrderSet(2, 3)
    print mis[0]
    print mis[1]
    print mis[2]
    print mis[3]
    w = MultiVectorWithProjection()
    w[mis[0]] = vec1
    w[mis[1]] = vec2
    w[mis[2]] = vec1
    w[mis[3]] = vec2
    print w
    v = A * w
    for i in range(4):
        print w[mis[i]]
        print v[mis[i]]
        print

    #A*mv




test_main()
