import numpy as np

from spuq.utils.testing import *
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.stochastics.random_variable import NormalRV
from spuq.application.egsz.multi_vector import MultiVector
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.linalg.vector import FlatVector
from spuq.linalg.operator import DiagonalMatrixOperator
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV

N = 4

def diag_assemble(func):
    import sys
    x = np.linspace(0, 1, N)
    diag = np.array([np.abs(func(y)) for y in x])
    return DiagonalMatrixOperator(diag)

def test_init():
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    A = MultiOperator(coeff_field, diag_assemble)
    assert_raises(TypeError, 3, diag_assemble)
    assert_raises(TypeError, coeff_field, 7)

def test_apply():
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    A = MultiOperator(coeff_field, diag_assemble)
    vec1 = FlatVector(np.random.rand(N, 1))
    vec2 = FlatVector(np.random.rand(N, 1))

    print vec1
    print diag_assemble(a[2]) * vec1
    print vec2
    print diag_assemble(a[1]) * vec2

    #A*mv




test_main()
