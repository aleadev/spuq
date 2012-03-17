import numpy as np

from spuq.utils.type_check import InputParameterError
from spuq.utils.testing import *
from spuq.linalg.operator import *
from spuq.linalg.test_support import *

def test_matrixop_init():
    l = [[1, 2, 4], [3, 4, 5]]
    arri = np.array(l, dtype=int)
    arr = np.array(l, dtype=float)

    A1 = MatrixOperator.from_sequence(l)
    assert_equal(A1.codomain, CanonicalBasis(2))
    assert_equal(A1.domain, CanonicalBasis(3))
    assert_array_equal(A1.as_matrix(), arr)

    MatrixOperator(arr)
    MatrixOperator(arr, domain=FooBasis(3))

    assert_raises(TypeError, MatrixOperator, arr, 2)
    assert_raises(TypeError, MatrixOperator, [2])

    assert_raises(TypeError, MatrixOperator, arr, domain=CanonicalBasis(5))
    assert_raises(TypeError, MatrixOperator, arr, codomain=CanonicalBasis(5))

def test_matrixop_equal():
    arr1 = np.array([[1, 2, 4], [3, 4, 5]], dtype=float)
    arr2 = np.array([[1, 2, 4], [3, 4, 5]], dtype=float)
    arr3 = np.array([[1, 2, 3], [3, 4, 6]], dtype=float)
    A1 = MatrixOperator(arr1)
    A2 = MatrixOperator(arr2)
    A3 = MatrixOperator(arr3)

    assert_true(A1 == A2)
    assert_true(not (A1 != A2))
    assert_true(A1 != A3)
    assert_true(not (A1 == A3))

def test_matrixop_apply():
    arr = np.array([[1, 2, 4], [3, 4, 5]], dtype=float)
    A = MatrixOperator(arr)
    vec1 = FlatVector([2, 3, 4])
    vec2 = FlatVector([24, 38])
    assert_equal(A * vec1, vec2)

def test_compose_types():
    A = MatrixOperator(1 + rand(3, 5))
    B = MatrixOperator(1 + rand(7, 3))

    # operators can be multiplied
    C = B * A
    assert_equal(C.domain, A.domain)
    assert_equal(C.codomain, B.codomain)

    A = MatrixOperator(1 + rand(3, 5), domain=FooBasis(5))
    C = B * A
    assert_equal(C.domain, A.domain)

    A = MatrixOperator(1 + rand(3, 5), codomain=FooBasis(3))
    assert_raises(Exception, Operator.__mul__, B, A)


    A = MatrixOperator(rand(3, 5), domain=FooBasis(5), codomain=FooBasis(3))
    B = MatrixOperator(rand(7, 3), domain=FooBasis(3), codomain=FooBasis(7))
    C = B * A
    assert_equal(C.domain, A.domain)
    assert_equal(C.codomain, B.codomain)

def test_compose_value():
    A = MatrixOperator(1 + rand(3, 5))
    B = MatrixOperator(1 + rand(7, 3))
    x = FlatVector(rand(5))
    y = B(A(x))

    assert_equal((B * A)(x), y)
    assert_equal(B * A * x, y)
    assert_equal(B * (A * x), y)
    assert_equal((B * A) * x, y)

    assert_equal((B * A).as_matrix(), B.as_matrix() * A.as_matrix())

def test_compose_transpose():
    A = MatrixOperator(1 + rand(3, 5))
    B = MatrixOperator(1 + rand(7, 3))
    C = B * A

    AT = A.transpose()
    BT = B.transpose()
    CT = C.transpose()

    y = FlatVector(rand(CT.domain.dim))
    assert_equal(CT * y, AT * (BT * y))


def test_add_sub_operators():
    A = MatrixOperator(1 + rand(3, 5))
    B = MatrixOperator(1 + rand(3, 5))
    x = FlatVector(rand(5))

    assert_equal((A + B) * x, A * x + B * x)
    assert_equal((A - B) * x, A * x - B * x)
    assert_equal((2.5 * A) * x, 2.5 * (A * x))
    assert_equal((25 * A) * x, 25 * (A * x))
    assert_equal((A * 25) * x, 25 * (A * x))


def test_diag_operator_apply():
    diag = np.array([1, 2, 3])
    x = FlatVector([3, 4, 5])
    y = FlatVector([3, 8, 15])
    A = DiagonalMatrixOperator(diag)
    assert_equal(A * x, y)

def assert_operator_consistency():
    A = MatrixOperator.from_sequence([[1, 2], [3, 4]], FooBasis(2), BarBasis(2))
    assert_operator_is_consistent(A)

    A = MatrixOperator.from_sequence([[1, 2], [3, 4], [4, 5]])
    assert_operator_is_consistent(A)

    A = MultiplicationOperator(3.0, FooBasis(4))
    assert_operator_is_consistent(A)




test_main()
