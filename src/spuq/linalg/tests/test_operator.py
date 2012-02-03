import numpy as np

from spuq.utils.type_check import InputParameterError
from spuq.utils.testing import *
from spuq.linalg.operator import *

class FooBasis(CanonicalBasis):
    """Dummy basis class for testing that operator methods correctly
    respect bases."""
    pass

def test_matrixop_init():
    l = [[1, 2, 4], [3, 4, 5]]
    arri = np.array(l, dtype=int)
    arr = np.array(l, dtype=float)
    A1 = MatrixOperator(l)
    A2 = MatrixOperator(arri)
    A3 = MatrixOperator(arr)
    A4 = MatrixOperator(arr, domain=FooBasis(3))

    assert_equal(A1.codomain, CanonicalBasis(2))
    assert_equal(A1.domain, CanonicalBasis(3))
    assert_array_equal(A1.as_matrix(), arr)

    assert_raises(TypeError, MatrixOperator, arr, 2)
    assert_raises(TypeError, MatrixOperator, [2])

    assert_raises(TypeError, MatrixOperator, arr, domain=CanonicalBasis(5))
    assert_raises(TypeError, MatrixOperator, arr, codomain=CanonicalBasis(5))

def test_matrixop_equal():
    l = [[1, 2, 4], [3, 4, 5]]
    arr = np.array(l, dtype=float)
    A1 = MatrixOperator(l)
    A2 = MatrixOperator(arr)
    A3 = MatrixOperator([[1, 2, 3], [4, 5, 6]])

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
    x = FlatVector(rand(5, 1))
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

    y = FlatVector(rand(CT.domain.dim, 1))
    assert_equal(CT * y, AT * (BT * y))


def test_add_sub_operators():
    A = MatrixOperator(1 + rand(3, 5))
    B = MatrixOperator(1 + rand(3, 5))
    x = FlatVector(rand(5, 1))

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




test_main()
