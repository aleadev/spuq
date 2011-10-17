import numpy as np

from spuq.utils.type_check import InputParameterError
from spuq.utils.testing import *
from spuq.linalg.operator import *


class TestMatrixOperator(TestCase):

    def test_init(self):
        l = [[1, 2, 4], [3, 4, 5]]
        arri = np.array(l, dtype=int)
        arr = np.array(l, dtype=float)
        A1 = MatrixOperator(l)
        A2 = MatrixOperator(arri)
        A3 = MatrixOperator(arr)

        assert_equal(A1.codomain, CanonicalBasis(2))
        assert_equal(A1.domain, CanonicalBasis(3))
        assert_array_equal(A1.as_matrix(), arr)

        assert_raises(TypeError, MatrixOperator, arr, 2)
        assert_raises(TypeError, MatrixOperator, [2])

    def xxxtest_equal(self):
        l = [[1, 2, 4], [3, 4, 5]]
        arr = np.array(l, dtype=float)
        A1 = MatrixOperator(l)
        A2 = MatrixOperator(arr)
        A3 = MatrixOperator([[1, 2, 3], [4, 5, 6]])

        assert_true(A1 == A2)
        assert_true(not (A1 != A2))
        assert_true(A1 != A3)
        assert_true(not (A1 == A3))

    def xxxtest_apply(self):
        arr = np.array([[1, 2, 4], [3, 4, 5]], dtype=float)
        A = MatrixOperator(self.arr)
        vec1 = FlatVector([2, 3, 4])
        vec2 = FlatVector([24, 38])
        assert_equal(A * vec1, vec2)


test_main()
