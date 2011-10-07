import numpy as np

from spuq.utils.type_check import InputParameterError
from spuq.utils.testing import *
from spuq.linalg.operator import *


class TestMatrixOperator(TestCase):

    def setUp(self):
        self.arr1 = np.array([[1, 2, 4], [3, 4, 5]], dtype=float)
        self.vec1 = np.array([2, 3, 4], dtype=float)
        self.vec2 = np.array([24, 38], dtype=float)

    def test_init(self):
        """Make sure ..."""
        A = MatrixOperator(self.arr1)
        assert_equal(A.codomain.dim, 2)
        assert_equal(A.domain.dim, 3)
        assert_array_equal(A.as_matrix(), self.arr1)

        assert_raises(TypeError, lambda: MatrixOperator(self.arr1, 2))
        assert_raises(TypeError, MatrixOperator, self.arr1, 2)
        assert_raises(TypeError, MatrixOperator, [2])

        vec1 = FlatVector(self.vec1)
        vec2 = FlatVector(self.vec2)
        assert_equal(A * vec1, vec2)

if __name__ == "__main__":
    run_module_suite()
