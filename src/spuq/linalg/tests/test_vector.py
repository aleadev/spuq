import numpy as np

from spuq.utils.testing import *
from spuq.linalg.vector import FlatVector


class TestFlatVector(TestCase):

    def test_equals(self):
        """Make sure we can compare vectors for equality
        """
        fv1 = FlatVector(np.array([1.0, 2, 3]))
        fv2 = FlatVector(np.array([1.0, 2, 3]))
        fv3 = FlatVector(np.array([1.0, 2]))
        fv4 = FlatVector(np.array([1.0, 2, 4]))

        assert_true(fv1==fv2)
        assert_false(fv1!=fv2)

        assert_equal(fv1, fv2)
        assert_not_equal(fv1, fv3)
        assert_not_equal(fv1, fv4)

    def test_mul(self):
        """Make sure we can multiply vectors with scalars
        """
        fv1 = FlatVector(np.array([1.0, 2, 3]))
        fv2 = FlatVector(np.array([2.5, 5, 7.5]))
        assert_equal(2.5 * fv1, fv2)
        assert_equal(fv1 * 2.5, fv2)


if __name__ == "__main__":
    run_module_suite()
