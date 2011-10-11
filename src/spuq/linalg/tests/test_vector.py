import numpy as np

from spuq.utils.testing import *
from spuq.linalg.vector import *
from spuq.linalg.basis import *


class FooBasis(CanonicalBasis):
    """Dummy basis class for testing that vector methods correcly
    check for equality of bases."""
    pass


class TestVector(TestCase):

    def test_abstract(self):
        assert_raises(TypeError, Vector)


class TestFlatVector(TestCase):

    def test_init(self):
        arr = np.array([1.0, 2, 3])
        fv1 = FlatVector(arr)
        fv2 = FlatVector([1.0, 2.0, 3.0])
        fv3 = FlatVector([1, 2, 3])
        fv4 = FlatVector([1, 2, 3], CanonicalBasis(3))
        assert_raises(TypeError, FlatVector, ["str", "str"])
        assert_raises(TypeError, FlatVector, [1, 2, 3], object)

    def test_as_array(self):
        fv1 = FlatVector([1, 2, 3])
        assert_equal(fv1.as_array(), np.array([1.0, 2, 3]))
        assert_is_instance(fv1.as_array()[0], float)

    def test_equals(self):
        fv1 = FlatVector([1, 2, 3])
        fv2 = FlatVector([1, 2, 3])
        fv3 = FlatVector([1, 2])
        fv4 = FlatVector([1, 2, 4])
        fv5 = FlatVector([1, 2, 3], FooBasis(3))

        # make sure both operators are overloaded
        assert_true(fv1 == fv2)
        assert_false(fv1 != fv2)
        assert_true(fv1 != fv3)
        assert_false(fv1 == fv3)

        # now test for (in)equality
        assert_equal(fv1, fv2)
        assert_not_equal(fv1, fv3)
        assert_not_equal(fv1, fv4)
        assert_not_equal(fv1, fv5)
        assert_equal(fv5, fv5)

    def test_add(self):
        fv1 = FlatVector(np.array([1.0, 2, 3]))
        fv2 = FlatVector(np.array([2, 4, 6]))
        assert_equal(fv1 + fv1, fv2)

        fv3 = FlatVector([1, 2])
        fv4 = FlatVector([1, 2, 3], FooBasis(3))
        assert_raises(BasisMismatchError, lambda: fv1 + fv3)
        assert_raises(BasisMismatchError, lambda: fv1 + fv4)

    def test_sub(self):
        b = FooBasis(3)
        fv1 = FlatVector(np.array([5, 7, 10]), b)
        fv2 = FlatVector(np.array([2, 4, 6]), b)
        fv3 = FlatVector(np.array([3, 3, 4]), b)
        assert_equal(fv1 - fv2, fv3)

        # test the __sub__ method of the base class
        del FlatVector.__sub__
        assert_equal(fv1 - fv2, fv3)

    def test_mul(self):
        fv1 = FlatVector(np.array([1.0, 2, 3]))
        fv2 = FlatVector(np.array([2.5, 5, 7.5]))
        fv3 = FlatVector(np.array([2, 4, 6]))
        assert_equal(2.5 * fv1, fv2)
        assert_equal(2 * fv1, fv3)
        assert_equal(fv1 * 2.5, fv2)
        assert_equal(fv1 * 2, fv3)
        assert_raises(lambda: fv1 * fv3)

        fv4 = FlatVector([1, 2, 3], FooBasis(3))
        fv5 = FlatVector([2, 4, 6], FooBasis(3))
        assert_equal(fv4 * 2, fv5)
        assert_equal(2 * fv4, fv5)

    def test_repr(self):
        fv1 = FlatVector(np.array([1.0, 2, 3]))
        assert_equal(str(fv1),
                     "<FlatVector basis=<CanonicalBasis dim=3>, " +
                     "coeffs=[ 1.  2.  3.]>")

if __name__ == "__main__":
    run_module_suite()
