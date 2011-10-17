from spuq.utils.testing import *
from spuq.linalg.basis import *


class TestBasis(TestCase):

    def test_abstract(self):
        assert_raises(TypeError, Basis)
        assert_raises(TypeError, FunctionBasis)
        pass

class TestCanonicalBasis(TestCase):

    def test_init(self):
        CanonicalBasis(5)

    def test_compare(self):
        b5 = CanonicalBasis(5)
        b7a = CanonicalBasis(7)
        b7b = CanonicalBasis(7)

        assert_true(b5 == b5)
        assert_true(not (b5 != b5))
        assert_true(b5 != b7a)
        assert_true(not (b5 == b7a))

        assert_not_equal(b5, b7a)
        assert_equal(b7a, b7b)

    def test_dim(self):
        assert_equal(CanonicalBasis(3).dim, 3)
        assert_equal(CanonicalBasis(13).dim, 13)

    def test_check_basis(self):
        b5 = CanonicalBasis(5)
        b7 = CanonicalBasis(7)

        assert_raises(BasisMismatchError, check_basis, b5, b7)
        exc = assert_raises(BasisMismatchError, check_basis, 
                            b5, b7, "basisabc", "basisxyz")
        assert_true(str(exc).find("basisabc") > -1)
        assert_true(str(exc).find("basisxyz") > -1)

    def test_repr(self):
        b5 = CanonicalBasis(5)
        assert_equal(str(b5), "<CanonicalBasis dim=5>")

        FooBasis = type("FooBasis", (CanonicalBasis,), {})
        b5 = FooBasis(5)
        assert_equal(str(b5), "<FooBasis dim=5>")


test_main()
