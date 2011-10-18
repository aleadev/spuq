import numpy as np

from spuq.utils.testing import *
from spuq.utils.multiindex_set import *


class TestMultiindexSet(TestCase):

    def setUp(self):
        self.mi1 = MultiindexSet.createCompleteOrderSet(1,  1)
        self.mi2 = MultiindexSet.createCompleteOrderSet(2,  3)
        self.mi3 = MultiindexSet(np.array([[1, 1], [2, 3]]))

    def test_mp(self):
        assert_equal(self.mi1.m, 1)
        assert_equal(self.mi1.p, 1)
        assert_equal(self.mi2.m, 2)
        assert_equal(self.mi2.p, 3)

    def test_count(self):
        assert_equal(self.mi1.count, 2)
        assert_equal(self.mi2.count, 10)

    def test_power(self):
        p = self.mi3.power(np.array([5, 7]))
        assert_true((p == np.array([35, 8575])).all())

    def test_factorial(self):
        f = self.mi3.factorial()
        assert_true((f == np.array([1, 12])).all())


test_main()
