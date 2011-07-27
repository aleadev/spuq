from numpy.testing import *
from numpy import array
from spuq.utils.multiindex_set import MultiindexSet

class TestMultiindexSet(TestCase):
    def setUp(self):
        self.mi1 = MultiindexSet.createCompleteOrderSet( 1,  1)
        self.mi2 = MultiindexSet.createCompleteOrderSet( 2,  3)
        self.mi3 = MultiindexSet( array([[1, 1], [2, 3]]))
    def test_mp(self):
        self.assertEqual( self.mi1.m, 1 )
        self.assertEqual( self.mi1.p, 1 )
        self.assertEqual( self.mi2.m, 2 )
        self.assertEqual( self.mi2.p, 3 )
    def test_count(self):
        self.assertEqual( self.mi1.count, 2 )
        self.assertEqual( self.mi2.count, 10 )
    def test_repr(self):
        pass # todo
    def test_power(self):
        p = self.mi3.power(array([5,7]))
        self.assertTrue( (p==array([35, 8575])).all() )
    def test_factorial(self):
        f = self.mi3.factorial()
        self.assertTrue( (f==array([1,12])).all() )

if __name__ == "__main__":
    pass
