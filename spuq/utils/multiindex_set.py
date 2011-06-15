class MultiindexSet(object):
    def __init__(self, arr):
        self.arr=arr
        self.m=arr.shape[1]
        self.p=max(arr.sum(1))
        self.count=arr.shape[0]
        pass
    
    def power(self,vec):
        for i in xrange(vec.size):
            if i==0:
                res=vec[i]**self.arr[:,i]
            else:
                res=res*vec[i]**self.arr[:,i]
        return res
        
    def factorial(self):
        from scipy import factorial
        return factorial(self.arr).prod(1)

    @staticmethod
    def createCompleteOrderSet(m,p):
        def createMultiindexSet(m, p):
            from numpy import int8,  zeros,  vstack, hstack
            if m==0:
                return zeros( (1, 0),  int8 )
            else:
                I=zeros( (0, m),  int8 )
                for q in xrange(0, p+1):
                    J=createMultiindexSet(m-1, q)
                    Jn=q-J.sum(1).reshape((J.shape[0],1))
                    I=vstack( (I,  hstack( (J, Jn))))
                return I
        arr=createMultiindexSet(m,p)
        return MultiindexSet(arr)
        
    def __repr__(self):
        return "MI(m={0},p={1},arr={2})".format(self.m,self.p,self.arr)

import unittest
from numpy import array

class TestMultiindexSet(unittest.TestCase):
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
    unittest.main()
