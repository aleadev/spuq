from polynomial_basis import PolynomialBasis

class PCBasis(PolynomialBasis):
    # we already have that (normalised or not?)
    pass


def hermite( n,  x ):
    if n==0:
        return 1
    elif n==1:
        return x
    else:
        h0=1
        h1=x
        for i in xrange(2,n+1):
            h1,h0=x*h1-(i-1)*h0,h1
        return h1
    
    
import unittest
class TestHermite(unittest.TestCase):
    def test_hermite(self):
        x=3.14159
        self.assertEquals( hermite(3, x), x**3-3*x )
    
    
if __name__=="__main__":
    import numpy
    x=numpy.poly1d([1,0])
    #x=10
    print x
    print x*x
    print x*x*x
    print hermite(3,x).c[::-1]
    
    #unittest.main()
    
