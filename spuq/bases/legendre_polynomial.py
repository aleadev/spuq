import numpy
from polynomial_family import PolynomialFamily

class LegendrePolynomial(PolynomialFamily):
    def __init__( self ):
        self.structcoeffs = numpy.empty((0, 0, 0))
    
    def getCoefficients( self,  n ):
        l = self.eval( n,  poly1d([1, 0]) )
        return l.coeffs[::-1]

    def eval( self, n,  x ):
        if n==0:
            return 0*x+1        # ensure return value has the right format
        elif n==1:
            return x
        else:
            h0=1
            h1=x
            for i in xrange(2,n+1):
                h1, h0 = (2*i-1.0)*x*h1/i - (i-1.0)*h0/i, h1
            return h1
    
    def getStructureCoefficients( self, n ):
        if n > self.structcoeffs.shape[0]:
            j = self.structcoeffs[0]
            self.structcoeffs.resize((n, n, n));
            for a in xrange(n):
                for b in xrange(n):
                    for c in xrange(n):
                        self.structcoeffs[a, b, c] = self.getStructureCoefficient( a, b, c )
        return self.structcoeffs[0:n, 0:n, 0:n]
        
    def getStructureCoefficient( self, a, b, c ):
        return NotImplemented

import unittest
class TestLegendre(unittest.TestCase):
    def test_legendre(self):
        x = 3.14159
        l = LegendrePolynomial()
        print l.eval(0, numpy.poly1d([1, 0]))
        print l.eval(1, numpy.poly1d([1, 0]))
        print l.eval(3, numpy.poly1d([1, 0]))
        self.assertEquals( l.eval(3, x), 2.5*x**3-1.5*x )

if __name__ == "__main__":
    unittest.main()
    
