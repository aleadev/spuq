from spuq.polynomials.polynomial_family import PolynomialFamily
import numpy

class StochasticHermitePolynomials(PolynomialFamily):
    def __init__( self ):
        self.structcoeffs = numpy.empty((0, 0, 0))
    
    def getCoefficients( self,  n ):
        h = self.eval( n,  poly1d([1, 0]) )
        return h.coeffs[::-1]

    def eval( self, n,  x ):
        if n==0:
            return 0*x+1        # ensure return value has the right format
        elif n==1:
            return x
        else:
            h0=1
            h1=x
            for i in xrange(2,n+1):
                h1,h0=x*h1-(i-1)*h0,h1
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
        from scipy import factorial
        n = max((a, b, c))
        if n <= self.structcoeffs.shape[0]:
            return self.structcoeffs[a, b, c]
        else:
            s = a+b+c
            if bool(s % 1) or a<=b+c or b<=a+c or c<=a+b:
                c = 0
            else:
                s /= 2
                c = factorial(s-a)*factorial(s-b)*factorial(s-c)/(factorial(a)*factorial(b)*factorial(c))


import unittest
class TestHermite(unittest.TestCase):
    def test_hermite(self):
        x = 3.14159
        h = StochasticHermitePolynomials()
        self.assertAlmostEquals( h.eval(3, x), x**3-3*x )
    
    
if __name__=="__main__":
    x=numpy.poly1d([1,0])
    #x=10
    print x
    print x*x
    print x*x*x
    H = StochasticHermitePolynomials()
    print H.eval(3,x).coeffs[::-1]
    # start unit test
    unittest.main()
