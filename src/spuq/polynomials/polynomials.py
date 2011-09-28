class PolynomialFamily(object):
    """abstract base for polynomials"""

    def getCoefficients( self, n ):
        """return coefficients of polynomial"""
        return NotImplemented

    def eval( self, n,  x ):
        """evaluate polynomial of degree n at points x"""
        return NotImplemented
        
    def getStructureCoefficients( self, n ):
        """return structure coefficients of indices up to n"""
        return NotImplemented
        
    def getStructureCoefficient( self, a, b, c ):
        """return specific structure coefficient"""
        return NotImplemented
    
    def norm( self, n ):
        """returns norm of polynomial"""
        return NotImplemented
        
    def isNormalised(self):
        """return True if polynomials are normalised"""
        return False
    
    
class NormalisedPolynomialFamily(PolynomialFamily):
    def getCoefficients( self, n ):
        """return normalised coefficients of polynomial"""
        return PolynomialFamily.getCoefficients( n )/self.norm()

    def eval( self, n,  x ):
        """evaluate normalised polynomial of degree n at points x"""
        return PolynomialFamily.eval(n, k)/self.norm()
        
    def getStructureCoefficients( self, n ):
        """return normalised structure coefficients of indices up to n"""
        return PolynomialFamily.getCoefficients(n)/self.norm()
        
    def getStructureCoefficient( self, a, b, c ):
        """return specific normalised structure coefficient"""
        return PolynomialFamily.getStructureCoefficient(a, b, c)/self.norm()
        
    def isNormalised(self):
        """returns True"""
        return True
import numpy
from spuq.polynomials.polynomial_family import PolynomialFamily

class LegendrePolynomials(PolynomialFamily):
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
        l = LegendrePolynomials()
        print l.eval(0, numpy.poly1d([1, 0]))
        print l.eval(1, numpy.poly1d([1, 0]))
        print l.eval(3, numpy.poly1d([1, 0]))
        self.assertEquals( l.eval(3, x), 2.5*x**3-1.5*x )

if __name__ == "__main__":
    unittest.main()
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
