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
