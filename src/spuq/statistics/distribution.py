class Distribution(object):
    def pdf( self, x ): 
        return NotImplemented
    def cdf( self, x ): 
        return NotImplemented
    def invcdf( self, x ): 
        return NotImplemented
    
    def mean( self ): 
        return NotImplemented
    def var( self ): 
        return NotImplemented
    def skew( self ): 
        return NotImplemented
    def excess( self ): 
        return NotImplemented
        
    def median( self ): 
        return NotImplemented
    
    def getOrthogonalPolynomials(self):
        return NotImplemented
        
    def sample(self, size):
        return NotImplemented
