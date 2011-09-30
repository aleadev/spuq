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
class ShiftedDistribution(ProbabilityDistribution):
    def __init__(self,dist,delta):
        self.dist=dist
        self.delta=delta
    def mean(self):
        return dist.mean()+delta
    def var(self):
        return dist.var()
    def __repr__(self):
        return self.dist.__repr__()+"+"+str(self.delta)
    
class ShiftedDistribution(ProbabilityDistribution):
    def __init__(self,dist,delta):
        self.dist=dist
        self.delta=delta
    def mean(self):
        return dist.mean()+delta
    def var(self):
        return dist.var()
    def __repr__(self):
        return self.dist.__repr__()+"+"+str(self.delta)
    
