class UniformDistribution(ProbabilityDistribution):
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def mean( self ): return self.a+self.b
    def var( self ): pass
    def skew( self ): pass
    def excess( self ): pass
    def median( self ): pass
    def pdf( self, x ): pass
    def cdf( self, x ): pass
    def invcdf( self, x ): pass
    def __repr__(self):
        return "U["+str(self.a)+","+str(self.b)+"]"
