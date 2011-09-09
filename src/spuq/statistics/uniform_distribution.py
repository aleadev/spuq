from distribution import Distribution

class UniformDistribution(Distribution):
    def __init__(self,a=-1,b=1):
        self.a=1.0*min(a, b)
        self.b=1.0*max(a, b)
        
    def pdf( self, x ):
        return 1.0*(self.a<=x)*(x<=self.b) / (self.b-self.a)
        
    def cdf( self, x ): 
        a=self.a
        b=self.b
        x=x+(x<a)*(a-x)-(x>b)*(x-b)
        c=1.0*(x-self.a)/(self.b-self.a)
        return c
        
    def invcdf( self, x ): 
        pass


    def mean( self ): 
        return 0.5*(self.a+self.b)
    
    def var( self ):
        return 1.0/12.0*(self.b-self.a)**2
        
    def skew( self ): 
        return 0
        
    def kurtex( self ):
        return -6/5.0
        
    def median( self ): 
        return 0.5*(self.a+self.b)


    def shift(self, delta):
        return UniformDistribution(self.a+delta, self.b+delta)
        
    def scale(self, scale):
        m=0.5*(self.a+self.b)
        d=scale*0.5*(self.b-self.a)
        return UniformDistribution(m-d, m+d)

    def getOrthogonalPolynomials(self):
        assert( self.a==-1.0 and self.b==1.0 )
        from spuq.polynomials.legendre_polynomials import LegendrePolynomials
        return LegendrePolynomials()
        
    def sample(self,  size):
        from numpy.random import uniform
        return uniform( self.a,  self.b,  size )

    def __repr__(self):
        return "U["+str(self.a)+","+str(self.b)+"]"



import unittest
class TestUniformDistribution(unittest.TestCase):
    def setUp(self):
        from uniform_distribution import UniformDistribution
        self.ud1=UniformDistribution()
        self.ud2=UniformDistribution(2, 6)
    def test_mean(self):
        self.assertEquals( self.ud1.mean(), 0 )
        self.assertEquals( self.ud2.mean(), 4 )
    def test_var(self):
        self.assertEquals( self.ud1.var(), 1.0/3.0 )
        self.assertEquals( self.ud2.var(), 4.0/3.0 )
    def test_pdf(self):
        from numpy import exp, sqrt, pi
        self.assertEquals( self.ud1.pdf(0.5), 0.5 )
        self.assertEquals( self.ud2.pdf(1.8), 0 )
        self.assertEquals( self.ud2.pdf(6.2), 0 )
    def test_cdf(self):
        from scipy import inf
        self.assertEquals( self.ud1.cdf(0),  0.5)
        #self.assertEquals( self.ud1.cdf(inf),  1)
        #self.assertEquals( self.ud1.cdf(-inf),  0)
        self.assertEquals( self.ud1.cdf(1000),  1)
        self.assertEquals( self.ud1.cdf(-1000),  0)
        self.assertEquals( self.ud2.cdf(3),  0.25)
    def test_sample(self):
        s=self.ud1.sample(5)
        self.assertEquals( s.shape,  (5, ) )
    def test_shift_scale(self):
        d=self.ud2
        self.assertEquals( d.shift(2).mean(), d.mean()+2  )
        self.assertEquals( d.shift(2).var(), d.var() )
        self.assertEquals( d.scale(2).mean(), d.mean()  )
        self.assertEquals( d.scale(2).var(), d.var()*4 )
            

if __name__=="__main__":
        unittest.main()
