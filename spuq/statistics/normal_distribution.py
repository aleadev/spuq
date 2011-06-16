from distribution import Distribution

class NormalDistribution(Distribution):
    def __init__(self,  mu=0,  sigma=1 ):
        self.mu=mu
        self.sigma=sigma
    
    def pdf(self,  x):
        from numpy import exp, sqrt, pi
        return exp( -0.5*((x-self.mu)/self.sigma)**2)/sqrt(2.0*pi)/self.sigma

    def cdf(self,  x):
        from scipy.special import erfc,  sqrt
        return 0.5*erfc(-sqrt(0.5)*(x-self.mu)/self.sigma)

    def mean(self):
        return self.mu
        
    def var(self):
        return self.sigma**2
        
    def getOrthogonalPolynomials(self):
        assert( self.mu==0 and self.sigma==1 )
        from polynomials.hermite_polynomials import StochasticHermitePolynomials
        return StochasticHermitePolynomials()
        
    def sample(self,  size):
        from numpy.random import normal
        return normal( self.mu,  self.sigma,  size )
    
    import unittest
    class TestNormalDistribution(unittest.TestCase):
        def setUp(self):
            from normal_distribution import NormalDistribution
            self.nd1=NormalDistribution()
            self.nd2=NormalDistribution(2, 3)
        def test_mean(self):
            self.assertEquals( self.nd1.mean(), 0 )
            self.assertEquals( self.nd2.mean(), 2 )
        def test_var(self):
            self.assertEquals( self.nd1.var(), 1 )
            self.assertEquals( self.nd2.var(), 9 )
        def test_pdf(self):
            from numpy import exp, sqrt, pi
            self.assertEquals( self.nd1.pdf(2), exp(-2)/sqrt(2*pi) )
            self.assertEquals( self.nd2.pdf(5), exp(-1.0/2)/sqrt(18*pi) )
        def test_cdf(self):
            from scipy import inf
            self.assertEquals( self.nd1.cdf(0),  0.5)
            self.assertEquals( self.nd1.cdf(inf),  1)
            self.assertEquals( self.nd1.cdf(-inf),  0)
            self.assertEquals( self.nd2.cdf(2),  0.5)
        def test_sample(self):
            s=self.nd1.sample(5)
            self.assertEquals( s.shape,  (5, ) )
            
    if __name__=="__main__":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNormalDistribution)
        unittest.TextTestRunner(verbosity=2).run(suite)
        
