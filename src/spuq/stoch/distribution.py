from abc import *

class Distribution(object):
    """Base class for probability distributions"""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def pdf( self, x ):
        """Return the probability distribution function at x"""
        return NotImplemented
    
    @abstractmethod
    def cdf( self, x ): 
        """Return the cumulative distribution function at x"""
        return NotImplemented

    @abstractmethod
    def invcdf( self, x ): 
        """Return the cumulative distribution function at x"""
        return NotImplemented
    
    @abstractproperty
    def mean( self ): 
        """The mean of the distribution"""
        return NotImplemented
    
    @abstractproperty
    def var( self ): 
        """The variance of the distribution"""
        return NotImplemented

    @abstractproperty
    def skew( self ): 
        """The skewness of the distribution"""
        return NotImplemented
    
    @abstractproperty
    def kurt( self ): 
        """The kurtosis excess of the distribution"""
        return NotImplemented
    
    @abstractproperty
    def median( self ): 
        """The median of the distribution"""
        return NotImplemented
    
    @abstractproperty
    def orth_polys(self):
        """The median of the distribution"""
        return NotImplemented
    
    @abstractmethod
    def sample(self, size):
        """Sample from the distribution"""
        return NotImplemented


class ShiftedDistribution(Distribution):

    def __init__(self, dist, delta):
        self.dist=dist
        self.delta=delta

    def mean(self):
        return dist.mean()+delta


    def __repr__(self):
        return self.dist.__repr__()+"+"+str(self.delta)

    def __getattr__( self, name ):
        return getattr( self.__subject, name )  

    
    
class ScipyDistribution(Distribution):
    dist = scipy.stats.norm
    pdf = dist.pdf
    invcdf = dist.ppf
    

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


class NormalDistribution(Distribution):
    def __init__(self,  mu=0,  sigma=1 ):
        self.mu=1.0*mu
        self.sigma=1.0*sigma
    
    def pdf(self,  x):
        from numpy import exp, sqrt, pi
        return exp( -0.5*((x-self.mu)/self.sigma)**2)/sqrt(2.0*pi)/self.sigma

    def cdf(self,  x):
        from scipy.special import erfc,  sqrt
        return 0.5*erfc(-sqrt(0.5)*(x-self.mu)/self.sigma)
        
    def invcdf(self, x):
        return NotImplemented

    def mean(self):
        return self.mu
        
    def var(self):
        return self.sigma**2
    
    def skew(self): 
        return 0
        
    def kurtex(self):
        return 0
        
    def median(self): 
        return self.mu
    
    def shift(self, delta):
        return NormalDistribution(self.mu+delta, self.sigma)
        
    def scale(self, scale):
        return NormalDistribution(self.mu, self.sigma*scale)

    def getOrthogonalPolynomials(self):
        assert( self.mu==0 and self.sigma==1 )
        from spuq.polynomials.stochastic_hermite_polynomials import StochasticHermitePolynomials
        return StochasticHermitePolynomials()
        
    def sample(self,  size):
        from numpy.random import normal
        return normal( self.mu, self.sigma, size )

    def __repr__(self):
        return "N["+str(self.mu)+","+str(self.sigma)+"**2]"
