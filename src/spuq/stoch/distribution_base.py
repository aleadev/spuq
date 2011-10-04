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
    def __init__(self,dist,delta):
        self.dist=dist
        self.delta=delta
    def mean(self):
        return dist.mean()+delta
    def var(self):
        return dist.var()
    def __repr__(self):
        return self.dist.__repr__()+"+"+str(self.delta)
    
