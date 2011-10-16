from abc import ABCMeta, abstractmethod

from numpy import array, ndarray, zeros, ones, diag, arange, sqrt
from numpy import linalg as la

class QuadRule1d(object):
    """Abstract base class for 1d quadrature rules on [0,1]"""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getPointsWeights(self, np):
        return NotImplemented

    def transform(p, interval):
        """Transform points p from [0,1] to interval"""
        return (interval[1]-interval[0])*p + interval[0]


class QuadRuleGauss(QuadRule1d):
    """Gauss quadrature rule"""
    
    def getPointsWeights(self, np):
        g = arange(1,10)/sqrt(4.*arange(1,10)**2-ones(9))
        p,V = la.eig(diag(g,k=1) + diag(g,k=-1))
        w = 2*V[0,:]**2
        # transform to [0,1]
        p = .5 * p + .5
        w = .5 * w
        return p, w
