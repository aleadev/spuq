from abc import *

import numpy as np
import scipy
import scipy.stats

import spuq.polyquad.polynomials as polys


class RandomVariable(object):
    """Base class for random variables"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def pdf(self, x):
        """Return the probability distribution function at x"""
        return NotImplemented

    @abstractmethod
    def cdf(self, x):
        """Return the cumulative distribution function at x"""
        return NotImplemented

    @abstractmethod
    def invcdf(self, x):
        """Return the cumulative distribution function at x"""
        return NotImplemented

    @abstractproperty
    def mean(self):
        """The mean of the distribution"""
        return NotImplemented

    @abstractproperty
    def var(self):
        """The variance of the distribution"""
        return NotImplemented

    @abstractproperty
    def skew(self):
        """The skewness of the distribution"""
        return NotImplemented

    @abstractproperty
    def kurtosis(self):
        """The kurtosis excess of the distribution"""
        return NotImplemented

    @abstractproperty
    def median(self):
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


class ShiftedRandomVariable(RandomVariable):
    """Proxy class that shifts a given random variable by some amount.
    
    Do not use yet as not all methods are appropriately
    overridden. Especially the orthogonal polynomials need some work.
    """
    def __init__(self, dist, delta):
        self.dist = dist
        self.delta = delta

    @property
    def mean(self):
        return dist.mean() + delta

    @abstractmethod
    def pdf(self, x):
        return dist.pdf(x - dist)

    def __repr__(self):
        return self.dist.__repr__() + " + " + str(self.delta)

    def __getattr__(self, name):
        return getattr(self.__subject, name)


class ScipyRandomVariable(RandomVariable):
    """Utility class for probability distributions that wrap a SciPy
    distribution"""

    def __init__(self, dist):
        self._dist = dist

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def invcdf(self, x):
        return self._dist.ppf(x)

    @property
    def median(self):
        return self._dist.ppf(0.5)

    @property
    def mean(self):
        return self._dist.stats(moments="m")

    @property
    def var(self):
        return self._dist.stats(moments="v")

    @property
    def skew(self):
        return self._dist.stats(moments="s")

    @property
    def kurtosis(self):
        return self._dist.stats(moments="k")

    def sample(self, size):
        return self._dist.rvs(size=size)


class NormalRV(ScipyRandomVariable):

    def __init__(self, mu=0, sigma=1):
        super(NormalRV, self).__init__(scipy.stats.norm(mu, sigma))
        self.mu = float(mu)
        self.sigma = float(sigma)

    def shift(self, delta):
        return NormalRV(self.mu + delta, self.sigma)

    def scale(self, scale):
        return NormalRV(self.mu, self.sigma * scale)

    @property
    def orth_polys(self):
        return polys.StochasticHermitePolynomials(self.mu, 
                                                  self.sigma, 
                                                  normalised=False)

    def __repr__(self):
        return "N[" + str(self.mu) + ", " + str(self.sigma) + " ** 2]"


class UniformRV(ScipyRandomVariable):

    def __init__(self, a=-1, b=1):
        self.a = float(min(a, b))
        self.b = float(max(a, b))
        loc = a
        scale = (self.b - self.a)
        super(UniformRV, self).__init__(scipy.stats.uniform(loc,
                                                            scale))

    def shift(self, delta):
        return UniformRV(self.a + delta, self.b + delta)

    def scale(self, scale):
        m = 0.5 * (self.a + self.b)
        d = scale * 0.5 * (self.b - self.a)
        return UniformRV(m - d, m + d)

    @property
    def orth_polys(self):
        return polys.LegendrePolynomials(self.a, self.b, normalised=False)

    def __repr__(self):
        return "U[" + str(self.a) + ", " + str(self.b) + "]"
