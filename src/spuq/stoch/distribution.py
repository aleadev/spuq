from abc import *

import numpy as np
import scipy
import scipy.stats

import spuq.polyquad.polynomials as polys


class Distribution(object):
    """Base class for probability distributions"""
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


class ShiftedDistribution(Distribution):

    def __init__(self, dist, delta):
        self.dist = dist
        self.delta = delta

    def mean(self):
        return dist.mean() + delta

    def __repr__(self):
        return self.dist.__repr__() + " + " + str(self.delta)

    def __getattr__(self, name):
        return getattr(self.__subject, name)


class ScipyDistribution(Distribution):
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

    def median(self):
        return self._dist.median()

    def mean(self):
        return self._dist.stats(moments="m")

    def var(self):
        return self._dist.stats(moments="v")

    def skew(self):
        return self._dist.stats(moments="s")

    def kurtosis(self):
        return self._dist.stats(moments="k")

    def sample(self,  size):
        return self._dist.rvs(size=size)


class NormalDistribution(ScipyDistribution):

    def __init__(self, mu=0, sigma=1):
        super(NormalDistribution, self).__init__(scipy.stats.norm(mu, sigma))
        self.mu = float(mu)
        self.sigma = float(sigma)

    def shift(self, delta):
        return NormalDistribution(self.mu + delta, self.sigma)

    def scale(self, scale):
        return NormalDistribution(self.mu, self.sigma * scale)

    @property
    def orth_polys(self):
        assert(self.mu == 0 and self.sigma == 1)
        return StochasticHermitePolynomials()

    def __repr__(self):
        return "N[" + str(self.mu) + ", " + str(self.sigma) + " ** 2]"


class UniformDistribution(ScipyDistribution):

    def __init__(self, a=-1, b=1):
        self.a = float(min(a, b))
        self.b = float(max(a, b))
        loc = a
        scale = (self.b - self.a)
        super(UniformDistribution, self).__init__(scipy.stats.uniform(loc,
                                                                      scale))

    def shift(self, delta):
        return UniformDistribution(self.a + delta, self.b + delta)

    def scale(self, scale):
        m = 0.5 * (self.a + self.b)
        d = scale * 0.5 * (self.b - self.a)
        return UniformDistribution(m - d, m + d)

    @property
    def orth_polys(self):
        assert(self.a == -1.0 and self.b == 1.0)
        return LegendrePolynomials()

    def __repr__(self):
        return "U[" + str(self.a) + ", " + str(self.b) + "]"
