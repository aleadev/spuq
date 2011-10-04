from abc import *

import numpy as np

from spuq.linalg.basis import FunctionBasis


class StochasticBasis(FunctionBasis):
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, n):
        return NotImplemented

    @property
    def dim(self):
        return NotImplemented

    def get_gramian(self):
        return NotImplemented


class MultiindexBasis(StochasticBasis):
    def __init__(self, I,  rvs):
        assert(I.m == len(rvs))
        self.I = I
        self.rvs = rvs

    def sample(self, n):
        from numpy import ones
        S = np.ones((self.I.count, n))
        for i, rv in enumerate(self.rvs):
            theta = rv.sample(n)
            Phi = rv.getOrthogonalPolynomials()
            Q = zeros((self.I.p + 1, n))
            for q in xrange(self.I.p + 1):
                Q[q, :] = Phi.eval(q, theta)
            S = S * Q[self.I.arr[:, i], :]
        return S
    pass


class PCBasis(StochasticBasis):
    # we already have that (normalised or not?)
    pass


class GPCBasis(StochasticBasis):
    def __init__(self, dist):
        self._dist = dist

    @property
    def dist(self):
        return _dist

    def sample(self, n):
        S = np.ones((self.I.count, n))
        for i, rv in enumerate(self.rvs):
            theta = dist.sample(n)
            Phi = dist.getOrthogonalPolynomials()
            Q = zeros((self.I.p + 1, n))
            for q in xrange(self.I.p + 1):
                Q[q, :] = Phi.eval(q, theta)
            S = S * Q[self.I.arr[:, i], :]
        return S
