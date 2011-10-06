from abc import *

import numpy as np

from spuq.utils.decorators import copydocs
from spuq.linalg.basis import FunctionBasis

@copydocs
class StochasticBasis(FunctionBasis):
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, n):
        """Sample from the underlying distribution(s) and evaluate at
        basis functions"""
        return NotImplemented


class MultiindexBasis(StochasticBasis):
    def __init__(self, I,  bases):
        assert(I.m == len(bases))
        self.I = I
        self.bases = bases
        # assert bases are instances of StochasticBasis
        # assert dim of bases larger or equal to max in I
        for k, B in enumerate(bases):
            assert isinstance(B, StochaticBasis)
            assert I.arr[:, i].max() < B.dim

    def sample(self, n):
        S = np.ones((self.I.count, n))
        for i, rv in enumerate(self.rvs):
            theta = rv.sample(n)
            Phi = rv.getOrthogonalPolynomials()
            Q = zeros((self.I.p + 1, n))
            for q in xrange(self.I.p + 1):
                Q[q, :] = Phi.eval(q, theta)
            S = S * Q[self.I.arr[:, i], :]
        return S
    
    @property
    def gramian(self):
        return NotImplemented

    @property
    def dim(self):
        return NotImplemented


class GPCBasis(StochasticBasis):
    def __init__(self, rv, p):
        self._rv = rv
        self._p = p

    @property
    def rv(self):
        return self._rv

    @property
    def degree(self):
        return self._p

    @property
    def dim(self):
        return self._p+1

    @property
    def gramian(self):
        # return DiagonalMatrix( [rv.orthpoly.norm(q) for q in xrange(p+1)])
        return NotImplemented

    def sample(self, n):
        rv = self._rv
        p = self._p
        theta = rv.sample(n)
        Phi = rv.orth_polys
        Q = np.zeros((p + 1, n))
        for q in xrange(p + 1):
            Q[q, :] = Phi.eval(q, theta)
        return Q
