"""Implements a class for generating and storing sets of multiindices"""

import numpy as np
import scipy as sp

__all__ = ["MultiindexSet"]


class MultiindexSet(object):
    def __init__(self, arr):
        self.arr = arr

    @property
    def m(self):
        return self.arr.shape[1]

    @property
    def p(self):
        return max(self.arr.sum(1))

    @property
    def count(self):
        return len(self)
        
    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, i):
        return self.arr[i]

    def __repr__(self):
        return "<MISet m={0}, p={1}, arr={2}>".format(self.m, self.p, self.arr)

    def power(self, vec):
        assert vec.size == self.m
        res = vec[0] ** self.arr[:, 0]
        for i in xrange(1, vec.size):
            res = res * vec[i] ** self.arr[:, i]
        return res

    def factorial(self):
        return sp.factorial(self.arr).prod(1)

    @staticmethod
    def _makeGenerator(m, func):
        p = 0
        k = 0
        while True:
            mis = func(m, p)
            for i in xrange(k, len(mis)):
                yield mis[i]
            k = len(mis)
            p = p + 1

    @classmethod
    def createCompleteOrderSet(cls, m, p=None, reversed=False):
        if p is None:
            return cls._makeGenerator(m, cls.createCompleteOrderSet)

        def create(m, p):
            if m == 0:
                return np.zeros((1, 0), np.int8)
            else:
                I = np.zeros((0, m), np.int8)
                for q in xrange(0, p + 1):
                    J = create(m - 1, q)
                    Jn = q - J.sum(1).reshape((J.shape[0], 1))
                    I = np.vstack((I, np.hstack((J, Jn))))
                return I
        arr = create(m, p)
        if reversed:
            arr = arr[:, ::-1]
        return cls(arr)

# createFullTensorSet(m, p)
# createAnisoFullTensorSet(p)
# createLimitedCompleteOrderSet(m, l, p)
