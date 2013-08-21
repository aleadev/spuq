#from abc import ABCMeta, abstractmethod, abstractproperty
from hashlib import sha1
import numpy as np
import scipy as sp

from spuq.utils import strclass
from spuq.utils.decorators import total_ordering
from spuq.utils.type_check import takes, anything, list_of, optional

__all__ = ["Multiindex"]

@total_ordering
class Multiindex(object):
    @takes(anything, optional(np.ndarray, list_of(int)))
    def __init__(self, arr=None):
        # create numpy array or make a copy if it already is
        if arr is None or len(arr) == 0:
            arr = [0]
        arr = np.array(arr)
        if not issubclass(arr.dtype.type, int):
            raise TypeError
        self._arr = arr
        self._normalise()
        self.__hash = None

    def _normalise(self):
        nz = np.nonzero(self._arr)[0]
        if len(nz):
            l = np.max(nz)
            self._arr = self._arr[:l + 1].copy()
        else:
            self._arr = np.resize(self._arr, 0)

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.__hash__() == other.__hash__() and
                np.array_equal(self._arr, other._arr))

    def __ne__(self, other):
        return not self.__eq__(other)

    def cmp_by_order(self, other):
        assert type(self) is type(other)
        sa = self._arr
        oa = other._arr
        cmpval = cmp(sum(sa), sum(oa))
        if not cmpval:
            l = min(len(sa), len(oa))
            for i in xrange(l):
                cmpval = cmp(sa[i], oa[i])
                if cmpval:
                    break
        return cmpval

    def cmp_by_index(self, other):
        assert type(self) is type(other)
        sa = self._arr
        oa = other._arr
        l = len(sa) - len(oa)
        if l <= 0:
            return np.append(sa, np.zeros(-l)) == oa
        else:
            return np.append(oa, np.zeros(l)) == sa

    def __le__(self, other):
        cmpval = self.cmp_by_order(other)
        return cmpval <= 0

    def __gt__(self, other):
        return not self <= other

    def __hash__(self):
        if self.__hash is None:
            self.__hash = int(sha1(self._arr).hexdigest(), 16)
        return self.__hash

    def __len__(self):
        return len(self._arr)

    def __repr__(self):
        return "<%s inds=%s>" % \
               (strclass(self.__class__), self._arr)

    def __getitem__(self, i):
        if 0 <= i < len(self._arr):
            return self._arr[i]
        else:
            return 0

    @property
    def order(self):
        return sum(self._arr)

    @property
    def as_array(self):
        return self._arr

    @property
    def supp(self):
        return np.flatnonzero(self._arr)

    def inc(self, pos, by=1):
        assert pos >= 0
        value = self._arr[pos] if pos < len(self._arr) else 0
        newval = value + by
        if newval < 0:
            return None
#         if False:    # TODO: debugging work-around, clean up
#             arr = self._arr.copy()
#             if pos >= len(arr):
#                 if True or len(arr):
#                     print arr, type(arr), repr(arr)
#                     arr = arr.copy()
#                     arr.resize(pos + 1)
#                 else:
#                     arr = np.array([0] * (pos + 1))
#         if True:
        l = len(self._arr)
        arr = np.array([0] * max(l, (pos + 1)))
        if l:
            arr[:l] = self._arr[:l]
        arr[pos] = newval
        return self.__class__(arr)

    def dec(self, pos, by=1):
        return self.inc(pos, -by)

    def factorial(self):
        return sp.misc.factorial(self._arr)

    @staticmethod
    def createCompleteOrderSet(m, p):
        from multiindex_set import MultiindexSet
        class MultiindexSetEx(MultiindexSet):
            def __getitem__(self, i):
                return Multiindex(MultiindexSet.__getitem__(self, i))
        return MultiindexSetEx.createCompleteOrderSet(m, p)
