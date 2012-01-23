#from abc import ABCMeta, abstractmethod, abstractproperty
from hashlib import sha1

import numpy as np

from spuq.utils.type_check import takes, anything, list_of

__all__ = ["Multiindex"]


class Multiindex(object):
    @takes(anything, (np.ndarray, list_of(int)))
    def __init__(self, arr):
        # create numpy array or make a copy if it already is
        arr = np.array(arr)
        if not issubclass(arr.dtype.type, int):
            raise TypeError
        self._arr = arr
        self._normalise()
        self.__hash = None

    def _normalise(self):
        l = np.max(np.nonzero(self._arr))
        self._arr = self._arr[:l + 1]

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.__hash == other.__hash and
                np.array_equal(self._arr, other._arr))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self.__hash is None:
            self.__hash = int(sha1(self._arr).hexdigest(), 16)
        return self.__hash

    def __len__(self):
        return len(self._arr)

    def __repr__(self):
        return "<mi %s>" % self._arr

    @property
    def order(self):
        return sum(self._arr)

    @property
    def as_array(self):
        return self._arr

    def inc(self, pos, by=1):
        assert pos >= 0
        value = self._arr[pos] if pos < len(self._arr) else 0
        newval = value + by
        if newval < 0:
            return None
        arr = self._arr.copy()
        if pos >= len(arr):
            arr.resize(pos + 1)
        arr[pos] = newval
        return self.__class__(arr)

    def dec(self, pos, by=1):
        return self.inc(pos, -by)



# class MultiindexSet(object):
#     pass


# class FixedLengthSet(MultiindexSet):
#     pass


# class VariableLengthSet(MultiindexSet):
#     pass
