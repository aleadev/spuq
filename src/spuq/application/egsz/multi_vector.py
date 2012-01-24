import numpy as np

from spuq.linalg.vector import Vector
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import *

__all__ = ["MultiVector"]

class MultiVector(Vector):
    """Accommodates tuples of type (MultiindexSet, Vector/Object).
    
    This class manages a set of Vectors associated to MultiindexSet instances.
    A Vector contains a coefficient vector and the respespective basis.
    Note that the type of the second value of the tuple is not restricted to
    anything specific."""

    @takes(anything)
    def __init__(self):
        self.mi2vec = dict()

    @property
    def basis(self):  # pragma: no cover
        """Implementation of Basis too complicated for MultiVector"""
        raise NotImplementedError

    @property
    def coeffs(self):  # pragma: no cover
        """Not defined for MultiVector"""
        raise NotImplementedError

    def as_array(self):  # pragma: no cover
        """Not defined for MultiVector"""
        raise NotImplementedError

    @takes(anything, Multiindex)
    def __getitem__(self, mi):
        return self.mi2vec[mi]
    
    @takes(anything, Multiindex, Vector)
    def __setitem__(self, mi, val):
        self.mi2vec[mi] = val
    
    def keys(self):
        return self.mi2vec.keys()

    def active_indices(self):
        return self.keys()


    @takes(anything)
    def copy(self):
        mv = MultiVector()
        for mi in self.keys():
            mv[mi] = self[mi].copy()
        return mv

    @takes(anything, MultiindexSet, Vector)
    def set_defaults(self, multiindex_set, init_vector):
        # initialise
        for mi in multiindex_set:
            self[Multiindex(mi)] = init_vector.copy()

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.mi2vec == other.mi2vec)
        


    def __add__(self, other):
#        assert self.active_indices() == other.active_indices()
        newvec = MultiVector(other)
        for mi in self.active_indices():
            try:
                nv = newvec[mi]
                newvec[mi] = self[mi] + nv
            except:
                newvec[mi] = self[mi]
        return newvec
    
    def __mul__(self):
        return NotImplemented
    
    def __sub__(self):
        return NotImplemented

    def __repr__(self):
        return "MultiVector("+str(self.mi2vec.keys())+")"
#        return "MultiVector("+str(self.mi2vec)+")"
