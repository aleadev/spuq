from exceptions import TypeError
from spuq.utils.hashable_ndarray import hashable_ndarray  
from spuq.utils.type_check import *
from numpy import ndarray

class MultiVector(object):
    """Accommodates tuples of type (MultiindexSet, Vector/Object).
    
    This class manages a set of Vectors associated to MultiindexSet instances.
    A Vector contains a coefficient vector and the respespective basis.
    Note that the type of the second value of the tuple is not restricted to
    anything specific."""

    def __init__(self, multiindex=None, initvector=None):
        self.mi2vec = dict()
        # initialise
        if multiindex:
            assert initvector
            for mi in multiindex:
                self[mi] = initvector

    @staticmethod
    @takes("MultiVector")
    def create(multivec):
        MV = MultiVector()
        # setup
        MV.mi2vec = multivec.mi2vec

    def __getitem__(self, mi):
        if isinstance(mi, hashable_ndarray):
            return self.mi2vec[mi]
        else:
            assert(isinstance(mi, ndarray))
            return self.mi2vec[hashable_ndarray(mi)]
    
    def __setitem__(self, mi, val):
        if not isinstance(mi, hashable_ndarray):
            assert(isinstance(mi, ndarray))
            mi = hashable_ndarray(mi)
        self.mi2vec[mi] = val
    
    def active_indices(self):
        return self.keys()

    def keys(self):
        return self.mi2vec.keys()

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
