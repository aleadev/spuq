from weakref import WeakValueDictionary

from spuq.linalg.vector import Scalar, Vector
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import takes, anything, optional
from spuq.utils import strclass

__all__ = ["MultiVector", "MultiVectorWithProjection"]

class MultiVector(Vector):
    """Accommodates tuples of type (MultiindexSet, Vector/Object).
    
    This class manages a set of Vectors associated to MultiindexSet instances.
    A Vector contains a coefficient vector and the respespective basis.
    Note that the type of the second value of the tuple is not restricted to
    anything specific."""

    @takes(anything, optional(callable))
    def __init__(self, on_modify=lambda : None):
        self.mi2vec = dict()
        self.on_modify = on_modify

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
        self.on_modify()
        self.mi2vec[mi] = val

    def keys(self):
        return self.mi2vec.keys()

    def active_indices(self):
        return sorted(self.keys())

    def copy(self):
        mv = self.__class__()
        for mi in self.keys():
            mv[mi] = self[mi].copy()
        return mv

    @takes(anything, MultiindexSet, Vector)
    def set_defaults(self, multiindex_set, init_vector):
        self.on_modify()
        for mi in multiindex_set:
            self[Multiindex(mi)] = init_vector.copy()

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.mi2vec == other.mi2vec)

    def __neg__(self):
        new = self.copy()
        for mi in self.active_indices():
            new[mi] = -self[mi]
        return new

    def __iadd__(self, other):
        assert self.active_indices() == other.active_indices()
        self.on_modify()
        for mi in self.active_indices():
            self[mi] += other[mi]
        return self

    def __isub__(self, other):
        assert self.active_indices() == other.active_indices()
        self.on_modify()
        for mi in self.active_indices():
            self[mi] -= other[mi]
        return self

    def __imul__(self, other):
        assert isinstance(other, Scalar)
        self.on_modify()
        for mi in self.keys():
            self[mi] *= other
        return self

    def __repr__(self):
        return "<%s keys=%s>" % (strclass(self.__class__), self.mi2vec.keys())




class MultiVectorWithProjection(MultiVector):
    @takes(anything, optional(callable))
    def __init__(self, project=None):
        MultiVector.__init__(self, self.clear_cache)
        if not project:
            project = MultiVectorWithProjection.default_project
        self.project = project
        self._proj_cache = WeakValueDictionary()
        self._back_cache = WeakValueDictionary()

    @staticmethod
    def default_project(vec_src, vec_dest):
        """Project the source vector onto the basis of the destination vector."""
        assert hasattr(vec_dest.basis, "project_onto")
        return vec_dest.basis.project_onto(vec_src)

    def copy(self):
        mv = MultiVector.copy(self)
        mv.project = self.project
        return mv

    def __eq__(self, other):
        return (MultiVector.__eq__(self, other) and
                self.project == other.project)

    def clear_cache(self):
        self._back_cache.clear()
        self._proj_cache.clear()

    @takes(anything, Multiindex, Multiindex)
    def get_projection(self, mu_src, mu_dest):
        """Return projection of vector in multivector"""
        args = (mu_src, mu_dest, self.project)
        vec = self._proj_cache.get(args)
        if not vec:
            vec = self.project(self[mu_src], self[mu_dest])
            self._proj_cache[args] = vec
        return vec


    @takes(anything, Multiindex, Multiindex)
    def get_back_projection(self, mu_src, mu_dest):
        """Return back projection of vector in multivector"""
        args = (mu_src, mu_dest, self.project)
        vec = self._back_cache.get(args)
        if not vec:
            vec_prj = self.get_projection(mu_src, mu_dest)
            vec = self.project(vec_prj, self[mu_src])
            self._back_cache[args] = vec
        return vec
