from spuq.linalg.vector import Scalar, Vector, inner
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
    def __init__(self, on_modify=lambda: None):
        self.mi2vec = dict()
        self.on_modify = on_modify

    @property
    def basis(self):  # pragma: no cover
        """Implementation of Basis too complicated for MultiVector"""
        raise NotImplementedError

    def flatten(self):
        """Not yet defined for MultiVector"""
        raise NotImplementedError

    @property
    def max_order(self):
        """Returns the maximum order of the multiindices."""
        return max(len(mu) for mu in self.keys())

    @takes(anything, Multiindex)
    def __getitem__(self, mi):
        return self.mi2vec[mi]

    @takes(anything, Multiindex, Vector)
    def __setitem__(self, mi, val):
        self.on_modify()
        self.mi2vec[mi] = val

    def keys(self):
        return self.mi2vec.keys()

    def iteritems(self):
        return self.mi2vec.iteritems()

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

    def __inner__(self, other):
        assert isinstance(other, MultiVector)
        s = 0.0
        for mi in self.keys():
            s += inner(self[mi], other[mi])
        return s

    def __repr__(self):
        return "<%s keys=%s>" % (strclass(self.__class__), self.mi2vec.keys())


class MultiVectorWithProjection(MultiVector):
    @takes(anything, optional(callable))
    def __init__(self, project=None):
        MultiVector.__init__(self, self.clear_cache)
        if not project:
            project = MultiVectorWithProjection.default_project
        self.project = project
        self._proj_cache = {}               # NOTE: WeakValueDictionary is not appropriate for this type of cache
        self._back_cache = {}

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
        #        print "P MultiVector get_projection", mu_src, mu_dest
        if not vec:
        #            print "P ADDING TO CACHE: new projection required..."
            vec = self.project(self[mu_src], self[mu_dest])
            self._proj_cache[args] = vec
            #            print "P proj_cache size", len(self._proj_cache)
        #            print "P with keys", self._proj_cache.keys()
        #        else:
        #            print "P CACHED!"
        #        print "P dim mu_src =", self[mu_src].coeffs.size()
        #        print "P dim mu_dest =", self[mu_dest].coeffs.size()
        #        print "P dim vec =", vec.coeffs.size()
        return vec

    @takes(anything, Multiindex, Multiindex)
    def get_back_projection(self, mu_src, mu_dest):
        """Return back projection of vector in multivector"""
        args = (mu_src, mu_dest, self.project)
        vec = self._back_cache.get(args)
        #        print "BP MultiVector get_back_projection", mu_src, mu_dest
        if not vec:
        #            print "BP ADDING TO CACHE: new back_projection required..."
            vec_prj = self.get_projection(mu_src, mu_dest)
            vec = self.project(vec_prj, self[mu_src])
            self._back_cache[args] = vec
            #            print "BP back_cache size", len(self._back_cache)
        #            print "BP with keys", self._back_cache.keys()
        #        else:
        #            print "BP CACHED!"
        #        print "BP dim mu_src =", self[mu_src].coeffs.size()
        #        print "BP dim mu_dest =", self[mu_dest].coeffs.size()
        #        print "BP dim vec =", vec.coeffs.size()
        return vec
