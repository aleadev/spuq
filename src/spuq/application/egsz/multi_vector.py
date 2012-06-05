from spuq.linalg.vector import Scalar, Vector, inner
from spuq.linalg.basis import Basis
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import takes, anything, optional
from spuq.utils import strclass

import os
import pickle
from collections import defaultdict

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

    def pickle(self, outdir):
        """pickle object"""
#        outdir = os.path.abspath(os.path.abspath(outdir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        Lambda = self.active_indices()
        # save active multiindex set
        with open(os.path.join(outdir, 'MI.pkl'), 'wb') as f:
            pickle.dump(Lambda, f)
        # iteratively pickle multiindex data/mesh
        for mu in Lambda:
            self[mu].pickle(outdir, str(mu))
    
    @classmethod
    def from_pickle(cls, indir, veccls):
        """unpickle object"""
        with open(os.path.join(indir, 'MI.pkl'), "rb") as f:
            Lambda = pickle.load(f)
            print "Lambda:", Lambda
            w = cls()
            for mu in Lambda:
                w[mu] = veccls.from_pickle(indir, str(mu))
            return w
    

class MultiVectorWithProjection(MultiVector):
    @takes(anything, optional(callable), optional(bool))
    def __init__(self, project=None, cache_active=False):
        MultiVector.__init__(self, self.clear_cache)
        if not project:
            project = MultiVectorWithProjection.default_project
        self.project = project
        self._proj_cache = defaultdict(dict)
        self._back_cache = {}
        self._cache_active = cache_active

    @staticmethod
    def default_project(vec_src, dest):
        """Project the source vector onto the basis of the destination vector."""
        if not isinstance(dest, Basis):
            basis = dest.basis
        else:
            basis = dest

        assert hasattr(basis, "project_onto")
        return basis.project_onto(vec_src)

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

    @takes(anything, Multiindex, Multiindex, anything)
    def get_projection(self, mu_src, mu_dest, degree=None):
        """Return projection of vector in multivector"""
        if degree is None:
            degree = self[mu_dest].degree
        # w/o caching
        if not self._cache_active:
            if self[mu_dest].degree == degree:
                return self.project(self[mu_src], self[mu_dest])
            else:
                V = self[mu_dest].basis.copy(degree)
                return V.project_onto(self[mu_src])
        
        # with caching
        args = (mu_src, mu_dest, self.project)
        vec = self._proj_cache[degree].get(args)            # check cache
        #        print "P MultiVector get_projection", mu_src, mu_dest
        if not vec:
        #            print "P ADDING TO CACHE: new projection required..."
            if self[mu_dest].degree == degree:
                vec = self.project(self[mu_src], self[mu_dest])
            else:
                try:
                    V = self._proj_cache[degree]["V"]       # try to retrieve basis
                except:
                    V = self[mu_dest].basis.copy(degree)    # create and store basis if necessary
                    self._proj_cache[degree]["V"] = V
                vec = V.project_onto(self[mu_src])
            self._proj_cache[degree][args] = vec
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
        # w/o caching
        if not self._cache_active:
            vec_prj = self.get_projection(mu_src, mu_dest)
            return self.project(vec_prj, self[mu_src])

        # with caching
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

    @takes(anything, Multiindex, Multiindex, int, bool)
    def get_projection_error_function(self, mu_src, mu_dest, reference_degree, refine_mesh=0):
        """Construct projection error function by projecting mu_src vector to mu_dest space of dest_degree.
        From this, the projection of mu_src onto the mu_dest space, then to the mu_dest space of dest_degree is subtracted.
        If refine_mesh > 0, the destination mesh is refined uniformly n times."""
        # TODO: If refine_mesh is True, the destination space of mu_dest is ensured to include the space of mu_src by mesh refinement
        # TODO: proper description
        # TODO: separation of fenics specific code
        from dolfin import refine, FunctionSpace
        from spuq.fem.fenics.fenics_basis import FEniCSBasis
        import numpy as np
        if not refine_mesh:
            w_reference = self.get_projection(mu_src, mu_dest, reference_degree)
            w_dest = self.get_projection(mu_src, mu_dest)
            w_dest = w_reference.basis.project_onto(w_dest)
            sum_up = lambda vals: vals
        else:
            # uniformly refine destination mesh
            # NOTE: the cell_marker based refinement used in FEniCSBasis is a bisection of elements
            # while refine(mesh) carries out a red-refinement of all cells (split into 4)
            basis_src = self[mu_src].basis 
            basis_dest = self[mu_dest].basis
            mesh_reference = basis_dest.mesh
            for _ in range(refine_mesh):
                mesh_reference = refine(mesh_reference)
            fs_reference = FunctionSpace(mesh_reference, basis_dest._fefs.ufl_element().family(), reference_degree)
            basis_reference = FEniCSBasis(fs_reference, basis_dest._ptype)
            # project both vectors to reference space
            w_reference = basis_reference.project_onto(self[mu_src])
            w_dest = self.get_projection(mu_src, mu_dest)
            w_dest = basis_reference.project_onto(w_dest)
            sum_up = lambda vals: np.array([sum(vals[i * 4:(i + 1) * 4]) for i in range(len(vals) / 4 ** refine_mesh)])
        return w_dest - w_reference, sum_up
                    
#            # ensure that source space is included in reference space by mesh refinement
#            basis_src = self[mu_src].basis 
#            minh = basis_src.minh
#            basis_dest = self[mu_dest].basis.copy(dest_degree)
#            basis_reference = basis_dest.refine_maxh(minh)
#            w_reference = basis_reference.project_onto(self[mu_src])
#            w_dest = self.get_projection(mu_src, mu_dest)
#            w_dest = basis_reference.project_onto(w_dest)
#            sum_up = lambda vals: [sum(t[i * 4:(i + 1) * 4]) for i in range(vals(t) / 4 ** refine_mesh)]
#            return w_dest - w_reference, sum_up

    @property
    def cache_active(self):
        return self._cache_active
    
    @cache_active.setter
    def cache_active(self, val):
        self._cache_active = val
        if not val:
            self.clear_cache()
