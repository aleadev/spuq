from exceptions import TypeError
from spuq.utils.hashable_ndarray import hashable_ndarray  
from numpy import ndarray

class MultiVector(object):
    '''Accommodates tuples of type (MultiindexSet, Vector).
    
    This class manages a set of Vectors associated to MultiindexSet instances.
    A Vector contains a coefficient vector and the respespective basis.'''
    #map multiindex to Vector (=coefficients + basis)
    def __init__(self, multivec=None):
        if multivec is not None:
            if isinstance(multivec, MultiVector):
                self.mi2vec = multivec.mi2vec
            else:
                raise TypeError
        else:
            self.mi2vec = dict()
    
    def __getitem__(self, mi):
        if isinstance(mi, hashable_ndarray):
            return self.mi2vec[mi]
        else:
            assert(isinstance(mi,ndarray))
            return self.mi2vec[hashable_ndarray(mi)]
    
    def __setitem__(self, mi, vec):
        if not isinstance(mi, hashable_ndarray):
            assert(isinstance(mi, ndarray))
            mi = hashable_ndarray(mi)
        self.mi2vec[mi] = vec
    
    def active_indices(self):
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
        return "MultiVector("+str(self.mi2vec)+")"


import unittest
class TestMultiVector(unittest.TestCase):
    def test_multivector(self):
        from spuq.utils.multiindex_set import MultiindexSet
        from spuq.utils.plot.plotter import Plotter
#        from spuq.fem.fenics import FEniCSMesh, FEniCSBasis, FEniCSVector
        from spuq.fem.fenics.fenics_mesh import FEniCSMesh
        from spuq.fem.fenics.fenics_basis import FEniCSBasis
        from spuq.fem.fenics.fenics_vector import FEniCSVector
        from dolfin import UnitSquare,Expression
        
        # intialise mesh and basis
        mesh = FEniCSMesh(mesh=UnitSquare(4,5))
        basis = FEniCSBasis(mesh)
        # create coefficient vector by interpolation
        F = Expression("sin(x[0]) + cos(x[1])")
        x1 = basis.interpolate(F)       # -> Function
        # create FEMVectors (should be equivalent)
        vec1a = FEniCSVector(x1,basis)
# TODO: the following should also work
#        vec1b = FEniCSVector(x1.vector(),x1.function_space())
        vec1b = FEniCSVector(x1.vector(),basis)
        # store vector with some multi-indices
        MIS = MultiindexSet.createCompleteOrderSet(2,3)
        MV = MultiVector()
        MV[MIS[1]] = vec1a
        MV[MIS[3]] = vec1b
        # uniformly refine basis, prolong coefficient vector and store with some other multi-indices
        basis2, prolongate, restrict = basis.refine()
        assert(isinstance(basis2, FEniCSBasis))
# TODO: whats the problem with the next call???
#        vec2a = basis2.project(vec1a, ptype=(FEniCSBasis.PROJECTION).INTERPOLATION)
        basis2.project(vec1a, ptype=(FEniCSBasis.PROJECTION).INTERPOLATION)
        vec2a = basis2.project(vec1a)
        vec2b = prolongate(vec1b)
        MV[MIS[6]] = vec2a
        MV[MIS[7]] = vec2b
        print MV
        # plot FEMVectors
        Plotter.meshplot(MV[MIS[1]])
        Plotter.vectorplot(MV[MIS[1]])
        Plotter.meshplot(MV[MIS[6]])
        Plotter.vectorplot(MV[MIS[6]])
        Plotter.vectorplot(MV[MIS[10]])
        

if __name__=="__main__":
    unittest.main()
    