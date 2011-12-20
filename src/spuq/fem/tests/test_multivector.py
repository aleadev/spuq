from spuq.fem.multi_vector import MultiVector

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
        vec1a = FEniCSVector(function=x1)
# TODO: the following should also work
#        vec1b = FEniCSVector(x1.vector(),x1.function_space())
        vec1b = FEniCSVector(x1.vector(),basis)
        # store vector with some multi-indices
        MIS = MultiindexSet.createCompleteOrderSet(2,3)
        MV = MultiVector()
        MV[MIS[1]] = vec1a
        MO1 = vec1a.basis.get_gramian()
        print 'Gram matrix 1: ',MO1
        print 'active indices: ', MV.active_indices()
        vec1a = MV[MIS[1]]
        MV[MIS[3]] = vec1b
        # uniformly refine basis, prolong coefficient vector and store with some other multi-indices
        basis2, prolongate, restrict = basis.refine()
        print basis.mesh, basis2.mesh
        print basis.mesh.mesh, basis2.mesh.mesh
        assert(isinstance(basis2, FEniCSBasis))
        vec2a = basis2.project(vec1a, ptype=(FEniCSBasis.PROJECTION).INTERPOLATION)
#        vec2a = basis2.project(vec1a)
        vec2b = prolongate(vec1b)
        MO2 = basis2.get_gramian()
        print 'Gram matrix 2: ',MO2
        MV[MIS[6]] = vec2a
        MV[MIS[7]] = vec2b
        print MV
        # plot FEMVectors
        Plotter.meshplot(MV[MIS[1]].basis.mesh.mesh)
        Plotter.vectorplot(MV[MIS[1]])
        print 'MIS[1]: ',type(MV[MIS[1]]),type(MV[MIS[1]].basis),type(MV[MIS[1]].basis.mesh)
        print 'MIS[3]: ',type(MV[MIS[3]]),type(MV[MIS[3]].basis),type(MV[MIS[3]].basis.mesh)
        print 'MIS[6]: ',type(MV[MIS[6]]),type(MV[MIS[6]].basis),type(MV[MIS[6]].basis.mesh)
        print 'MIS[7]: ',type(MV[MIS[7]]),type(MV[MIS[7]].basis),type(MV[MIS[7]].basis.mesh)
        Plotter.meshplot(MV[MIS[6]].basis.mesh)
        Plotter.vectorplot(MV[MIS[6]])
        Plotter.vectorplot(MV[MIS[7]])
        
test_main()
