from exceptions import TypeError, AttributeError

from spuq.fem.fem_basis import FEMBasis
from spuq.linalg.operator import MatrixOperator
from spuq.utils.type_check import takes, optional, anything
from spuq.utils.enum import Enum

from dolfin import (FunctionSpace, FunctionSpaceBase, Function,
                TestFunction, TrialFunction, Mesh, assemble, dx)
from dolfin.fem.interpolation import interpolate
from dolfin.fem.projection import project


class FEniCSBasis(FEMBasis):
    """wrapper for FEniCS/dolfin FunctionSpace"""

    PROJECTION = Enum('INTERPOLATION', 'L2PROJECTION')

    @takes(anything, optional(Mesh), optional(str), optional(FunctionSpaceBase))
    def __init__(self, mesh=None, family='CG', degree=1, functionspace=None):
        """initialise discrete basis on mesh"""
        if functionspace:
            assert mesh == None and functionspace
            self._functionspace = functionspace
            UFL = functionspace.ufl_element()
            family = UFL.family()
            degree = UFL.degree()
            mesh = functionspace.mesh()
        else:
            self._functionspace = None
        assert mesh != None
        self.family = family
        self.degree = degree
        self._mesh = mesh
        if not self._functionspace:
            self._functionspace = FunctionSpace(mesh, family, degree)
        self._dim = self._functionspace.dim()

    def refine(self, cells=None):
        """refines mesh of basis uniformly or wrt cells, return new (FEniCSBasis,prolongate,restrict)"""
        import spuq.fem.fenics.fenics_vector   # NOTE: from ... import FEniCSVector does not work (cyclic dependencies require module imports)
        newmesh = self.mesh.refine(cells)
        newFB = FEniCSBasis(newmesh, self.family, self.degree)
        prolongate = lambda vec: spuq.fem.fenics.fenics_vector.FEniCSVector(function=project(vec.F, newFB.functionspace))
        restrict = lambda vec: spuq.fem.fenics.fenics_vector.FEniCSVector(function=project(vec.F, self.functionspace))
        return (newFB, prolongate, restrict)

    def project(self, vec, vecbasis=None, ptype=PROJECTION.INTERPOLATION):
        """Project vector vec to this basis.
        
            vec can either be a FEniCSVector (in which case vecbasis has to be None) or an array
            (in which case vecbasis has to be provided as dolfin FunctionSpace)."""
        import spuq.fem.fenics.fenics_vector   # NOTE: from ... import FEniCSVector does not work (cyclic dependencies require module imports)
        if ptype == self.PROJECTION.INTERPOLATION:
            T = interpolate
        elif ptype == self.PROJECTION.L2PROJECTION:
            T = project
        else:
            raise AttributeError

        if isinstance(vec, spuq.fem.fenics.fenics_vector.FEniCSVector):
            assert(vecbasis is None)
            F = T(vec.function, self.functionspace)
            newvec = spuq.fem.fenics.fenics_vector.FEniCSVector(function=F)
#            newvec = spuq.fem.fenics.fenics_vector.FEniCSVector(F.vector(), F.function_space())
            return newvec
        else:
            F = T(Function(vecbasis, vec), self.functionspace)
            return (F.vector().array(), F.function_space())

    def interpolate(self, F):
        """interpolate FEniCS Expression/Function on basis"""
        return interpolate(F, self.functionspace)

    @property
    def functionspace(self):
        return self._functionspace

    @property
    def mesh(self):
        return self._mesh

    @property
    def dim(self):
        return self._dim

    @property
    def gramian(self):
        """Returns the Gramian as a LinearOperator"""
        u = TrialFunction(self._functionspace)
        v = TestFunction(self._functionspace)
        a = (u * v) * dx
        A = assemble(a)
        return MatrixOperator(A.array())
