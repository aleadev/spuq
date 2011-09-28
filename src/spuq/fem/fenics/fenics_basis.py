from exceptions import TypeError, AttributeError
from spuq.utils.enum import Enum
#from spuq.fem import *
#from spuq.fem.fenics import *
from spuq.fem.fem_basis import FEMBasis
#from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.fem.fenics.fenics_mesh import FEniCSMesh
from spuq.linalg.operator import MatrixOperator
from dolfin import FunctionSpace, FunctionSpaceBase, Function, \
                TestFunction, TrialFunction, Mesh, assemble, dx
from dolfin.fem.interpolation import interpolate
from dolfin.fem.projection import project

class FEniCSBasis(FEMBasis):
    '''wrapper for FEniCS/dolfin FunctionSpace'''
    
    PROJECTION = Enum('INTERPOLATION','L2PROJECTION')

    def __init__(self, mesh=None, family='CG', degree=1, functionspace=None):
        '''initialise discrete basis on mesh'''
        if functionspace is not None:
            assert(mesh is None)
            assert(isinstance(functionspace, FunctionSpaceBase))
            UFL = functionspace.ufl_element()
            family = UFL.family()
            degree = UFL.degree()
            mesh = functionspace.mesh()
            
        assert(mesh is not None)
        self.family = family
        self.degree = degree
        if isinstance(mesh, FEniCSMesh):
            self.__mesh = mesh
            mesh = mesh.mesh
        elif isinstance(mesh, Mesh):
            self.__mesh = FEniCSMesh(mesh)
        else:
            raise TypeError
        self.__functionspace = FunctionSpace(mesh, family, degree)
        self.__dim = self.__functionspace.dim()

    def refine(self, cells=None):
        '''refines mesh of basis uniformly or wrt cells, return new (FEniCSBasis,prolongate,restrict)'''
        import spuq.fem.fenics.fenics_vector   # NOTE: from ... import FEniCSVector does not work (cyclic dependencies require module imports)
        newmesh = self.mesh.refine(cells)
        newFB = FEniCSBasis(newmesh, self.family, self.degree)
        prolongate = lambda vec: spuq.fem.fenics.fenics_vector.FEniCSVector(function=project(vec.F, newFB.functionspace))
        restrict = lambda vec: spuq.fem.fenics.fenics_vector.FEniCSVector(function=project(vec.F, self.functionspace))
        return (newFB,prolongate,restrict)

    def project(self, vec, vecbasis=None, ptype=PROJECTION.INTERPOLATION):
        import spuq.fem.fenics.fenics_vector   # NOTE: from ... import FEniCSVector does not work (cyclic dependencies require module imports)
        if ptype == self.PROJECTION.INTERPOLATION:
            T = interpolate
        elif ptype == self.PROJECTION.L2PROJECTION:
            T = project
        else:
            raise AttributeError
                
        if isinstance(vec, spuq.fem.fenics.fenics_vector.FEniCSVector):
            assert(vecbasis is None)
            F = T(vec.F, self.functionspace)
            newvec = spuq.fem.fenics.fenics_vector.FEniCSVector(F.vector(), F.function_space())
            return newvec
        else:
            F = T(Function(vecbasis,vec), self.functionspace)
            return (F.vector().array(), F.function_space())

    def interpolate(self, F):
        '''interpolate FEniCS Expression/Function on basis'''
        return interpolate(F, self.functionspace)

    @property
    def functionspace(self):
        return self.__functionspace

    @property
    def mesh(self):
        return self.__mesh
    
    @property
    def dim(self):
        return self.__dim
    
    def get_gramian(self):
        """Returns the Gramian as a LinearOperator"""
        u = TrialFunction(self.__functionspace)
        v = TestFunction(self.__functionspace)
        a = (u*v)*dx
        A = assemble(a)
        return MatrixOperator(A.array())
