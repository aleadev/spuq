from exceptions import *
from spuq.utils.enum import Enum
#from spuq.fem import *
#from spuq.fem.fenics import *
from spuq.fem.fem_basis import FEMBasis
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.fem.fenics.fenics_mesh import FEniCSMesh
from dolfin import FunctionSpace, Function, Mesh
from dolfin.fem.interpolation import interpolate
from dolfin.fem.projection import project

class FEniCSBasis(FEMBasis):
    '''wrapper for FEniCS/dolfin FunctionSpace'''
    
    PROJECTION = Enum('INTERPOLATION','L2PROJECTION')

    def __init__(self, mesh=None, family='CG', degree=1):
        '''initialise discrete basis on mesh'''
        assert(mesh is not None)
        self.family = family
        self.degree = degree
        if isinstance(mesh, FEniCSMesh):
            self.mesh = mesh
            mesh = mesh.mesh
        elif isinstance(mesh, Mesh):
            self.mesh = FEniCSMesh(mesh)
        else:
            raise TypeError
        self.basis = FunctionSpace(mesh, family, degree)
        self._dim = self.basis.dim()

    def refine(self, cells=None):
        '''refines mesh of basis uniformly or wrt cells, return new (FEniCSBasis,prolongate,restrict)'''
        newmesh = self.mesh.refine(cells)
        newFB = FEniCSBasis(newmesh, self.family, self.degree)
        prolongate = lambda vec: FEniCSVector(function=project(vec.F, newFB.basis))
        restrict = lambda vec: FEniCSVector(function=project(vec.F, self.basis))
        return (newFB,prolongate,restrict)

    def project(self, vec, vecbasis=None, ptype=PROJECTION.INTERPOLATION):
        if ptype == self.PROJECTION.INTERPOLATION:
            T = interpolate
        elif ptype == self.PROJECTION.L2PROJECTION:
            T = project
        else:
            raise AttributeError
                
        if isinstance(vec, FEniCSVector):
            assert(vecbasis is None)
            F = T(vec.F, self.basis)
            newvec = FEniCSVector(F.vector(), F.function_space()) 
            return newvec
        else:
            F = T(Function(vecbasis,vec), self.basis)
            return (F.vector(), F.function_space())

    def interpolate(self, F):
        '''interpolate FEniCS Expression/Function on basis'''
        return interpolate(F, self.basis)

    @property
    def basis(self):
        return self._basis          # dolfin FunctionSpace

    @basis.setter
    def basis(self, val):
        self._basis = val

    @property
    def mesh(self):
        return self._mesh
    
    @mesh.setter
    def mesh(self, val):
        self._mesh = val
    
    @property
    def dim(self):
        return self._dim
    
    # TODO: implement       
    def get_gramian(self):
        """Returns the Gramian as a LinearOperator"""
        return NotImplemented
