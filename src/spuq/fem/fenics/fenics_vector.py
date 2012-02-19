import numpy as np

from dolfin import Function, FunctionSpace, FunctionSpaceBase, TestFunction, TrialFunction, CellFunction, assemble, dx
import dolfin as fe

from spuq.utils.type_check import takes, anything, optional
from spuq.utils.enum import Enum
from spuq.linalg.operator import MatrixOperator
from spuq.linalg.vector import Scalar
from spuq.linalg.basis import check_basis
from spuq.fem.fem_vector import FEMVector
from spuq.fem.fem_basis import FEMBasis

# Set option to allow extrapolation outside mesh (for interpolation)
fe.parameters["allow_extrapolation"] = True

PROJECTION = Enum('INTERPOLATION', 'L2PROJECTION')

class FEniCSBasis(FEMBasis):

    @takes(anything, FunctionSpaceBase, optional(anything))
    def __init__(self, fefs, ptype=PROJECTION.INTERPOLATION):
        self._fefs = fefs
        self._ptype = ptype

    def refine(self, cell_ids=None):
        """Refine mesh of basis uniformly or wrt cells, returns
        (prolongate,restrict,...)."""
        mesh = self._fefs.mesh()
        cell_markers = CellFunction("bool", mesh)
        if cell_ids == None:
            cell_markers.set_all(True)
        else:
            cell_markers.set_all(False)
            for cid in cell_ids:
                cell_markers[cid] = True
        new_mesh = fe.refine(mesh, cell_markers)
        new_fs = FunctionSpace(new_mesh, self._fefs.ufl_element().family(), self._fefs.ufl_element().degree())
        new_basis = FEniCSBasis(new_fs)
        prolongate = new_basis.project_onto
        restrict = self.project_onto
        return (new_basis, prolongate, restrict)

    @takes(anything, "FEniCSVector")
    def project_onto(self, vec):
        if self._ptype == PROJECTION.INTERPOLATION:
            new_fefunc = fe.interpolate(vec._fefunc, self._fefs)
        elif self._ptype == PROJECTION.L2PROJECTION:
            new_fefunc = fe.project(vec._fefunc, self._fefs)
        else:
            raise AttributeError
        return FEniCSVector(new_fefunc)

    @property
    def dim(self):
        return self._fefs.dim()

    @property
    def mesh(self):
        return self._fefs.mesh()

    def eval(self, x):  # pragma: no coverage
        """Evaluate the basis functions at point x where x has length domain_dim."""
        raise NotImplementedError

    @property
    def domain_dim(self):
        """The dimension of the domain the functions are defined upon."""
        return self._fefs.cell().topological_dimension()

    @property
    def gramian(self):
        """The Gramian as a LinearOperator (not necessarily a matrix)"""
        # TODO: wrap sparse FEniCS matrix A in FEniCSOperator
        u = TrialFunction(self._fefs)
        v = TestFunction(self._fefs)
        a = (u * v) * dx
        A = assemble(a)
        return MatrixOperator(A.array())

    @takes(anything, "FEniCSBasis")
    def __eq__(self, other):
        return (type(self) == type(other) and
                self._fefs == other._fefs)


class FEniCSVector(FEMVector):
    '''Wrapper for FEniCS/dolfin Function.

        Provides a FEniCSBasis and a FEniCSFunction (with the respective coefficient vector).'''

    @takes(anything, Function)
    def __init__(self, fefunc):
        '''Initialise with coefficient vector and Function'''
        self._fefunc = fefunc

    @property
    def basis(self):
        '''return FEniCSBasis'''
        return FEniCSBasis(self._fefunc.function_space())

    @property
    def coeffs(self):
        '''return FEniCS coefficient vector of Function'''
        return self._fefunc.vector()

    @coeffs.setter
    def coeffs(self, val):
        '''set FEniCS coefficient vector of Function'''
        self._fefunc.vector()[:] = val

    def array(self):
        '''return copy of coefficient vector as numpy array'''
        return self._fefunc.vector().array()

    def eval(self, x):
        return self._fefunc(x)

    def _create_copy(self, coeffs):
        # TODO: remove create_copy and retain only copy()
        new_fefunc = Function(self._fefunc.function_space(), coeffs)
        return self.__class__(new_fefunc)

    def __eq__(self, other):
        """Compare vectors for equality.

        Note that vectors are only considered equal when they have
        exactly the same type."""
#        print "************* EQ "
#        print self.coeffs.array()
#        print other.coeffs.array()
#        print (type(self) == type(other),
#                self.basis == other.basis,
#                self.coeffs.size() == other.coeffs.size())

        return (type(self) == type(other) and
                self.basis == other.basis and
                self.coeffs.size() == other.coeffs.size() and
                (self.coeffs == other.coeffs).all())

    @takes(anything)
    def __neg__(self):
        return self._create_copy(-self.coeffs)

    @takes(anything, "FEniCSVector")
    def __iadd__(self, other):
        check_basis(self.basis, other.basis)
        self.coeffs += other.coeffs
        return self

    @takes(anything, "FEniCSVector")
    def __isub__(self, other):
        check_basis(self.basis, other.basis)
        self.coeffs -= other.coeffs
        return self

    @takes(anything, Scalar)
    def __imul__(self, other):
        self.coeffs *= other
        return self
