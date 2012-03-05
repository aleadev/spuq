from dolfin import FunctionSpace, FunctionSpaceBase, TestFunction, TrialFunction, CellFunction, assemble, dx
import dolfin as fe

from spuq.utils.type_check import takes, anything, optional
from spuq.utils.enum import Enum
from spuq.linalg.operator import MatrixOperator
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
        if cell_ids is None:
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
        import spuq.fem.fenics.fenics_vector as FV          # this circumvents circular inclusions
        if self._ptype == PROJECTION.INTERPOLATION:
            new_fefunc = fe.interpolate(vec._fefunc, self._fefs)
        elif self._ptype == PROJECTION.L2PROJECTION:
            new_fefunc = fe.project(vec._fefunc, self._fefs)
        else:
            raise AttributeError
        return FV.FEniCSVector(new_fefunc)

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
