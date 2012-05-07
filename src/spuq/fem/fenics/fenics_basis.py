from dolfin import FunctionSpace, FunctionSpaceBase, Function, TestFunction, TrialFunction, CellFunction, assemble, dx, refine, cells
import dolfin

from spuq.utils.type_check import takes, anything, optional
from spuq.utils.enum import Enum
from spuq.linalg.operator import MatrixOperator
from spuq.fem.fem_basis import FEMBasis

# Set option to allow extrapolation outside mesh (for interpolation)
dolfin.parameters["allow_extrapolation"] = True

PROJECTION = Enum('INTERPOLATION', 'L2PROJECTION')

class FEniCSBasis(FEMBasis):

    @takes(anything, FunctionSpaceBase, optional(anything))
    def __init__(self, fefs, ptype=PROJECTION.INTERPOLATION):
        self._fefs = fefs
        self._ptype = ptype

    def copy(self, degree=None):
        """Make a copy of self. The degree may be overriden optionally."""
        if degree is None or self._fefs.ufl_element().degree() == degree:
            return FEniCSBasis(self._fefs, self._ptype)
        else:
            newfs = FunctionSpace(self._fefs.mesh(), self._fefs.ufl_element().family(), degree)
            return FEniCSBasis(newfs, self._ptype) 

    def new_vec(self):
        """Create null vector on this space."""
        import spuq.fem.fenics.fenics_vector as FV          # this circumvents circular inclusions
        return FV.FEniCSVector(Function(self._fefs)) 

    def refine(self, cell_ids=None):
        """Refine mesh of basis uniformly or wrt cells, returns
        (new_basis,prolongate,restrict)."""
        mesh = self._fefs.mesh()
        cell_markers = CellFunction("bool", mesh)
        if cell_ids is None:
            cell_markers.set_all(True)
        else:
            cell_markers.set_all(False)
            for cid in cell_ids:
                cell_markers[cid] = True
        new_mesh = refine(mesh, cell_markers)
        new_fs = FunctionSpace(new_mesh, self._fefs.ufl_element().family(), self._fefs.ufl_element().degree())
        new_basis = FEniCSBasis(new_fs)
        prolongate = new_basis.project_onto
        restrict = self.project_onto
        return (new_basis, prolongate, restrict)

    def refine_maxh(self, maxh, uniform=False):
        """Refine mesh of FEM basis such that maxh of mesh is smaller than given value."""
        if maxh == 0 or self.mesh.hmax() < maxh:
            return self
        ufl = self._fefs.ufl_element()
        mesh = self.mesh
        if uniform:
            while mesh.hmax() > maxh:
                mesh = refine(mesh)
        else:
            while mesh.hmax() > maxh:
                cell_markers = CellFunction("bool", mesh)
                cell_markers.set_all(False)
                for c in cells(mesh):
                    if c.diameter() > maxh:
                        cell_markers[c.index()] = True
                mesh = refine(mesh, cell_markers) 
        new_fefs = FunctionSpace(mesh, ufl.family(), ufl.degree())
        return FEniCSBasis(new_fefs, self._ptype)

    @takes(anything, "FEniCSVector", anything)
    def project_onto(self, vec, ptype=None):
        import spuq.fem.fenics.fenics_vector as FV          # this circumvents circular inclusions
        if ptype is None:
            ptype = self._ptype
        if ptype == PROJECTION.INTERPOLATION:
            new_fefunc = dolfin.interpolate(vec._fefunc, self._fefs)
        elif ptype == PROJECTION.L2PROJECTION:
            new_fefunc = dolfin.project(vec._fefunc, self._fefs)
        else:
            raise AttributeError
        return FV.FEniCSVector(new_fefunc)

    @property
    def dim(self):
        return self._fefs.dim()

    @property
    def mesh(self):
        return self._fefs.mesh()

    @property
    def maxh(self):
        return self.mesh.maxh()

    @property
    def degree(self):        
        return self._fefs.ufl_element().degree()

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
