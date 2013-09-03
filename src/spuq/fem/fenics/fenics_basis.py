from dolfin import FunctionSpace, VectorFunctionSpace, FunctionSpaceBase, Function, TestFunction, TrialFunction, CellFunction, assemble, dx, refine, cells, nabla_grad
import dolfin
import numpy as np

from spuq.utils.type_check import takes, anything, optional
from spuq.utils.enum import Enum
from spuq.linalg.operator import MatrixOperator
from spuq.fem.fem_basis import FEMBasis
from spuq.linalg.basis import CanonicalBasis

# Set option to allow extrapolation outside mesh (for interpolation)
dolfin.parameters["allow_extrapolation"] = True

PROJECTION = Enum('INTERPOLATION', 'L2PROJECTION')

class FEniCSBasis(FEMBasis):

    @takes(anything, FunctionSpaceBase, optional(anything))
    def __init__(self, fefs, ptype=PROJECTION.INTERPOLATION):
        self._fefs = fefs
        self._ptype = ptype

    def copy(self, degree=None, mesh=None):
        """Make a copy of self. The degree may be overriden optionally."""
        if (degree is None or self._fefs.ufl_element().degree() == degree) and mesh is None:
            return FEniCSBasis(self._fefs, self._ptype)
        else:
            if mesh is None:
                mesh = self._fefs.mesh()
            if degree is None:
                degree = self._fefs.ufl_element().degree()
            if self._fefs.num_sub_spaces() > 1:
                newfs = VectorFunctionSpace(mesh, self._fefs.ufl_element().family(), degree)
            else:
                newfs = FunctionSpace(mesh, self._fefs.ufl_element().family(), degree)
            return FEniCSBasis(newfs, self._ptype)

    def get_dof_coordinates(self):
        V = self._fefs
        # degrees of freedom
        N = V.dofmap().global_dimension()
        c4dof = np.zeros((N, V.mesh().geometry().dim()))
        # evaluate nodes matrix by local-to-global map on each cells
        for c in cells(V.mesh()):
            # coordinates of nodes in current cell
            cell_c4dof = V.dofmap().tabulate_coordinates(c)
            # global indices of nodes in current cell
            nodes = V.dofmap().cell_dofs(c.index())
            # set global nodes coordinates
            c4dof[nodes] = cell_c4dof
        return c4dof

    def new_vector(self, sub_spaces=None):
        """Create null vector on this space."""
        import spuq.fem.fenics.fenics_vector as FV          # this circumvents circular inclusions
        if sub_spaces == 0 and self.num_sub_spaces > 0:
            V = FunctionSpace(self._fefs.mesh(), self._fefs.ufl_element().family(), self._fefs.ufl_element().degree())
            return FV.FEniCSVector(Function(V))
        else:
            assert sub_spaces is None or sub_spaces == self.num_sub_spaces
            return FV.FEniCSVector(Function(self._fefs))

    def refine(self, cell_ids=None):
        """Refine mesh of basis uniformly or wrt cells, returns (new_basis,prolongate,restrict)."""
        mesh = self._fefs.mesh()
        cell_markers = CellFunction("bool", mesh)
        if cell_ids is None:
            cell_markers.set_all(True)
        else:
            cell_markers.set_all(False)
            for cid in cell_ids:
                cell_markers[cid] = True
        new_mesh = refine(mesh, cell_markers)
#        if isinstance(self._fefs, VectorFunctionSpace):
        if self._fefs.num_sub_spaces() > 1:
            new_fs = VectorFunctionSpace(new_mesh, self._fefs.ufl_element().family(), self._fefs.ufl_element().degree())
        else:
            new_fs = FunctionSpace(new_mesh, self._fefs.ufl_element().family(), self._fefs.ufl_element().degree())
        new_basis = FEniCSBasis(new_fs)
        prolongate = new_basis.project_onto
        restrict = self.project_onto
        return new_basis, prolongate, restrict

    def refine_maxh(self, maxh, uniform=False):
        """Refine mesh of FEM basis such that maxh of mesh is smaller than given value."""
        if maxh <= 0 or self.mesh.hmax() < maxh:            
            return self, self.project_onto, self.project_onto, 0
        ufl = self._fefs.ufl_element()
        mesh = self.mesh
        num_cells_refined = 0
        if uniform:
            while mesh.hmax() > maxh:
                num_cells_refined += mesh.num_cells()
                mesh = refine(mesh)         # NOTE: this global refine results in a red-refinement as opposed to bisection in the adaptive case
        else:
            while mesh.hmax() > maxh:
                cell_markers = CellFunction("bool", mesh)
                cell_markers.set_all(False)
                for c in cells(mesh):
                    if c.diameter() > maxh:
                        cell_markers[c.index()] = True
                        num_cells_refined += 1
                mesh = refine(mesh, cell_markers)
        if self._fefs.num_sub_spaces() > 1:
            new_fefs = VectorFunctionSpace(mesh, ufl.family(), ufl.degree())
        else:
            new_fefs = FunctionSpace(mesh, ufl.family(), ufl.degree())
        new_basis = FEniCSBasis(new_fefs)
        prolongate = new_basis.project_onto
        restrict = self.project_onto
        return new_basis, prolongate, restrict, num_cells_refined

    @takes(anything, "FEniCSVector", anything)
    def project_onto(self, vec, ptype=None):
        import spuq.fem.fenics.fenics_vector as FV          # this circumvents circular inclusions
        if ptype is None:
            ptype = self._ptype
        if ptype == PROJECTION.INTERPOLATION:
#            print "fenics_basis::project_onto"
#            print vec._fefunc.value_size()
#            print "sub spaces", self.num_sub_spaces
#            print type(self._fefs)
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
    def num_sub_spaces(self):
        return self._fefs.num_sub_spaces()

    @property
    def mesh(self):
        return self._fefs.mesh()

    @property
    def maxh(self):
        return self.mesh.hmax()

    @property
    def minh(self):
        return self.mesh.hmin()

    @property
    def degree(self):
        return self._fefs.ufl_element().degree()

    @property
    def family(self):
        return self._fefs.ufl_element().family()

    def eval(self, x):  # pragma: no coverage
        """Evaluate the basis functions at point x where x has length domain_dim."""
        raise NotImplementedError

    @property
    def domain_dim(self):
        """The dimension of the domain the functions are defined upon."""
        return self._fefs.cell().topological_dimension()

    @property
    def gramian(self):
        """The Gramian as a LinearOperator (not necessarily a matrix)."""
        # TODO: wrap sparse FEniCS matrix A in FEniCSOperator
        u = TrialFunction(self._fefs)
        v = TestFunction(self._fefs)
        a = (u * v) * dx
        A = assemble(a)
        return MatrixOperator(A.array())

    @property
    def stiffness(self):
        """The stiffness matrix as a LinearOperator."""
        u = TrialFunction(self._fefs)
        v = TestFunction(self._fefs)
        a = (nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        return MatrixOperator(A.array())

    @takes(anything, "FEniCSBasis")
    def __eq__(self, other):
        ufl1 = self._fefs.ufl_element()
        ufl2 = other._fefs.ufl_element()
        mesh1 = self._fefs.mesh()
        mesh2 = other._fefs.mesh()
        return (type(self) == type(other) and
                ufl1.degree() == ufl2.degree() and
                ufl1.family() == ufl2.family() and
                self._fefs.num_sub_spaces() == other._fefs.num_sub_spaces() and
                (mesh1.cells() == mesh2.cells()).all() and
                (mesh1.coordinates() == mesh2.coordinates()).all())

    def as_canonical_basis(self):
        return CanonicalBasis(self.dim)
