from __future__ import division

from spuq.utils.type_check import takes, anything, optional, sequence_of
from dolfin import (FunctionSpace, Expression, dx, inner,
                    nabla_grad, TrialFunction, TestFunction,
                    assemble, Constant, DirichletBC, refine,
                    Function, norm, Mesh, CellFunction, cells,
                    GenericMatrix, GenericVector, interpolate, plot)
from spuq.application.egsz.multi_vector import MultiVector
from spuq.fem.fenics.fenics_vector import FEniCSVector

from types import NoneType
from collections import defaultdict
from math import sqrt
import numpy as np


@takes(Expression, FunctionSpace)
def dirichlet_bc(u0, V):
    def u0_boundary(x, on_boundary):
        return on_boundary
    return DirichletBC(V, u0, u0_boundary)


@takes(float, FunctionSpace)
def constant_dirichlet_bc(u0, V):
    return dirichlet_bc(Constant(u0))


def homogeneous_dirichlet_bc(V):
    return dirichlet_bc(Constant(0.0))


DEFAULT_BC = object()


@takes((Expression, Function), FunctionSpace)
def poisson_bilinear_form(coeff_func, V):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # setup problem, assemble and apply boundary conditions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(coeff_func * nabla_grad(u), nabla_grad(v)) * dx
    return a


@takes(anything, (DirichletBC, NoneType, object), optional(FunctionSpace))
def apply_bc(fenics_obj, bc=DEFAULT_BC, V=None):
    if bc is not None:
        if bc is DEFAULT_BC:
            if V is None:
                V = fenics_obj.function_space()
            bc = homogeneous_dirichlet_bc(V)
        bc.apply(fenics_obj)
    return fenics_obj


@takes(GenericMatrix, (DirichletBC, sequence_of(DirichletBC)))
def _remove_boundary_entries(A, bcs):
    if not isinstance(bcs, DirichletBC):
        for bc in bcs:
            remove_boundary_entries(A, bc)
    else:
        dofs = bcs.get_boundary_values().keys()
        values = np.zeros(1, dtype=np.float_)
        rows = np.array([0], dtype=np.uintc)
        cols = np.array([0], dtype=np.uintc)
        A.apply("insert")   # TODO: not sure about these applies
        for d in dofs:
            rows[0] = d
            cols[0] = d
            A.set(values, rows, cols)
        A.apply("insert")


@takes(GenericMatrix, (DirichletBC, sequence_of(DirichletBC)))
def get_dirichlet_mask(A, bcs):
    import collections
    if not isinstance(bcs, collections.Iterable):
        bcs = [bcs]
    mask = np.ones(A.size(0))
    for bc in bcs:
        dofs = bc.get_boundary_values().keys()
        mask[dofs] = 0
    return mask


@takes(GenericVector, (DirichletBC, sequence_of(DirichletBC)), bool)
def set_dirichlet_bc_entries(u, bcs, homogeneous):
    if not isinstance(bcs, DirichletBC):
        for bc in bcs:
            set_dirichlet_bc_entries(u, bc, homogeneous)
    else:
        dof2val = bcs.get_boundary_values()
        dofs = dof2val.keys()
        if homogeneous:
            u[dofs] = 0.0
        else:
            vals = dof2val.values()
            u[dofs] = np.array(vals)


@takes((Expression, Function), FunctionSpace, (DirichletBC, NoneType, object))
def assemble_poisson_matrix(coeff_func, V, bc=DEFAULT_BC):
    a = poisson_bilinear_form(coeff_func, V)
    A = assemble(a)
    return apply_bc(A, bc)


@takes(FunctionSpace)
def l2_gramian_bilinear_form(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    b = u * v * dx
    return b


@takes(FunctionSpace)
def assemble_l2_gramian_matrix(V):
    b = l2_gramian_bilinear_form(V)
    return assemble(b)


@takes((Expression, Function), FunctionSpace, (DirichletBC, NoneType, object))
def rhs_linear_form(coeff_func, V):
    v = TestFunction(V)
    L = coeff_func * v * dx
    return L


@takes((Expression, Function), FunctionSpace, (DirichletBC, NoneType, object))
def assemble_rhs(coeff_func, V, bc=DEFAULT_BC):
    L = rhs_linear_form(coeff_func, V)
    b = apply_bc(L, bc)
    return b


#@takes(MultiVector, MultiVector, optional(str))
def error_norm(vec1, vec2, normstr="L2"):
        e = 0.0
        if isinstance(vec1, MultiVector):
            assert isinstance(vec2, MultiVector)
            assert vec1.active_indices() == vec2.active_indices()
            for mi in vec1.keys():
                V = vec1[mi]._fefunc.function_space()
                errfunc = Function(V, vec1[mi]._fefunc.vector() - vec2[mi]._fefunc.vector())
                if isinstance(normstr, str):
                    e += norm(errfunc, normstr) ** 2
                else:
                    e += normstr(errfunc) ** 2
            return sqrt(e)
        else:
            V = vec1.function_space()
            errfunc = Function(V, vec1.vector() - vec2.vector())
            if isinstance(normstr, str):
                return norm(errfunc, normstr)
            else:
                return normstr(errfunc)


def plot_indicators(indicators, mesh, refinements=1, interactive=True):
    DG = FunctionSpace(mesh, 'DG', 0)
    for _ in range(refinements):
        mesh = refine(mesh)
    V = FunctionSpace(refine(mesh), 'CG', 1)
    if len(indicators) == 1 and not isinstance(indicators[0], (list, tuple)):
        indicators = [indicators]
    for eta, title in indicators:
        e = Function(DG, eta)
        f = interpolate(e, V) 
        plot(f, title=title)
    if interactive:
        from dolfin import interactive
        interactive()


@takes(anything, FEniCSVector)
def weighted_H1_norm(w, vec, piecewise=False):
    if piecewise:
        DG = FunctionSpace(vec.basis.mesh, "DG", 0)
        s = TestFunction(DG)
        ae = assemble(w * inner(nabla_grad(vec._fefunc), nabla_grad(vec._fefunc)) * s * dx)
        norm_vec = np.array([sqrt(e) for e in ae])
        # map DG dofs to cell indices
        dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(vec.basis.mesh)]
        norm_vec = norm_vec[dofs]
    else:
        ae = assemble(w * inner(nabla_grad(vec._fefunc), nabla_grad(vec._fefunc)) * dx)
        norm_vec = sqrt(ae)
    return norm_vec


@takes((list, tuple), optional(Mesh))
def create_joint_mesh(meshes, destmesh=None, additional_refine=0):
    if destmesh is None:
        # start with finest mesh to avoid (most) refinements
#        hmin = [m.hmin() for m in meshes]
#        mind = hmin.index(min(hmin))
        numcells = [m.num_cells() for m in meshes]
        mind = numcells.index(max(numcells))
        destmesh = meshes.pop(mind)
        
    # setup parent cells
    parents = {}
    for c in cells(destmesh):
        parents[c.index()] = [c.index()]
    PM = []

    # refinement loop for destmesh    
    for m in meshes:
        # loop until all cells of destmesh are finer than the respective cells in the set of meshes
        while True: 
            cf = CellFunction("bool", destmesh)
            cf.set_all(False)
            rc = 0      # counter for number of marked cells
            # get cell sizes of current mesh
            h = [c.diameter() for c in cells(destmesh)]
            # check all cells with destination sizes and mark for refinement when destination mesh is coarser (=larger)
            for c in cells(m):
                p = c.midpoint()
                cid = destmesh.closest_cell(p)
                if h[cid] > c.diameter():
                    cf[cid] = True
                    rc += 1
            # carry out refinement if any cells are marked
            if rc:
                # refine marked cells
                newmesh = refine(destmesh, cf)
                # determine parent cell association map
                pc = newmesh.data().array("parent_cell", newmesh.topology().dim())
                pmap = defaultdict(list)
                for i, cid in enumerate(pc):
                    pmap[cid].append(i)
                PM.append(pmap)
                # set refined mesh as current mesh
                destmesh = newmesh
            else:
                break

        # carry out additional uniform refinements
        for _ in range(additional_refine):
            # refine uniformly
            newmesh = refine(destmesh)
            # determine parent cell association map
            pc = newmesh.data().array("parent_cell", newmesh.topology().dim())
            pmap = defaultdict(list)
            for i, cid in enumerate(pc):
                pmap[cid].append(i)
            PM.append(pmap)
            # set refined mesh as current mesh
            destmesh = newmesh

    # determine association to parent cells
    for level in range(len(PM)):
        for parentid, childids in parents.iteritems():
            newchildids = []
            for cid in childids:
                for cid in PM[level][cid]:
                    newchildids.append(cid)
            parents[parentid] = newchildids
    
    return destmesh, parents
