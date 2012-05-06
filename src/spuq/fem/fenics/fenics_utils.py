from types import NoneType
from spuq.utils.type_check import takes, anything, optional
from dolfin import (FunctionSpace, Expression, dx, inner,
                    nabla_grad, TrialFunction, TestFunction,
                    assemble, Constant, DirichletBC,
                    Function, norm)
from dolfin.cpp import BoundaryCondition
from spuq.application.egsz.multi_vector import MultiVector
from spuq.fem.fenics.fenics_vector import FEniCSVector

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


@takes(anything, (BoundaryCondition, NoneType, object), optional(FunctionSpace))
def apply_bc(fenics_obj, bc=DEFAULT_BC, V=None):
    if bc is not None:
        if bc is DEFAULT_BC:
            if V is None:
                V = fenics_obj.function_space()
            bc = homogeneous_dirichlet_bc(V)
        bc.apply(fenics_obj)
    return fenics_obj


@takes((Expression, Function), FunctionSpace, (BoundaryCondition, NoneType, object))
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


@takes((Expression, Function), FunctionSpace, (BoundaryCondition, NoneType, object))
def rhs_linear_form(coeff_func, V):
    v = TestFunction(V)
    L = coeff_func * v * dx
    return L


@takes((Expression, Function), FunctionSpace, (BoundaryCondition, NoneType, object))
def assemble_rhs(coeff_func, V, bc=DEFAULT_BC):
    L = rhs_linear_form(coeff_func, V)
    b = apply_bc(L, bc)
    return b


@takes(MultiVector, MultiVector, optional(str))
def error_norm(vec1, vec2, normstr="L2"):
        assert vec1.active_indices() == vec2.active_indices()
        e = 0.0
        for mi in vec1.keys():
            V = vec1[mi]._fefunc.function_space()
            errfunc = Function(V, vec1[mi]._fefunc.vector() - vec2[mi]._fefunc.vector())
            e += norm(errfunc, normstr) ** 2
        return sqrt(e)


@takes(anything, FEniCSVector)
def weighted_H1_norm(w, vec, piecewise=False):
    ae = assemble(w * inner(nabla_grad(vec._fefunc), nabla_grad(vec._fefunc)) * dx)
    if piecewise:
        norm_vec = np.array([sqrt(e) for e in ae])
    else:
        norm_vec = sqrt(ae)
    return norm_vec
