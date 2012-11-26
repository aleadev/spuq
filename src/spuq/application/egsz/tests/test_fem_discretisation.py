import logging
import numpy as np
import dolfin

from spuq.utils.testing import *

import spuq.application.egsz.fem_discretisation as fem
from spuq.fem.fenics.fenics_basis import FEniCSBasis

def UnitSquare(n, m):
    if hasattr(dolfin, "UnitSquareMesh"):
        return dolfin.UnitSquareMesh(n, m)
    else:
        return dolfin.UnitSquare(n, m)


def to_array(z_func, V):
    """Converts a dolfin function or expression to an array by
    interpolating  or projecting onto some FunctionSpace"""
    try:
        return dolfin.interpolate(z_func, V).vector().array()
    except:
        return dolfin.project(z_func, V).vector().array()


def assert_zero_func(z_func, V):
    N = V.dim()
    z_ex = np.zeros(N)
    z = to_array(z_func, V)
    assert_equal(z, z_ex)

def assert_equal_func(func1, func2, V):
    z1 = to_array(func1, V)
    z2 = to_array(func2, V)
    assert_equal(z1, z2)


def test_helper():
    assert_equal(fem.make_list(1), [1])
    assert_equal(fem.make_list([2]), [2])
    assert_equal(fem.make_list([1, 2]), [1, 2])
    assert_equal(fem.make_list([5], 3), [5, 5, 5])
    assert_equal(fem.make_list([1, 2], 3), [1, 2])

    mesh = UnitSquare(3, 3)

    V1 = dolfin.FunctionSpace(mesh, "Lagrange", 1)
    V1v = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
    V2 = dolfin.FunctionSpace(mesh, "Lagrange", 2)
    V2v = dolfin.VectorFunctionSpace(mesh, "Lagrange", 2)
    
    assert_zero_func(fem.zero_function(V1), V1)
    assert_zero_func(fem.zero_function(V1v), V1v)
    assert_zero_func(fem.zero_function(V2), V2)
    assert_zero_func(fem.zero_function(V2v), V2v)

    assert_equal(fem.element_degree(dolfin.Function(V1)), 1)
    assert_equal(fem.element_degree(dolfin.TestFunction(V1)), 1)
    assert_equal(fem.element_degree(dolfin.TrialFunction(V1)), 1)
    assert_equal(fem.element_degree(dolfin.Function(V1v)), 1)
    assert_equal(fem.element_degree(dolfin.TestFunction(V1v)), 1)
    assert_equal(fem.element_degree(dolfin.TrialFunction(V1v)), 1)
    assert_equal(fem.element_degree(dolfin.Function(V2)), 2)
    assert_equal(fem.element_degree(dolfin.TestFunction(V2)), 2)
    assert_equal(fem.element_degree(dolfin.TrialFunction(V2)), 2)
    assert_equal(fem.element_degree(dolfin.Function(V2v)), 2)
    assert_equal(fem.element_degree(dolfin.TestFunction(V2v)), 2)
    assert_equal(fem.element_degree(dolfin.TrialFunction(V2v)), 2)


def test_poisson_wf():
    wf = fem.PoissonWeakForm()

    N = 25
    mesh = UnitSquare(N, N)
    V = wf.function_space(mesh, 1)
    a = dolfin.Expression("cos(x[0])*cos(x[1])", element=V.ufl_element())
    u = dolfin.TrialFunction(V)
    uc = dolfin.interpolate(dolfin.Expression("x[0]*x[1]"), V)

    wf.flux(u, a)
    wf.flux(uc, a)
    assert_raises(TypeError, wf.flux, a, u)

    wf.flux_derivative(u, a)
    wf.flux_derivative(uc, a)
    assert_raises(TypeError, wf.flux_derivative, a, u)

    wf.bilinear_form(V, a)
    wf.loading_linear_form(V, uc)


def test_poisson_construct():
    pde = fem.FEMPoisson()

    N = 25
    mesh = UnitSquare(N, N)
    V = pde.weak_form.function_space(mesh, 1)
    a = dolfin.Expression("cos(x[0])*cos(x[1])", element=V.ufl_element())
    u = dolfin.TrialFunction(V)

    basis = FEniCSBasis(V)
    pde.assemble_rhs(basis)
    pde.assemble_lhs(basis)
    pde.assemble_operator(basis)


def test_navierlame_construct():
    lmbda = dolfin.Constant(2400)
    mu = dolfin.Constant(400)
    pde = fem.FEMNavierLame(lmbda, mu)

    N = 25
    mesh = UnitSquare(N, N)
    V = pde.weak_form.function_space(mesh, 1)
    u = dolfin.TrialFunction(V)

    coeff = (lmbda, mu)
    basis = FEniCSBasis(V)
    pde.assemble_rhs(basis, coeff)
    pde.assemble_lhs(basis, coeff)
    pde.assemble_operator(basis, coeff)


logging.getLogger("spuq").setLevel(logging.WARNING)
logging.getLogger("nose.config").setLevel(logging.WARNING)
logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
dolfin.set_log_active(False)

test_main()
