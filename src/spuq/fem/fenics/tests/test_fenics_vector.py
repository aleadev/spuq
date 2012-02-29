import numpy as np
from dolfin import UnitSquare, FunctionSpace, Expression, interpolate

from spuq.utils.testing import *
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fenics.fenics_vector import FEniCSVector

def test_fenics_basis():
    mesh = UnitSquare(5, 5)
    fs1 = FunctionSpace(mesh, "CG", 1)
    fs2 = FunctionSpace(mesh, "DG", 1)
    basis1 = FEniCSBasis(fs1)
    basis1b = FEniCSBasis(fs1)
    basis2 = FEniCSBasis(fs2)
    assert_true(basis1 == basis1)
    assert_false(basis1 == basis2)
    assert_true(basis1 == basis1b)
    assert_false(basis1 != basis1)
    assert_true(basis1 != basis2)
    assert_false(basis1 != basis1b)

    assert_equal(basis1.dim, 36)
    assert_equal(basis1.domain_dim, 2)

    basis1.gramian.as_matrix()

def test_fenics_vector():
    mesh = UnitSquare(3, 3)
    fs1 = FunctionSpace(mesh, "CG", 2)
    fs2 = FunctionSpace(mesh, "DG", 1)
    ex1 = Expression("1.+x[0]*x[1]")
    ex2 = Expression("2.*x[0] - x[1]")
    ex3 = Expression("1. + 2.*x[0] - x[1] + x[0]*x[1]")
    f1 = interpolate(ex1, fs1)
    f2 = interpolate(ex2, fs1)
    f3 = interpolate(ex1, fs2)
    f4 = interpolate(ex1, fs1)
    f5 = interpolate(ex3, fs1)
    vec1 = FEniCSVector(f1)
    vec2 = FEniCSVector(f2)
    vec3 = FEniCSVector(f3)
    vec4 = FEniCSVector(f4)
    vec5 = FEniCSVector(f5)

    assert_true(vec1 == vec1)
    assert_false(vec1 == vec2)
    assert_false(vec1 == vec3)
    assert_true(vec1 == vec4)
    assert_false(vec1 != vec1)
    assert_true(vec1 != vec2)
    assert_true(vec1 != vec3)
    assert_false(vec1 != vec4)

    vec12 = vec1 + vec2
    vec12b = vec1 + vec2
    vec1m = -vec1
    vec1m3 = 3 * vec1
    vec21 = vec2 + vec1
    vec14 = vec1 - vec4

    assert_equal(vec12, vec21)
    assert_almost_equal(vec12.array(), vec5.array())
    assert_almost_equal(vec12.array(), vec12b.array())
    assert_equal(vec14.array(), np.zeros(vec12.coeffs.size()))
    assert_equal(vec1m.array(), -vec1.array())
    assert_equal(vec1m3.array(), 3 * vec1.array())

    assert_almost_equal(vec1.eval([0.8, 0.4]), 1.32)


def test_fenics_project():
    mesh1 = UnitSquare(3, 3)
    fs1 = FunctionSpace(mesh1, "CG", 2)
    fs2 = FunctionSpace(mesh1, "CG", 2)
    exa = Expression("1.+x[0]*x[1]")
    exb = Expression("2.*x[0]-x[1]")
    f1a = interpolate(exa, fs1)
    f1b = interpolate(exb, fs1)
    f2a = interpolate(exa, fs2)
    f2b = interpolate(exb, fs2)
    vec1a = FEniCSVector(f1a)
    vec1b = FEniCSVector(f1b)
    vec2a = FEniCSVector(f2a)
    vec2b = FEniCSVector(f2b)

    assert_equal(vec1a.basis.project_onto(vec2b), vec1b)
    assert_equal(vec2a.basis.project_onto(vec1b), vec2b)
    #assert_equal(vec1a.basis.project_onto(vec2b).array(), vec1b.array())

def test_fenics_refine():
    mesh1 = UnitSquare(3, 3)
    fs1 = FunctionSpace(mesh1, "CG", 2)
    exa = Expression("1.+x[0]*x[1]")
    f1a = interpolate(exa, fs1)
    vec1a = FEniCSVector(f1a)
    (basis2, prolongate, restrict) = vec1a.basis.refine((1, 3, 15))
    vec2 = prolongate(vec1a)
    assert_equal(vec2.basis, basis2)
    vec1b = restrict(vec2)
    assert_equal(vec1b.basis, vec1a.basis)
    assert_almost_equal(vec1a.array(), vec1b.array())


test_main()
