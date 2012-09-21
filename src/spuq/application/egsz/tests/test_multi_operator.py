from __future__ import division
import numpy as np
import logging

from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.coefficient_field import CoefficientField, ListCoefficientField
from spuq.linalg.basis import CanonicalBasis
from spuq.linalg.vector import FlatVector
from spuq.linalg.operator import DiagonalMatrixOperator, MultiplicationOperator
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.polyquad.polynomials import LegendrePolynomials, StochasticHermitePolynomials
from spuq.utils.testing import assert_equal, assert_almost_equal, skip_if, test_main, assert_raises
from spuq.math_utils.multiindex import Multiindex

try:
    from dolfin import Expression, FunctionSpace, UnitSquare, interpolate
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector

    HAVE_FENICS = True
except:
    HAVE_FENICS = False

# setup logging
logging.basicConfig(filename=__file__[:-2] + 'log', level=logging.INFO)

class SimpleProjectBasis(CanonicalBasis):
    def project_onto(self, vec):
        assert self.dim == vec.basis.dim
        return vec.copy()


def diag_assemble(func, basis):
    diag = np.array([np.abs(func(x)) for x in np.linspace(0, 1, basis.dim)])
    return DiagonalMatrixOperator(diag, domain=basis, codomain=basis)


def test_init():
    mean_func = ConstFunction(1.0)
    a = [SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = ListCoefficientField(mean_func, a, rvs)

    MultiOperator(coeff_field, diag_assemble)
    assert_raises(TypeError, MultiOperator, 3, diag_assemble)
    assert_raises(TypeError, MultiOperator, coeff_field, 7)

    domain = CanonicalBasis(3)
    codomain = CanonicalBasis(5)
    A = MultiOperator(coeff_field, diag_assemble, None, domain, codomain)
    assert_equal(A.domain, domain)
    assert_equal(A.codomain, codomain)


def test_apply():
    N = 4
    #a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    mean_func = ConstFunction(2.0)
    a = [ConstFunction(3.0), ConstFunction(4.0)]
    rvs = [UniformRV(), NormalRV(mu=0.5)]
    coeff_field = ListCoefficientField(mean_func, a, rvs)

    A = MultiOperator(coeff_field, diag_assemble)
    mis = [Multiindex([0]),
           Multiindex([1]),
           Multiindex([0, 1]),
           Multiindex([0, 2])]
    vecs = [FlatVector(np.random.random(N), SimpleProjectBasis(N)),
            FlatVector(np.random.random(N), SimpleProjectBasis(N)),
            FlatVector(np.random.random(N), SimpleProjectBasis(N)),
            FlatVector(np.random.random(N), SimpleProjectBasis(N))]

    w = MultiVectorWithProjection()
    for i in range(len(mis)):
        w[mis[i]] = vecs[i]
    v = A * w

    L = LegendrePolynomials(normalised=True)
    H = StochasticHermitePolynomials(mu=0.5, normalised=True)
    v0_ex = (2 * vecs[0] +
             3 * (L.get_beta(0)[1] * vecs[1] - L.get_beta(0)[0] * vecs[0]) +
             4 * (H.get_beta(0)[1] * vecs[2] - H.get_beta(0)[0] * vecs[0]))
    v2_ex = (2 * vecs[2] + 4 * (H.get_beta(1)[1] * vecs[3] -
                                H.get_beta(1)[0] * vecs[2] +
                                H.get_beta(1)[-1] * vecs[0]))

    assert_equal(v[mis[0]], v0_ex)
    assert_equal(v[mis[2]], v2_ex)


@skip_if(not HAVE_FENICS)
def test_fenics_vector():
    def mult_assemble(a, basis):
        return MultiplicationOperator(a(0), basis)

    mean_func = ConstFunction(2)
    a = [ConstFunction(3), ConstFunction(4)]
    rvs = [UniformRV(), NormalRV(mu=0.5)]
    coeff_field = ListCoefficientField(mean_func, a, rvs)

    A = MultiOperator(coeff_field, mult_assemble)
    mis = [Multiindex([0]),
           Multiindex([1]),
           Multiindex([0, 1]),
           Multiindex([0, 2])]
    mesh = UnitSquare(4, 4)
    fs = FunctionSpace(mesh, "CG", 4)
    F = [interpolate(Expression("*".join(["x[0]"] * i)), fs) for i in range(1, 5)]
    vecs = [FEniCSVector(f) for f in F]

    w = MultiVectorWithProjection()
    for mi, vec in zip(mis, vecs):
        w[mi] = vec
    v = A * w

    L = LegendrePolynomials(normalised=True)
    H = StochasticHermitePolynomials(mu=0.5, normalised=True)
    ex0 = Expression("2*x[0] + 3*(l01*x[0]*x[0]-l00*x[0]) + 4*(h01*x[0]*x[0]*x[0]-h00*x[0])",
        l01=L.get_beta(0)[1], l00=L.get_beta(0)[0],
        h01=H.get_beta(0)[1], h00=H.get_beta(0)[0])
    vec0 = FEniCSVector(interpolate(ex0, fs))

    assert_almost_equal(v[mis[0]].array(), vec0.array())

    # ======================================================================
    # test with different meshes
    # ======================================================================

    N = len(mis)
    meshes = [UnitSquare(i + 3, i + 3) for i in range(N)]
    fss = [FunctionSpace(mesh, "CG", 4) for mesh in meshes]
    F = [interpolate(Expression("*".join(["x[0]"] * (i + 1))), fss[i]) for i in range(N)]
    vecs = [FEniCSVector(f) for f in F]

    w = MultiVectorWithProjection()
    for mi, vec in zip(mis, vecs):
        w[mi] = vec
    v = A * w

    L = LegendrePolynomials(normalised=True)
    H = StochasticHermitePolynomials(mu=0.5, normalised=True)
    ex0 = Expression("2*x[0] + 3*(l01*x[0]*x[0]-l00*x[0]) + 4*(h01*x[0]*x[0]*x[0]-h00*x[0])",
        l01=L.get_beta(0)[1], l00=L.get_beta(0)[0],
        h01=H.get_beta(0)[1], h00=H.get_beta(0)[0])
    ex2 = Expression("2*x[0]*x[0]*x[0] + 4*(h11*x[0]*x[0]*x[0]*x[0] - h10*x[0]*x[0]*x[0] + h1m1*x[0])",
        h11=H.get_beta(1)[1], h10=H.get_beta(1)[0], h1m1=H.get_beta(1)[-1])
    vec0 = FEniCSVector(interpolate(ex0, fss[0]))
    vec2 = FEniCSVector(interpolate(ex2, fss[2]))

    assert_almost_equal(v[mis[0]].array(), vec0.array())
    assert_almost_equal(v[mis[2]].array(), vec2.array())


@skip_if(not HAVE_FENICS)
def test_fenics_with_assembly():
    a = [Expression('A*sin(pi*I*x[0]*x[1])', A=1, I=i, degree=2) for i in range(1, 4)]
    rvs = [UniformRV(), NormalRV(mu=0.5)]
    coeff_field = ListCoefficientField(a[0], a[1:], rvs)

    A = MultiOperator(coeff_field, FEMPoisson().assemble_operator)
    mis = [Multiindex([0]),
           Multiindex([1]),
           Multiindex([0, 1]),
           Multiindex([0, 2])]
    mesh = UnitSquare(4, 4)
    fs = FunctionSpace(mesh, "CG", 1)
    F = [interpolate(Expression("*".join(["x[0]"] * i)), fs) for i in range(1, 5)]
    vecs = [FEniCSVector(f) for f in F]

    w = MultiVectorWithProjection()
    for mi, vec in zip(mis, vecs):
        w[mi] = vec
    v = A * w

    #print '\n', v[mis[0]].array()
    #print w[mis[0]].array()

    #    L = LegendrePolynomials(normalised=True)
    #    H = StochasticHermitePolynomials(mu=0.5, normalised=True)
    #    ex0 = Expression("2*x[0] + 3*(l01*x[0]*x[0]-l00*x[0]) + 4*(h01*x[0]*x[0]*x[0]-h00*x[0])",
    #                     l01=L.get_beta(0)[1], l00=L.get_beta(0)[0],
    #                     h01=H.get_beta(0)[1], h00=H.get_beta(0)[0])
    #    vec0 = FEniCSVector(interpolate(ex0, fs))
    #
    #    assert_almost_equal(v[mis[0]].array(), vec0.array())


    # ======================================================================
    # test with different meshes
    # ======================================================================

    N = len(mis)
    meshes = [UnitSquare(i + 3, i + 3) for i in range(N)]
    fss = [FunctionSpace(mesh, "CG", 4) for mesh in meshes]
    F = [interpolate(Expression("*".join(["x[0]"] * (i + 1))), fss[i]) for i in range(N)]
    vecs = [FEniCSVector(f) for f in F]

    w = MultiVectorWithProjection()
    for mi, vec in zip(mis, vecs):
        w[mi] = vec
    v = A * w


test_main()
