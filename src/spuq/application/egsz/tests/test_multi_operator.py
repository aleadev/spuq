from __future__ import division
import numpy as np
from functools import partial

from spuq.utils.testing import *
from spuq.math_utils.multiindex import Multiindex
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.linalg.basis import Basis, CanonicalBasis
from spuq.linalg.vector import FlatVector
from spuq.linalg.operator import DiagonalMatrixOperator
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.polyquad.polynomials import LegendrePolynomials, StochasticHermitePolynomials

def fem1d_assemble(func, basis):
    """setup 1d FEM stiffness matrix with uniformly distributed nodes in [0,1] and piecewise constant coefficient (evaluated at element centers)"""
    N = basis.dim
    h = 1 / (N - 1)
    c = [func(x) for x in np.linspace(h / 2, 1 - h / 2, N - 1)]
    A = np.diag(np.hstack((c, 0)) + np.hstack((0, c))) - np.diag(c * np.ones(N - 1), -1) - np.diag(c * np.ones(N - 1), 1)
    A *= 1 / h
    return A

def fem1d_interpolate(func, N):
    x = np.linspace(0, 1, N)
    Ifunc = np.array([func(y) for y in x])
    return Ifunc

class SimpleProjectBasis(CanonicalBasis):
    def project_onto(self, vec):
        assert self.dim == vec.basis.dim
        return vec.copy()

def diag_assemble(func, basis):
    diag = np.array([np.abs(func(x)) for x in np.linspace(0, 1, basis.dim)])
    return DiagonalMatrixOperator(diag, domain=basis, codomain=basis)

def test_init():
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    MultiOperator(coeff_field, diag_assemble)
    assert_raises(TypeError, MultiOperator, 3, diag_assemble)
    assert_raises(TypeError, MultiOperator, coeff_field, 7)

    domain = CanonicalBasis(3)
    codomain = CanonicalBasis(5)
    A = MultiOperator(coeff_field, diag_assemble, domain, codomain)
    assert_equal(A.domain, domain)
    assert_equal(A.codomain, codomain)


def test_apply():
    N = 4
    #a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    a = [ConstFunction(2.0), ConstFunction(3.0), ConstFunction(4.0)]
    rvs = [UniformRV(), NormalRV(mu=0.5)]
    coeff_field = CoefficientField(a, rvs)

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

@skip_if(True)
def test_apply_fem1d():
    class DummyBase(object):
        def __init__(self, dim):
            self.dim = dim
    N = 4
    b = DummyBase(N)
    A = fem1d_assemble(ConstFunction(1.0), b)
    h = 1.0 / (N - 1)
    B = 1.0 / h * (np.diag(np.hstack((1, 2 * np.ones(N - 2), 1))) - np.diag(np.ones(N - 1), -1) - np.diag(np.ones(N - 1), 1))
    assert_equal(A, B)

    f1 = lambda x: x
    f2 = lambda x: 2 - x
    a = [ConstFunction(1.0), SimpleFunction(f1), SimpleFunction(f2)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    A = MultiOperator(coeff_field, fem1d_assemble)
    vec1 = FlatVector(fem1d_interpolate(f1, N))
    vec2 = FlatVector(fem1d_interpolate(f1, N))

    mis = Multiindex.createCompleteOrderSet(2, 3)
    w = MultiVectorWithProjection(project=partial(fem1d_interpolate, N=N))
    w[mis[0]] = vec1
    w[mis[1]] = vec2
    w[mis[2]] = vec1
    w[mis[3]] = vec2
    print w
    v = A * w
    for i in range(4):
        print w[mis[i]]
        print v[mis[i]]


test_main()
