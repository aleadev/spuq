from __future__ import division
import numpy as np
from functools import partial

from spuq.utils.testing import *
from spuq.math_utils.multiindex import Multiindex
from spuq.stochastics.random_variable import NormalRV
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.linalg.vector import FlatVector
from spuq.linalg.operator import DiagonalMatrixOperator
from spuq.linalg.function import ConstFunction, SimpleFunction
from spuq.stochastics.random_variable import NormalRV, UniformRV

def fem1d_assemble(func, basis):
    """setup 1d FEM stiffness matrix with uniformly distributed nodes in [0,1] and piecewise constant coefficient (evaluated at element centers)"""
    N = basis.dim
    h = 1/(N-1)
    c = [func(x) for x in np.linspace(h/2,1-h/2,N-1)]
    A = np.diag(np.hstack((c, 0)) + np.hstack((0, c))) - np.diag(c*np.ones(N-1), -1) - np.diag(c*np.ones(N-1), 1)
    A *= 1/h 
    return A

def fem1d_interpolate(func, N):
    x = np.linspace(0, 1, N)
    Ifunc = np.array([func(y) for y in x])
    return Ifunc

def diag_assemble(func, basis):
    diag = np.array([np.abs(func(x)) for x in np.linspace(0, 1, basis.dim)])
    return DiagonalMatrixOperator(diag)

def test_init():
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    A = MultiOperator(coeff_field, diag_assemble)
    assert_raises(TypeError, diag_assemble, 3)
    assert_raises(TypeError, coeff_field, 7)

def test_apply():
    N = 4
    a = [ConstFunction(1.0), SimpleFunction(np.sin), SimpleFunction(np.cos)]
    rvs = [UniformRV(), NormalRV()]
    coeff_field = CoefficientField(a, rvs)

    A = MultiOperator(coeff_field, diag_assemble)
    vec1 = FlatVector(np.random.random(N))
    vec2 = FlatVector(np.random.random(N))
    
    mis = Multiindex.createCompleteOrderSet(2, 3)
    print mis[0]
    print mis[1]
    print mis[2]
    print mis[3]
    w = MultiVectorWithProjection()
    w[mis[0]] = vec1
    w[mis[1]] = vec2
    w[mis[2]] = vec1
    w[mis[3]] = vec2
    print w
    v = A*w
    for i in range(4):
        print w[mis[i]]
        print v[mis[i]]

    #A*mv

def test_apply_fem1d():
    class DummyBase(object):
        def __init__(self, dim):
            self.dim = dim
    N = 4
    b = DummyBase(N)
    A = fem1d_assemble(ConstFunction(1.0), b)
    B = np.diag(np.hstack((1, 2*np.ones(N-2), 1))) - np.diag(np.ones(N-1), -1) - np.diag(np.ones(N-1), 1)
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
    w = MultiVectorWithProjection(project = partial(fem1d_interpolate, N=N))
    w[mis[0]] = vec1
    w[mis[1]] = vec2
    w[mis[2]] = vec1
    w[mis[3]] = vec2
    print w
    v = A*w
    for i in range(4):
        print w[mis[i]]
        print v[mis[i]]


test_main()
