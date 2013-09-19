import numpy as np
from spuq.linalg.tensor_basis import TensorBasis

from spuq.utils.testing import *
from spuq.linalg.vector import *
from spuq.linalg.cptensor import CPTensor
from spuq.linalg.basis import *
from spuq.linalg.test_support import *


def test_vector_is_abstract():
    assert_raises(TypeError, Vector)

def test_flatvec_init():
    X1 = np.random.random([4, 3])
    X2 = np.random.random([5, 3])
    cpt = CPTensor([X1, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    #assert_raises(Exception, CPTensor, [X1, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(6)]))
    assert_equal(cpt.order, 2)


def test_flatvec_as_array():
    X1 = np.random.random([4, 3])
    X2 = np.random.random([5, 3])
    cpt = CPTensor([X1, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))

    assert_equal(cpt.flatten().as_array(), np.dot(X1, X2.T))
    #TODO: (check for higher order, when implemented)


def test_flatvec_equals():
    #TODO: (postponed, check when implemented)
    X1 = np.random.random([4, 3])
    X2 = np.random.random([5, 3])
    cpt1 = CPTensor([X1, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cpt2 = CPTensor([X1*2, X2*0.5], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cpt3 = CPTensor([X1*2, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))

    # make sure both operators are overloaded
    assert_true(cpt1 == cpt2)
    assert_false(cpt1 != cpt2)
    assert_true(cpt1 != cpt3)
    assert_false(cpt1 == cpt3)

    # now test for (in)equality
    assert_equal(cpt1, cpt2)
    assert_not_equal(cpt1, cpt3)


def test_flatvec_copy():
    pass

def test_flatvec_neg():
    X1 = np.random.random([4, 3])
    X2 = np.random.random([5, 3])
    cpt = CPTensor([X1, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))

    v1 = X1[0,0]
    -cpt
    assert_equal(v1, X1[0,0])

    cptm = CPTensor([X1, -X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    assert_equal(-cpt, cptm)

def test_flatvec_add():
    X1a = np.random.random([4, 3])
    X2a = np.random.random([5, 3])
    X1b = np.random.random([4, 2])
    X2b = np.random.random([5, 2])
    cpta = CPTensor([X1a, X2a], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cptb = CPTensor([X1b, X2b], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cpts = CPTensor([np.hstack([X1a,X1b]), np.hstack([X2a,X2b])], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))

    assert_equal(cpta + cptb, cpts)


def test_flatvec_sub():
    X1a = np.random.random([4, 3])
    X2a = np.random.random([5, 3])
    X1b = np.random.random([4, 2])
    X2b = np.random.random([5, 2])
    cpta = CPTensor([X1a, X2a], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cptb = CPTensor([X1b, X2b], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cpts = CPTensor([np.hstack([X1a,X1b]), np.hstack([X2a,-X2b])], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))

    assert_equal(cpta - cptb, cpts)


def test_flatvec_mul():
    X1 = np.random.random([4, 3])
    X2 = np.random.random([5, 3])
    s1 = 2.3
    s2 = 3.7
    cpt = CPTensor([X1, X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))
    cptm = CPTensor([s1 * X1, s2 * X2], TensorBasis([CanonicalBasis(4), CanonicalBasis(5)]))

    assert_array_almost_equal(((s1 * s2) * cpt).flatten().as_array(), cptm.flatten().as_array())

def test_flatvec_repr():
    fv1 = FlatVector(np.array([1.0, 2, 3]))
    assert_equal(str(fv1),
                 "<FlatVector basis=<CanonicalBasis dim=3>, " +
                 "coeffs=[ 1.  2.  3.]>")


test_main(True)
