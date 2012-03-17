import numpy as np

from spuq.linalg.basis import CanonicalBasis
from spuq.linalg.vector import FlatVector
from spuq.utils.testing import assert_equal, assert_almost_equal

class FooBasis(CanonicalBasis):
    """Dummy basis class for testing that operator and vector methods 
    correctly respect bases."""
    pass

class BarBasis(CanonicalBasis):
    """Dummy basis class for testing that operator methods correctly
    respect bases."""
    pass

class FooVector(FlatVector):
    """Dummy vector class for testing that operator methods correctly
    respect vector classes."""
    pass

def assert_vector_almost_equal(vec1, vec2):
    assert_equal(type(vec1), type(vec2))
    assert_almost_equal(vec1.coeffs, vec2.coeffs)
    #assert_equal(vec1, vec2)

def assert_operator_is_consistent(op):
    vec = FooVector(np.random.random(op.domain.dim),
                    op.domain)

    res = op * vec
    assert_equal(res.basis, op.codomain)
    assert_equal(type(res), type(vec))

    assert_vector_almost_equal((3.0 * op) * vec, 3.0 * res)
    assert_vector_almost_equal((op * 3.0) * vec, 3.0 * res)

    if hasattr(op, "inverse") and op.domain.dim == op.codomain.dim:
        inv = op.inverse()
        assert_vector_almost_equal(vec, inv * res)
        assert_equal(inv.domain, op.codomain)
        assert_equal(inv.codomain, op.domain)

        iinv = inv.inverse()
        assert_vector_almost_equal(res, iinv * vec)
        assert_equal(inv.domain, op.domain)
        assert_equal(inv.codomain, op.codomain)

    if hasattr(op, "transpose"):
        trans = op.transpose()
        #assert_equal(vec, trans * res)
        assert_equal(trans.domain, op.codomain)
        assert_equal(trans.codomain, op.domain)

        ttrans = trans.transpose()
        assert_vector_almost_equal(res, ttrans * vec)
        assert_equal(ttrans.domain, op.domain)
        assert_equal(ttrans.codomain, op.codomain)

