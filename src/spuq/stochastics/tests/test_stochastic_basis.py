import numpy as np

from spuq.utils.testing import *

from spuq.stochastics.stochastic_basis import *
from spuq.math_utils.multiindex_set import MultiindexSet, createCompleteOrderSet
from spuq.stochastics.random_variable import *



class TestPolynomialBasis(TestCase):

    def test_foo(self):
        pass


class TestMultiindexBasis(TestCase):

    def test_init(self):
        I = createCompleteOrderSet(2, 4)
        #m = MultiindexBasis( I, [1, 2, 3])

class TestGPCBasis(TestCase):

    def test_init(self):
        rv = UniformRV()
        b = GPCBasis(rv, 4)

        assert_equal(b.rv, rv)
        assert_equal(b.degree, 4)
        assert_equal(b.dim, 5)

    def test_sample(self):
        rv = UniformRV()
        b = GPCBasis(rv, 4)

        x = b.sample(7)
        assert_equal(x.shape, (5, 7))

