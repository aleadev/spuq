import numpy as np

from spuq.utils.testing import *
from spuq.stochastics.multiindex import *


class TestMultiindex(TestCase):
    
    def test_init(self):
        # make sure we can create multiindices from integer arrays and
        # list, but not from floats
        Multiindex([0, 1, 3, 0])
        Multiindex(np.array([0, 1, 3, 0]))
        assert_raises(TypeError, Multiindex, [0, 1.3, 3, 0])
        assert_raises(TypeError, Multiindex, np.array([0, 1.3, 3, 0]))
        # make sure the result is normalised
        alpha = Multiindex(np.array([0, 1, 3, 0]))
        assert_array_equal(alpha.as_array, np.array([0, 1, 3]))
        
    def test_len(self):
        alpha = Multiindex(np.array([0, 1, 3, 0]))
        assert_equal(len(alpha), 3)
        assert_equal(alpha.len(), 3)
        alpha = Multiindex(np.array([0, 1, 3, 0, 4, 0, 0]))
        assert_equal(len(alpha), 5)

    def test_equality(self):
        alpha = Multiindex(np.array([0, 1, 3, 0]))
        beta = Multiindex(np.array([0, 1, 3, 0, 0, 0]))
        gamma = Multiindex(np.array([0, 1, 3, 0, 4, 0, 0]))
        
        assert_true(alpha == beta)
        assert_true(alpha != gamma)
        assert_true(not(alpha != beta))
        assert_true(not(alpha == gamma))

    def test_hash(self):
        alpha = Multiindex(np.array([0, 1, 3, 0]))
        beta = Multiindex(np.array([0, 1, 3, 0, 0, 0]))
        
        assert_equal(hash(alpha), hash(beta))
        
    def test_order(self):
        alpha = Multiindex(np.array([0, 1, 3, 0]))
        beta = Multiindex(np.array([0, 1, 3, 0, 4, 0]))
        assert_equal(alpha.order(), 4)
        assert_equal(beta.order(), 8)
        
    def test_inc_dec(self):
        alpha_orig = Multiindex(np.array([0, 1, 3, 0]))
        alpha = Multiindex(np.array([0, 1, 3, 0]))
        beta = Multiindex(np.array([0, 1, 3, 0, 7, 0]))
        
        assert_equal(alpha.inc(4, 7), beta)
        assert_equal(alpha, alpha_orig)
        assert_equal(beta.dec(4, 7), alpha)
        assert_equal(alpha.dec(0, 1), None)
        assert_equal(alpha.dec(8, 1), None)
        
