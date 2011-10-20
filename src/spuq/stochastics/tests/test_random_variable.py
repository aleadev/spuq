import numpy as np
from numpy import exp, sqrt, pi
from scipy import inf
import scipy

from spuq.utils.testing import *
from spuq.stochastics.random_variable import *
from spuq.polyquad.polynomials import *


def _test_rv_coherence(rv):
    m = rv.mean
    v = rv.var
    
    # make sure variance is positive
    assert_true(rv.var > 0)
    
    # make sure cdf and invcdf match
    assert_approx_equal(rv.invcdf(rv.cdf(m)), m)
    assert_approx_equal(rv.invcdf(rv.cdf(m + v)), m + v)
    assert_approx_equal(rv.invcdf(rv.cdf(m - v)), m - v)
    
    # make sure pdf and cdf match (using central difference
    # approximation, therefore check only 4 significant digits)
    eps = 1e-9
    for x in [m, m-v, m+v]:
        assert_approx_equal(rv.cdf(x + eps * v) - rv.cdf(x - eps * v), 
                            2 * eps * v * rv.pdf(x),
                            significant=4)

    # make sure median, cdf and invcdf match
    assert_approx_equal(rv.cdf(rv.median), 0.5)

    # make sure the orthogonal polynomials are truly orthonormal
    # w.r.t. to the measure induced by the random variable
    p = rv.orth_polys
    assert_almost_equal(rv.quad(p[2] * p[3]), 0)
    assert_almost_equal(rv.quad(p[1] * p[4]), 0)
    assert_almost_equal(rv.quad(p[1] * p[1]), 1)
    assert_almost_equal(rv.quad(p[5] * p[5]), 1)
    
    # make sure pdf and mean, var, skewness, and kurtosis match
    _1 = np.poly1d(1)
    x = (_1.integ()-rv.mean)/np.sqrt(rv.var)
    assert_almost_equal(rv.quad(_1), 1)
    assert_almost_equal(rv.quad(x), 0)
    assert_almost_equal(rv.quad(x**2), 1)
    assert_almost_equal(rv.quad(x**3), rv.skew)
    assert_almost_equal(rv.quad(x**4)-3, rv.kurtosis)

class TestUniformRV(TestCase):

    def setUp(self):
        self.ud1 = UniformRV()
        self.ud2 = UniformRV(2, 6)

    def test_repr(self):
        assert_equal(str(self.ud1), "<UniformRV a=-1.0 b=1.0>")
        assert_equal(str(self.ud2), "<UniformRV a=2.0 b=6.0>")

    def test_mean(self):
        assert_equal(self.ud1.mean, 0)
        assert_equal(self.ud2.mean, 4)

    def test_var(self):
        assert_equal(self.ud1.var, 1.0 / 3.0)
        assert_equal(self.ud2.var, 4.0 / 3.0)

    def test_pdf(self):
        assert_equal(self.ud1.pdf(0.5), 0.5)
        assert_equal(self.ud2.pdf(1.8), 0)
        assert_equal(self.ud2.pdf(6.2), 0)

    def test_cdf(self):
        assert_equal(self.ud1.cdf(0), 0.5)
        #assert_equal(self.ud1.cdf(inf), 1)
        #assert_equal(self.ud1.cdf(-inf), 0)
        assert_equal(self.ud1.cdf(1000), 1)
        assert_equal(self.ud1.cdf(-1000), 0)
        assert_equal(self.ud2.cdf(3), 0.25)

    def test_sample(self):
        s = self.ud1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        d = self.ud2
        assert_equal(d.shift(2).mean, d.mean + 2)
        assert_equal(d.shift(2).var, d.var)
        assert_equal(d.scale(2).mean, d.mean)
        assert_equal(d.scale(2).var, d.var * 4)

    def test_orth_polys(self):
        assert_is_instance(self.ud1.orth_polys, PolynomialFamily)

    def test_coherence(self):
        _test_rv_coherence(self.ud1)
        _test_rv_coherence(self.ud2)


class TestNormalRV(TestCase):

    def setUp(self):
        self.nd1 = NormalRV()
        self.nd2 = NormalRV(2, 3)

    def test_repr(self):
        assert_equal(str(self.nd1), "<NormalRV mu=0.0 sigma=1.0>")
        assert_equal(str(self.nd2), "<NormalRV mu=2.0 sigma=3.0>")

    def test_mean(self):
        assert_equal(self.nd1.mean, 0)
        assert_equal(self.nd2.mean, 2)

    def test_var(self):
        assert_equal(self.nd1.var, 1)
        assert_equal(self.nd2.var, 9)

    def test_pdf(self):
        assert_equal(self.nd1.pdf(2), exp(-2) / sqrt(2 * pi))
        assert_equal(self.nd2.pdf(5), exp(-0.5) / sqrt(18 * pi))

    def test_cdf(self):
        assert_equal(self.nd1.cdf(0), 0.5)
        assert_equal(self.nd1.cdf(inf), 1)
        assert_equal(self.nd1.cdf(-inf), 0)
        assert_equal(self.nd2.cdf(2),  0.5)
        assert_almost_equal(self.nd1.cdf(1) - self.nd1.cdf(-1), 0.682689492137)
        assert_almost_equal(self.nd2.cdf(5) - self.nd2.cdf(-1), 0.682689492137)

    def test_sample(self):
        s = self.nd1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        assert_equal(self.nd2.shift(2).mu, 4)
        assert_equal(self.nd2.shift(2).sigma, 3)
        assert_equal(self.nd2.scale(2).mu, 2)
        assert_equal(self.nd2.scale(2).sigma, 6)

    def test_coherence(self):
        _test_rv_coherence(self.nd1)
        _test_rv_coherence(self.nd2)


test_main()
