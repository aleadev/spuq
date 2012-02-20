import numpy as np
from numpy import exp, sqrt, pi
from scipy import inf
import scipy

from spuq.utils.testing import *
from spuq.stochastics.random_variable import *
from spuq.polyquad.polynomials import *


def _test_rv_coherence(rv):
    m = rv.mean
    s = 0.1 * scipy.sqrt(rv.var)

    # make sure variance is positive
    assert_true(rv.var > 0)

    # make sure cdf and invcdf match
    assert_approx_equal(rv.invcdf(rv.cdf(m)), m)
    assert_approx_equal(rv.invcdf(rv.cdf(m + s)), m + s)
    assert_approx_equal(rv.invcdf(rv.cdf(m - s)), m - s)

    # make sure pdf and cdf match (using central difference
    # approximation, therefore check only 4 significant digits)
    eps = 1e-9
    for x in [m, m - s, m + s]:
        assert_approx_equal(rv.cdf(x + eps * s) - rv.cdf(x - eps * s),
                            2 * eps * s * rv.pdf(x),
                            significant=4)

    # make sure median, cdf and invcdf match
    assert_approx_equal(rv.cdf(rv.median), 0.5)

    # make sure pdf and mean, var, skewness, and kurtosis match
    _1 = np.poly1d(1)
    x = (_1.integ() - rv.mean) / np.sqrt(rv.var)
    kwargs = {"decimal": 4}
    assert_almost_equal(rv.integrate(_1), 1, **kwargs)
    assert_almost_equal(rv.integrate(x), 0, **kwargs)
    assert_almost_equal(rv.integrate(x ** 2), 1, **kwargs)
    assert_almost_equal(rv.integrate(x ** 3), rv.skew, **kwargs)
    assert_almost_equal(rv.integrate(x ** 4) - 3, rv.kurtosis, **kwargs)

    # make sure the orthogonal polynomials are truly orthonormal
    # w.r.t. to the measure induced by the random variable
    p = rv.orth_polys
    assert_almost_equal(rv.integrate(p[2] * p[3]), 0, **kwargs)
    assert_almost_equal(rv.integrate(p[1] * p[4]), 0, **kwargs)
    assert_almost_equal(rv.integrate(p[1] * p[1]), 1, **kwargs)
    assert_almost_equal(rv.integrate(p[5] * p[5]), 1, **kwargs)


class TestUniformRV(TestCase):

    def setUp(self):
        self.rv1 = UniformRV()
        self.rv2 = UniformRV(2, 6)

    def test_repr(self):
        assert_equal(str(self.rv1), "<UniformRV a=-1.0 b=1.0>")
        assert_equal(str(self.rv2), "<UniformRV a=2.0 b=6.0>")

    def test_mean(self):
        assert_equal(self.rv1.mean, 0)
        assert_equal(self.rv2.mean, 4)

    def test_var(self):
        assert_equal(self.rv1.var, 1.0 / 3.0)
        assert_equal(self.rv2.var, 4.0 / 3.0)

    def test_pdf(self):
        assert_equal(self.rv1.pdf(0.5), 0.5)
        assert_equal(self.rv2.pdf(1.8), 0)
        assert_equal(self.rv2.pdf(6.2), 0)

    def test_cdf(self):
        assert_equal(self.rv1.cdf(0), 0.5)
        #assert_equal(self.rv1.cdf(inf), 1)
        #assert_equal(self.rv1.cdf(-inf), 0)
        assert_equal(self.rv1.cdf(1000), 1)
        assert_equal(self.rv1.cdf(-1000), 0)
        assert_equal(self.rv2.cdf(3), 0.25)

    def test_sample(self):
        s = self.rv1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        d = self.rv2
        assert_equal(d.shift(2).mean, d.mean + 2)
        assert_equal(d.shift(2).var, d.var)
        assert_equal(d.scale(2).mean, d.mean)
        assert_equal(d.scale(2).var, d.var * 4)

    def test_orth_polys(self):
        assert_is_instance(self.rv1.orth_polys, PolynomialFamily)

    def test_coherence(self):
        _test_rv_coherence(self.rv1)
        _test_rv_coherence(self.rv2)


class TestNormalRV(TestCase):

    def setUp(self):
        self.rv1 = NormalRV()
        self.rv2 = NormalRV(2, 3)

    def test_repr(self):
        assert_equal(str(self.rv1), "<NormalRV mu=0.0 sigma=1.0>")
        assert_equal(str(self.rv2), "<NormalRV mu=2.0 sigma=3.0>")

    def test_mean(self):
        assert_equal(self.rv1.mean, 0)
        assert_equal(self.rv2.mean, 2)

    def test_var(self):
        assert_equal(self.rv1.var, 1)
        assert_equal(self.rv2.var, 9)

    def test_pdf(self):
        assert_equal(self.rv1.pdf(2), exp(-2) / sqrt(2 * pi))
        assert_equal(self.rv2.pdf(5), exp(-0.5) / sqrt(18 * pi))

    def test_cdf(self):
        assert_equal(self.rv1.cdf(0), 0.5)
        assert_equal(self.rv1.cdf(inf), 1)
        assert_equal(self.rv1.cdf(-inf), 0)
        assert_equal(self.rv2.cdf(2), 0.5)
        assert_approx_equal(self.rv1.cdf(1) - self.rv1.cdf(-1), 0.682689492137)
        assert_approx_equal(self.rv2.cdf(5) - self.rv2.cdf(-1), 0.682689492137)

    def test_sample(self):
        s = self.rv1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        assert_equal(self.rv2.shift(2).mu, 4)
        assert_equal(self.rv2.shift(2).sigma, 3)
        assert_equal(self.rv2.scale(2).mu, 2)
        assert_equal(self.rv2.scale(2).sigma, 6)

    def test_coherence(self):
        _test_rv_coherence(self.rv1)
        _test_rv_coherence(self.rv2)

class TestBetaRV(TestCase):

    def setUp(self):
        self.rv1 = BetaRV(alpha=0.5, beta=1.5)
        self.rv2 = BetaRV(alpha=1, beta=1, a=2, b=6)

    def test_init(self):
        assert_raises(TypeError, BetaRV, -0.5, 0.5)

    def test_repr(self):
        assert_equal(str(self.rv1), "<BetaRV alpha=0.5 beta=1.5 a=0.0 b=1.0>")
        assert_equal(str(self.rv2), "<BetaRV alpha=1.0 beta=1.0 a=2.0 b=6.0>")

    def test_mean(self):
        assert_equal(self.rv1.mean, 0.25)
        assert_equal(self.rv2.mean, 4)

    def test_var(self):
        assert_equal(self.rv1.var, 0.75 / 12.0)
        assert_equal(self.rv2.var, 4.0 / 3.0)

    def test_pdf(self):
        assert_equal(self.rv1.pdf(1), 0)
        assert_equal(self.rv2.pdf(1.8), 0)
        assert_equal(self.rv2.pdf(6.2), 0)

    def test_cdf(self):
        assert_equal(self.rv1.cdf(1), 1)
        assert_equal(self.rv1.cdf(1000), 1)
        assert_equal(self.rv1.cdf(-1000), 0)
        assert_equal(self.rv2.cdf(4), 0.5)

    def test_sample(self):
        s = self.rv1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        d = self.rv2
        assert_equal(d.shift(2).mean, d.mean + 2)
        assert_equal(d.shift(2).var, d.var)
        assert_equal(d.scale(2).mean, d.mean)
        assert_equal(d.scale(2).var, d.var * 4)

    def test_orth_polys(self):
        assert_is_instance(self.rv1.orth_polys, PolynomialFamily)

    def test_cmp_uniform(self):
        rvb = BetaRV(alpha=1, beta=1, a=2, b=6)
        rvu = UniformRV(a=2, b=6)
        assert_approx_equal(rvb.cdf(2.0), rvu.cdf(2.0))
        assert_approx_equal(rvb.cdf(2.5), rvu.cdf(2.5))
        assert_approx_equal(rvb.cdf(3.0), rvu.cdf(3.0))
        assert_approx_equal(rvb.cdf(4.0), rvu.cdf(4.0))
        p = rvb.orth_polys
        q = rvu.orth_polys
        assert_array_almost_equal(p[0], q[0])
        assert_array_almost_equal(p[1], q[1])
        assert_array_almost_equal(p[2], q[2])
        assert_array_almost_equal(p[3], q[3])
        assert_array_almost_equal(p[4], q[4])

    def test_coherence(self):
        rv = BetaRV(alpha=1.5, beta=1.5, a=3, b=6)
        _test_rv_coherence(rv)
        rv = BetaRV(alpha=2.5, beta=2.5, a=3, b=6)
        _test_rv_coherence(rv)
        rv = BetaRV(alpha=0.6, beta=0.6, a=3, b=6)
        _test_rv_coherence(rv)
        rv = BetaRV(alpha=0.5, beta=0.5, a=3, b=6)
        _test_rv_coherence(rv)
        _test_rv_coherence(self.rv1)
        _test_rv_coherence(self.rv2)


class TestSemicircularRV(TestCase):

    def setUp(self):
        self.rv1 = SemicircularRV()
        self.rv2 = SemicircularRV(2, 6)

    def test_repr(self):
        assert_equal(str(self.rv1), "<SemicircularRV a=-1.0 b=1.0>")
        assert_equal(str(self.rv2), "<SemicircularRV a=2.0 b=6.0>")

    def test_mean(self):
        assert_equal(self.rv1.mean, 0)
        assert_equal(self.rv2.mean, 4)

    def test_var(self):
        assert_equal(self.rv1.var, 1.0 / 4.0)
        assert_equal(self.rv2.var, 4.0 / 4.0)

    def test_pdf(self):
        assert_equal(self.rv1.pdf(0), 2.0 / scipy.pi)
        assert_approx_equal(self.rv1.pdf(scipy.sqrt(0.5)), scipy.sqrt(2.0) / scipy.pi)
        assert_equal(self.rv2.pdf(1.99), 0)
        assert_equal(self.rv2.pdf(6.01), 0)

    def test_cdf(self):
        assert_equal(self.rv1.cdf(0), 0.5)
        assert_equal(self.rv1.cdf(1000), 1)
        assert_equal(self.rv1.cdf(-1000), 0)
        assert_equal(self.rv2.cdf(4), 0.5)

    def test_sample(self):
        s = self.rv1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        d = self.rv2
        assert_equal(d.shift(2).mean, d.mean + 2)
        assert_equal(d.shift(2).var, d.var)
        assert_equal(d.scale(2).mean, d.mean)
        assert_equal(d.scale(2).var, d.var * 4)

    def test_orth_polys(self):
        assert_is_instance(self.rv1.orth_polys, PolynomialFamily)

    def test_coherence(self):
        _test_rv_coherence(self.rv1)
        _test_rv_coherence(self.rv2)


class TestArcsineRV(TestCase):

    def setUp(self):
        self.rv1 = ArcsineRV()
        self.rv2 = ArcsineRV(2, 6)

    def test_repr(self):
        assert_equal(str(self.rv1), "<ArcsineRV a=0.0 b=1.0>")
        assert_equal(str(self.rv2), "<ArcsineRV a=2.0 b=6.0>")

    def test_mean(self):
        assert_equal(self.rv1.mean, 0.5)
        assert_equal(self.rv2.mean, 4)

    def test_var(self):
        assert_equal(self.rv1.var, 1.0 / 8.0)
        assert_equal(self.rv2.var, 16.0 / 8.0)

    def test_pdf(self):
        assert_almost_equal(self.rv1.pdf(1 / 4.0), 4.0 / scipy.pi / scipy.sqrt(3.0))
        assert_almost_equal(self.rv1.pdf(1 / 3.0), 3.0 / scipy.pi / scipy.sqrt(2.0))
        assert_almost_equal(self.rv1.pdf(2 / 3.0), 3.0 / scipy.pi / scipy.sqrt(2.0))
        assert_equal(self.rv2.pdf(1.99), 0)
        assert_equal(self.rv2.pdf(6.01), 0)

    def test_cdf(self):
        assert_almost_equal(self.rv1.cdf(0.5), 0.5)
        assert_equal(self.rv1.cdf(1000), 1)
        assert_equal(self.rv1.cdf(-1000), 0)
        assert_almost_equal(self.rv2.cdf(4), 0.5)

    def test_sample(self):
        s = self.rv1.sample(5)
        assert_equal(s.shape, (5,))

    def test_shift_scale(self):
        d = self.rv2
        assert_equal(d.shift(2).mean, d.mean + 2)
        assert_equal(d.shift(2).var, d.var)
        assert_equal(d.scale(2).mean, d.mean)
        assert_equal(d.scale(2).var, d.var * 4)

    def test_orth_polys(self):
        assert_is_instance(self.rv1.orth_polys, PolynomialFamily)

    def test_coherence(self):
        _test_rv_coherence(self.rv1)
        _test_rv_coherence(self.rv2)


test_main()
