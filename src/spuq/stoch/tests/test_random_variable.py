import numpy as np
from numpy import exp, sqrt, pi
from scipy import inf

from spuq.utils.testing import *
from spuq.stoch.random_variable import *


class TestRandomVariables(TestCase):

    def test_base(self):
        assert_equal(1, 1)


class TestUniformRV(TestCase):

    def setUp(self):
        self.ud1 = UniformRV()
        self.ud2 = UniformRV(2, 6)

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


class TestNormalRV(TestCase):

    def setUp(self):
        self.nd1 = NormalRV()
        self.nd2 = NormalRV(2, 3)

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

if __name__ == "__main__":
    run_module_suite()
