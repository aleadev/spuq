import numpy as np

from spuq.utils.testing import *
from spuq.stoch.distribution_base import *
from spuq.stoch.continuous_distributions import *


class TestDistributions(TestCase):

    class test_base(self):
        assert_equal( 1, 2 )


import unittest
class TestUniformDistribution(unittest.TestCase):
    def setUp(self):
        from uniform_distribution import UniformDistribution
        self.ud1=UniformDistribution()
        self.ud2=UniformDistribution(2, 6)
    def test_mean(self):
        self.assertEquals( self.ud1.mean(), 0 )
        self.assertEquals( self.ud2.mean(), 4 )
    def test_var(self):
        self.assertEquals( self.ud1.var(), 1.0/3.0 )
        self.assertEquals( self.ud2.var(), 4.0/3.0 )
    def test_pdf(self):
        from numpy import exp, sqrt, pi
        self.assertEquals( self.ud1.pdf(0.5), 0.5 )
        self.assertEquals( self.ud2.pdf(1.8), 0 )
        self.assertEquals( self.ud2.pdf(6.2), 0 )
    def test_cdf(self):
        from scipy import inf
        self.assertEquals( self.ud1.cdf(0),  0.5)
        #self.assertEquals( self.ud1.cdf(inf),  1)
        #self.assertEquals( self.ud1.cdf(-inf),  0)
        self.assertEquals( self.ud1.cdf(1000),  1)
        self.assertEquals( self.ud1.cdf(-1000),  0)
        self.assertEquals( self.ud2.cdf(3),  0.25)
    def test_sample(self):
        s=self.ud1.sample(5)
        self.assertEquals( s.shape,  (5, ) )
    def test_shift_scale(self):
        d=self.ud2
        self.assertEquals( d.shift(2).mean(), d.mean()+2  )
        self.assertEquals( d.shift(2).var(), d.var() )
        self.assertEquals( d.scale(2).mean(), d.mean()  )
        self.assertEquals( d.scale(2).var(), d.var()*4 )


import unittest
class TestNormalDistribution(unittest.TestCase):
    def setUp(self):
        self.nd1=NormalDistribution()
        self.nd2=NormalDistribution(2, 3)
    def test_mean(self):
        self.assertEquals( self.nd1.mean(), 0 )
        self.assertEquals( self.nd2.mean(), 2 )
    def test_var(self):
        self.assertEquals( self.nd1.var(), 1 )
        self.assertEquals( self.nd2.var(), 9 )
    def test_pdf(self):
        from numpy import exp, sqrt, pi
        self.assertEquals( self.nd1.pdf(2), exp(-2)/sqrt(2*pi) )
        self.assertEquals( self.nd2.pdf(5), exp(-1.0/2)/sqrt(18*pi) )
    def test_cdf(self):
        from scipy import inf
        self.assertEquals( self.nd1.cdf(0), 0.5)
        self.assertEquals( self.nd1.cdf(inf), 1)
        self.assertEquals( self.nd1.cdf(-inf), 0)
        self.assertEquals( self.nd2.cdf(2),  0.5)
        self.assertAlmostEquals( self.nd1.cdf(1)-self.nd1.cdf(-1), 0.682689492137)
        self.assertAlmostEquals( self.nd2.cdf(5)-self.nd2.cdf(-1), 0.682689492137)
    def test_sample(self):
        s=self.nd1.sample(5)
        self.assertEquals( s.shape,  (5, ) )
    def test_shift_scale(self):
        self.assertEquals( self.nd2.shift(2).mu, 4 )
        self.assertEquals( self.nd2.shift(2).sigma, 3 )
        self.assertEquals( self.nd2.scale(2).mu, 2 )
        self.assertEquals( self.nd2.scale(2).sigma, 6 )
