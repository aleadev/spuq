from spuq.utils.testing import *
from spuq.polyquad.quad_1d import *

class TestQuadRuleGauss(TestCase):

    def test_gauss(self):
        p,w = QuadRuleGauss().getPointsWeights(10)
        #print p,w
        assert_almost_equal(sum(w), 1)
