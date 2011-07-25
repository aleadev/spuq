from numpy import linalg as la
from numpy import array, ndarray, zeros, ones, diag, arange, sqrt
from spuq.quadrature.quad_rule_1d import Quad_Rule_1d

class Quad_Rule_Gauss(Quad_Rule_1d):
    """Gauss quadrature rule"""
    
    @staticmethod
    def getPointsWeights(np):
        g = arange(1,10)/sqrt(4.*arange(1,10)**2-ones(9))
        p,V = la.eig(diag(g,k=1) + diag(g,k=-1))
        w = 2*V[0,:]**2
        # transform to [0,1]
        p = .5 * p + .5
        w = .5 * w
        return p,w

import unittest
class TestQuadRuleGauss(unittest.TestCase):
    def test_gauss(self):
        p,w = Quad_Rule_Gauss.getPointsWeights(10)
        print p,w
        self.assertAlmostEquals( sum(w), 1 )

if __name__ == "__main__":
    unittest.main()
    
