from spuq.application.egsz.resiual_estimator import ResidualEstimator
from spuq.utils.testing import *

class TestResidualEstimator(TestCase):
    def test_estimator(self):


test_main()
    # define articifial coefficient field
    CF

    # set artificial initial solution
    wN

    # define source term
    f = FEniCSExpression("1.0", constant=True)
    
    # evaluate residual and projection error estimators
    RE = ResidualEstimator(CF, f, wN, 10, FEniCSBasis.PROJECTION.INTERPOLATION)
    res = RE.evaluateResidualEstimator()
    proj = RE.evaluateProjectionError()


#class ResidualEstimator(object):
#    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.
#    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
#    fenics/dolfin implementation is based on
#    https://answers.launchpad.net/dolfin/+question/177108
#    """
#    @takes(anything, CoefficientField, (FEniCSExpression, FEniCSFunction), MultiVector, optional(int))
#    def __init__(self, CF, f, wN, maxm=10, ptype=FEniCSBasis.PROJECTION.INTERPOLATION):
#    def evaluateResidualEstimator(self):
#    def evaluateProjectionError