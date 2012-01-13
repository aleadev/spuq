from spuq.application.egsz.residual_estimator import ResidualEstimator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.fem.fenics.fenics_function import FEniCSExpression
from spuq.fem.multi_vector import MultiVector
from spuq.utils.multiindex_set import MultiindexSet
from spuq.stochastics.random_variable import NormalRV
from spuq.utils.testing import *

from dolfin import Expression, UnitSquare, FunctionSpace

class TestResidualEstimator(TestCase):
    def test_estimator(self):
        # MultiindexSet
        m = 2
        p = 3
        mi = MultiindexSet.createCompleteOrderSet(m, p)

        # define articifial coefficient field
        F = list()
        for i, j in enumerate(mi.arr.tolist()):
            ex1 = Expression('x[0]*x[0]+A*sin(10.*x[1])', A=i)
            Dex1 = Expression('2.*x[0]+A*10.*sin(10.*x[1])', A=i)
            F.append(FEniCSExpression(fexpression=ex1, Dfexpression=Dex1))
        CF = CoefficientField(F, (NormalRV(),))

        # set zero initial solution
        # init MultiVector
        mesh = UnitSquare(5,5)
        degree = 1
        V = FunctionSpace(mesh, "CG", degree)
        initvector = FEniCSVector(basis=V)
        wN = MultiVector(mi, initvector)

        # define source term
        f = FEniCSExpression("1.0", constant=True)

        # evaluate residual and projection error estimators
        RE = ResidualEstimator(CF, f, wN, 10, FEniCSBasis.PROJECTION.INTERPOLATION)
        res = RE.evaluateResidualEstimator()
        proj = RE.evaluateProjectionError()
        print mesh
        print res.shape, proj.shape
        print res, shape

test_main()
