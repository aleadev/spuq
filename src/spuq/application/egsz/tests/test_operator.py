from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.fem_discretisation import FEMPoisson
from spuq.fem.multi_vector import MultiVector
from spuq.utils.multiindex_set import MultiindexSet
from spuq.stochastics.random_variable import NormalRV
from spuq.utils.testing import *

class TestOperator(TestCase):
    def test_multioperator(self):
        # instantiate discretisation
        FEM = FEMPoisson()

        # set initial solution and set of active indices Lambda
        # MultiindexSet
        m = 5
        p = 3
        mi = MultiindexSet.createCompleteOrderSet(m, p)
        # init multivector
        mesh = UnitSquare(5,5)
        degree = 1
        V = FunctionSpace(mesh, "CG", degree)
        f = Function(V)
        initvector = FEniCSVector(f)
        wN1 = MultiVector(mi, initvector)
        # init test coefficient field
        F = list()
        for i, j in enumerate(mi.arr.tolist()):
            ex1 = Expression('x[0]*x[0]+A*sin(10.*x[1])', A=i)
            Dex1 = Expression('2.*x[0]+A*10.sin(10.*x[1])', A=i)
            F.append(FEniCSExpression(fexpression=ex1, Dfexpression=Dex1))
        CF1 = CoefficientField(F, NormalRV())
        MO1 = MultiOperator(FEM, CF1, 3)
        uN1 = MO1.apply(wN1)

        # create test case with identical 5x5 meshes (order 1 spaces)
        M2 = SampleProblem.createMeshes("square",("uniform",(5,5)),mi.count)
        V2 = FEniCSVector.create(M2, "CG", degree)
        wN2 = MultiVector(mi, V2)
        CF2 = SampleProblem.createCF("EF", 10)
        MO2 = MultiOperator(FEM, CF2, 3)
        uN2 = MO2.apply(wN2)

        # create test case with random meshes (order 1 spaces)
        M3 = SampleProblem.createMeshes("square",("random",(5,10),(5,10)),mi.count)
        V3 = FEniCSVector.create(M3, "CG", degree)
        wN3 = MultiVector(mi, V3)
        CF3 = SampleProblem.createCF("monomial", 10)
        MO3 = MultiOperator(FEM, CF3, 3)
        uN3 = MO3.apply(wN3)

test_main()
