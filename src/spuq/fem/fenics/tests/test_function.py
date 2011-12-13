from dolfin import UnitSquare, FunctionSpace
from spuq.utils.testing import *
from spuq.fem.fenics.fenics_function import *

class TestFEniCSFunction(TestCase):
    def test_function(self):
        # setup mesh [0,1]^2 and scalar P1 FEM space
        degree = 1
        mesh1 = UnitSquare(10,10)
        V1 = FunctionSpace(mesh1, "CG", degree)

        # test Expression wrapper
        ex1 = FEniCSExpression("x[0]*x[0]+x[1]")
        ex2 = FEniCSExpression("x[0]*x[0]+x[1]", ("2.*x[0]","1."))

        # test Function wrapper
        f0 = FEniCSFunction(V1, fstr="x[0]*x[0]+x[1]", Dfstr=("2.*x[0]","1."))
        f1 = FEniCSFunction(V1, fexpression=ex1)
        f1 = FEniCSFunction(V1, fexpression=ex2)

        # test function values
        # TODO

test_main()
