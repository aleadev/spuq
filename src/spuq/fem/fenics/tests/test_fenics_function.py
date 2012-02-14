from spuq.utils.testing import *


try:
    from dolfin import UnitSquare, FunctionSpace
    from spuq.fem.fenics.fenics_function import *
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    HAVE_FENICS = True
except:
    HAVE_FENICS = False


@skip_if(not HAVE_FENICS, "FEniCS not installed.")
def test_function():
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
    f2 = FEniCSFunction(V1, fexpression=ex2)
    Df0 = f0.diff()
    Df1 = f1.diff()
    Df2 = f2.diff()

    # test function values
    posval = (((0,0),1,(1,1)),((1,1),1,(2,2)))
    for p in posval:
        pos = p[0]
        valf = p[1]
        valDf = p[2]
        print pos, valf, valDf
        print f0(pos)
        print f1(pos)
        print f2(pos)
        print Df0(pos)
        print Df1(pos)
        print Df2(pos)


def test_vector():
    pass


test_main()
