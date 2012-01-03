from __future__ import division

from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.fem.fenics.fenics_function import FEniCSExpression, FEniCSFunction
from spuq.fem.multi_vector import MultiVector
from spuq.stochastics.random_variable import NormalRV

from dolfin import UnitSquare, UnitCircle
from exceptions import TypeError
from itertools import repeat
from numpy import array


class SampleProblem(object):
    @classmethod
    def createMeshes(cls, domain, params):
        assert domain=="square"
        if params[0]=="uniform":
            M = [m for m in repeat(UnitSquare(params[1][0],params[1][1])), params[2]]
        elif params[0]=="random":
            r = array(map(int, 10*rand(params[2]*2)))
            r.shape = params[2], 2
            size1 = params[1][0]
            size2 = params[1][1]
            r[:,0] *= (size2[0]-size1[0])/10
            r[:,1] *= (size2[1]-size1[1])/10
            r[:,0] += size1[0]
            r[:,1] += size1[1]
            M = [UnitSquare(r[i,0],r[i,1]) for i in range(params[2])]
        else:
            raise TypeError
        return M

    @classmethod
    def createCF(cls, cftype, cfsize):
        def tuplegen():
            for m in count():
                for n in range(0,m+1):
                    yield (m,n)

        if cf[0]=="EF":
            f = lambda a,b: Expression("sin(A*pi*x[0])*sin(B*pi*x[1])",A=a,B=b)
            Df = lambda a,b: Expression("A*B*sin(A*pi*x[0])*sin(B*pi*x[1])",A=a,B=b)
        elif cf[0]=="monomials":
            f = lambda a,b: Expression("*".join(["x[0]" for _ in range(a)])+"+"\
                                        +"*".join(["x[1]" for _ in range(b)]))
            Df = lambda a,b: Expression(str(a)+"*"+"*".join(["x[0]" for _ in range(a)])+"+"\
                                       +str(b)+"+"+"*".join(["x[1]" for _ in range(b)]))
        else:
            raise TypeError('unsupported function type')

        F = [FEniCSExpression(fexpression=f(a,b), Dfexpression=Df(a,b)) for a,b in restrict(tuplegen(), cfsize)]
        RV = NormalRV()
        return CoefficientField(F, RV)
