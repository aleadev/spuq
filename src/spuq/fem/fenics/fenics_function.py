"""FEniCS discrete function wrapper"""

from dolfin import FunctionSpace, VectorFunctionSpace, Function, Expression, interpolate, project, grad
from spuq.utils.type_check import *
from spuq.linalg.function import GenericFunction

class FEniCSExpression(GenericFunction):
    """Wrapper for FEniCS Expressions"""
    
    def __init__(self, fstr=None, fexpression=None, Dfstr=None, Dfexpression=None, domain_dim=2, codomain_dim=1):
        GenericFunction.__init__(self, domain_dim, codomain_dim)
        if fstr:
            self.f = Expression(fstr)
        else:
            assert fexpression
            self.f = fexpression
        if Dfstr:
            self.Df = Expression(Dfstr)
        elif Dfexpression:
            self.Df = Dfexpression
        else:
            self.Df = None
            
    def eval(self, *x):
        return self.f(*x)
        
    def diff(self):
        assert self.Df
        return FEniCSExpression(fexpression=self.Df)


class FEniCSFunction(GenericFunction):
    """Wrapper for discrete FEniCS function"""
    
#    @takes(any, fFS=FunctionSpace, DfFS=optional(FunctionSpace), fexpression=optional(Expression,FEniCSExpression), Dfexpression=optional(Expression,FEniCSExpression), fstr=optional(str), Dfstr=optional(list_of(str)), domain_dim=int, codomain_dim=int, domain=optional(list_of(int)), numericalDf=optional(bool))
    def __init__(self, fFS, DfFS=None, fexpression=None, Dfexpression=None, fstr=None, Dfstr=None,\
                    domain_dim=2, codomain_dim=1, domain=None, numericalDf=True):
        """Initialise (discrete) function.
        
        TODO: document functionality!
        """
        if not domain:
            domain = [(min([x[d] for x in fFS.mesh().coordinates()]),\
                        max([x[d] for x in fFS.mesh().coordinates()])) for d in range(fFS.mesh().topology().dim())]
        GenericFunction.__init__(self, domain_dim, codomain_dim, domain=domain)
        self.fFS = fFS
        self.DfFS = DfFS

        if not self.DfFS:
            # construct "natural" derivative space
            self.DfFS = VectorFunctionSpace(self.fFS.mesh(), self.fFS.ufl_element().family(),\
                                                self.fFS.ufl_element().degree())

        # prepare function expression
        if fexpression:
            if isinstance(fexpression, FEniCSExpression):
                self.fex = fexpression.f
            else:
                self.fex = fexpression
        else:
            self.fex = Expression(fstr)

        # prepare derivative expression
        if Dfexpression:
            if isinstance(Dfexpression, FEniCSExpression):
                self.Dfex = Dfexpression.f
            else:
                self.Dfex = Dfexpression
        else:
            if Dfstr:
                self.Dfex = Expression(Dfstr)
            elif fexpression.Df:
                self.Dfex = fexpression.Df
            else:
                self.Dfex = None

        # interpolate function and derivative on FunctionSpaces
        self.f = interpolate(self.fex, self.fFS)
        if self.Dfex:
            self.Df = interpolate(self.Dfex, self.DfFS)
        elif numericalDf:
            self.Df = project(grad(self.f), self.DfFS)

    def eval(self, *x):
        """Function evaluation.
        
            Return function evaluated at x"""
        return self.f(*x)


    def diff(self):
        """Return derivative.
        
            Return function of interpolated explicit or numerical derivative"""
        return self.Df
