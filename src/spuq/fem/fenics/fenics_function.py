"""FEniCS discrete function wrapper"""

from dolfin import FunctionSpace, Function, Expression, interpolate, grad
from spuq.utils.type_check import *
from spuq.linalg.function import GenericFunction

class FEniCSExpression(GenericFunction):
    """Wrapper for FEniCS Expressions"""
    def __init__(self, fstr=None, fexpression=None, Dfstr=None, domain_dim=1, codomain_dim=1):
        GenericFunction.__init__(domain_dim, codomain_dim)
        if fstr:
            self.f = Expression(fstr)
        else:
            assert fexpression
            self.f = fexpression
        if Dfstr:
            self.Df = Expression(Dfstr)
            
    def eval(self, *x):
        return self.f(*x)
        
    def diff(self):
        return FEniCSExpression(fexpression=self.f)
        
    @takes(any, FunctionSpace)
    def discretise(self, V):
        return FEniCSFunction(self.f, self.Df)


class FEniCSFunction(GenericFunction):
    """Wrapper for discrete FEniCS function"""
    
    @takes(any, fexpression=optional(Expression), Dfexpression=optional(Expression), fstr=optional(str), Dfstr=optional((list_of(str),tuple_of(str))))
    def __init__(self, fexpression=None, Dfexpression=None, fstr=None, Dfstr=None, FS=None, domain_dim=1, codomain_dim=1):
        """Initialise (discrete) function.
        
        In case some function space is provided, the discrete interpolation of the function (given as string with 'x[0]' and 'x[1]') is constructed.
        Otherwise, the analytical representation is kept.
        If the two derivatives of fstr are not passed in Dfstr, SympyFunction is used to analytically determine the gradient.
        """
        GenericFunction.__init__(domain_dim, codomain_dim)
        if fexpression:
            self.f = fexpression
        elif fstr:
            self.f = Expression(fstr)
             
        if Dfexpression:
            self.Df = Dfexpression
        elif Dfstr:
            self.Df = Expression(Dfstr)
        
    def eval(self, *x):
        """Function evaluation.
        
            Return evaluated function at x"""
        return self.f(*x)

        
    def diff(self):
        """Return derivative.
        
            Return derivative at x or function string of derivative if x is not set"""
        return self.Df
