"""FEniCS discrete function wrapper"""

from dolfin import VectorFunctionSpace, Function, Expression, Constant, interpolate, project, grad
from spuq.utils.type_check import takes, optional, list_of
from spuq.linalg.function import GenericFunction
from spuq.fem.fenics.fenics_basis import FEniCSBasis

class FEniCSExpression(GenericFunction):
    """Wrapper for FEniCS Expressions"""

    def __init__(self, fstr=None, fexpression=None, Dfstr=None, Dfexpression=None, domain_dim=2, codomain_dim=1, constant=False):
        GenericFunction.__init__(self, domain_dim, codomain_dim)
        if fstr:
            if not constant:
                self.fex = Expression(fstr)
            else:
                self.fex = Constant(fstr)
        else:
            assert fexpression
            self.fex = fexpression
        if Dfstr or constant:
            if not constant:
                self.Dfex = Expression(Dfstr)
            else:
                if Dfstr:
                    self.Dfex = Constant(Dfstr)
                else:
                    self.Dfex = Constant("0.")
        elif Dfexpression:
            self.Dfex = Dfexpression
        else:
            self.Dfex = None

    def eval(self, *x):
        return self.fex(*x)

    def diff(self):
        assert self.Dfex
        return FEniCSExpression(fexpression=self.Dfex)


class FEniCSFunction(GenericFunction):
    """Wrapper for discrete FEniCS function (i.e. a Function instance) and optionally its gradient function.

        Initialised with either fenics Expressions or Functions."""

#    @takes(anything, function=Function, Dfunction=optional(Function), fexpression=optional(Expression,FEniCSExpression), Dfexpression=optional(Expression,FEniCSExpression), fstr=optional(str), Dfstr=optional(list_of(str)), domain_dim=int, codomain_dim=int, domain=optional(list_of(int)), numericalDf=optional(bool))
    def __init__(self, function=None, Dfunction=None, fexpression=None, Dfexpression=None, fstr=None, Dfstr=None, \
                    domain_dim=2, codomain_dim=1, domain=None, numericalDf=True):
        """Initialise (discrete) function.
        
        Initialisation can be done with some Function, Expression or string. If the first derivatives are needed, the user should also provide the analytical form when available. Otherwise, the numerical gradient is evaluated on the VectorFunctionSpace of the function. Furthermore, dimension and codimension should be specified.
        Usage:
            # from Function
            mesh = UnitSquare(5,5)
            V = FunctionSpace(mesh,'CG',1)
            F = Function(V)
            ...
            f = FEniCSFunction(function=F)
            # from Exression
            ex1 = Expression("x[0]*x[1]")
            ex2 = FEniCSExpression(ex1)
            f1 = FEniCSFunction(fexpression=ex1)
            f2 = FEniCSFunction(fexpression=ex2)
            # from string
            f3 = FEniCSFunction(fstr="x[0]*sin(10.*x[1])")
        """
        if function:
            assert isinstance(function, Function)
            self.f = function
            self.fFS = function.function_space()
        else:
            self.function = None
            self.fFS = None
        if Dfunction:
            assert isinstance(Dfunction, Function)
            self.Df = Dfunction
            self.DfFS = Dfunction.function_space()
        else:
            self.Dfunction = None
            self.DfFS = None

        if not domain and self.fFS:
            # determine domain from mesh coordinates
            domain = [(min([x[d] for x in self.fFS.mesh().coordinates()]), \
                       max([x[d] for x in self.fFS.mesh().coordinates()]))\
                      for d in range(self.fFS.mesh().topology().dim())]

        GenericFunction.__init__(self, domain_dim, codomain_dim, domain=domain)

        if not self.DfFS and self.fFS:
            # construct "natural" gradient space
            self.DfFS = VectorFunctionSpace(self.fFS.mesh(), self.fFS.ufl_element().family(), \
                                                self.fFS.ufl_element().degree())

        # prepare function expression
        if not function:
            if fexpression:
                if isinstance(fexpression, FEniCSExpression):
                    self.fex = fexpression.fex
                else:
                    self.fex = fexpression
            else:
                assert fstr
                self.fex = Expression(fstr)

        # prepare derivative expression
        if not Dfunction:
            if Dfexpression:
                if isinstance(Dfexpression, FEniCSExpression):
                    self.Dfex = Dfexpression.Dfex
                else:
                    self.Dfex = Dfexpression
            else:
                if Dfstr:
                    self.Dfex = Expression(Dfstr)
                elif isinstance(fexpression, FEniCSExpression) and fexpression.Dfex:
                    self.Dfex = fexpression.Dfex
                else:
                    self.Dfex = None

        # interpolate function and derivative on FunctionSpaces
        if not function:
            self.f = interpolate(self.fex, self.fFS)
        if not Dfunction:
            if self.Dfex:
                self.Df = interpolate(self.Dfex, self.DfFS)
            elif numericalDf:
                self.Df = project(grad(self.f), self.DfFS)

    def eval(self, *x):
        """Function evaluation.
        
            Return function evaluated at x"""
        return self.f(*x)


    def diff(self):
        """Return derivative as FEniCSFunction.
        
            Return function of interpolated explicit or numerical derivative"""
        return self.Df

    def function_space(self):
        """Return (fenics) FunctionSpace for Function."""
        return self.fFS

    def array(self):
        """Return numpy array (copy) of coefficients"""
        return self.f.vector().array()

    @property
    def functionspace(self):
        """Return fenics FunctionSpace"""
        return self.fFS

    @property
    def basis(self):
        """Return FEniCSBasis"""
        if not hasattr(self, 'fbasis'):
            self.fbasis = FEniCSBasis(functionspace=self.functionspace)
        return self.fbasis

    @property
    def coeffs(self):
        """Return fenics coefficient vector (assignable with numpy arrays)"""
        return self.f.vector()
