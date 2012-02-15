"""FEniCS discrete function wrapper"""

from dolfin import FunctionSpace, VectorFunctionSpace, Function, Expression, Constant, interpolate, project, grad
from spuq.utils.type_check import *
from spuq.linalg.function import GenericFunction
from spuq.fem.fenics.fenics_basis import FEniCSBasis


class FEniCSGenericFunction(GenericFunction):
    def __init__(self, domain_dim, codomain_dim):
        GenericFunction.__init__(domain_dim, codomain_dim)


class FEniCSExpression(FEniCSGenericFunction):
    """Wrapper for FEniCS Expressions"""

    @takes(anything, optional(int), optional(int), optional(bool))
    def __init__(self, domain_dim = 2, codomain_dim = 1, constant = False):
        FEniCSGenericFunction.__init__(self, domain_dim, codomain_dim)
        self.fexpr = None
        self.Dfexpr = None

    @classmethod
    @takes(anything, str, optional(sequence_of(str)))
    def from_string(cls, fstr, Dfstr, domain_dim = 2, codomain_dim = 1, constant = False):
        FEx = cls(domain_dim, codomain_dim, constant)
        if not constant:
            FEx.fexpr = Expression(fstr)
            if Dfstr:
                FEx.Dfexpr = Expression(Dfstr)
        else:
            FEx.fexpr = Constant(fstr)
            if Dfstr:
                FEx.Dfexpr = Constant(Dfstr)
            else:
                FEx.Dfexpr = Constant(tuple(["0.0"] * codomain_dim))

    @classmethod
    @takes(anything, Expression, optional(Expression))
    def from_expression(cls, fexpr, Dfexpr = None, domain_dim = 2, codomain_dim = 1, constant = False):
        FEx = cls(domain_dim, codomain_dim, constant)
        FEx.fexpr = fexpr
        FEx.Dfexpr = Dfexpr

    def eval(self, *x):
        return self.fexpr(*x)

    def diff(self):
        assert self.Dfexpr
        return FEniCSExpression.from_expression(fexpression = self.Dfexpr, None, self.domain_dim, self.domain_dim)


class FEniCSFunction(FEniCSGenericFunction):
    """Wrapper for discrete FEniCS function (i.e. a Function instance) and optionally its gradient function.

        Initialised with either FEniCS Expressions or Functions."""

    @takes(anything, optional(int), optional(int))
    def __init__(self, domain_dim=2, codomain_dim=1):
        """Initialise (discrete) function.
        
        Initialisation can be done with some Function, Expression or string. If the first derivatives are needed, the user should also provide the analytical form when available. Otherwise, the numerical gradient is evaluated on the VectorFunctionSpace of the function. Furthermore, dimension and codimension should be specified.
        Usage:
            # from Function
            mesh = UnitSquare(5,5)
            V = FunctionSpace(mesh,'CG',1)
            F = Function(V)
            ...
            f = FEniCSFunction.from_function(F)
            # from Exression
            ex1 = FEniCSExpression.from_string("x[0]*x[1]", ("x[1]","x[0]"))
            f1 = FEniCSFunction(ex1)
            # from string
            f2 = FEniCSFunction.from_string("x[0]*sin(10.*x[1])", ("sin(10.*x[1])","10.*x[0]*cos(10.*x[1])"))
        """
        FEniCSGenericFunction.__init__(self, domain_dim, codomain_dim)
        self.f = None
        self.Df = None

        @classmethod
        @takes(anything, Function, optional(Function), optional(bool))
        def from_function(cls, function, Dfunction=None, numericalDf=True):
            F = cls(domain_dim, codomain_dim)
            F.f = function
            F.fFS = function.function_space()
        if Dfunction:
            F.Df = Dfunction
            F.DfFS = Dfunction.function_space()
        else:
            F.Dfunction = None
            F.DfFS = None

        if not Dfunction and numericalDf:
            # construct "natural" gradient space
            F.DfFS = VectorFunctionSpace(self.fFS.mesh(), self.fFS.ufl_element().family(),
                                                self.fFS.ufl_element().degree())

            # project function derivative on VectorFunctionSpace
            F.Df = project(grad(self.f), self.DfFS)

#        if not domain and self.fFS:
#            # determine domain from mesh coordinates
#            domain = [(min([x[d] for x in self.fFS.mesh().coordinates()]), \
#                       max([x[d] for x in self.fFS.mesh().coordinates()]))\
#                      for d in range(self.fFS.mesh().topology().dim())]


    @classmethod
    @takes(anything, FunctionSpace, (Expression,FEniCSExpression), optional(Expression))
    def from_expression(cls, FS, fexpr, Dfexpr):
        if isinstance(fexpr, FEniCSExpression):
            Dfexpr = fexpr.Dfexpr
            fexpr = fexpr.fexpr
        f = Function(fexpr)
        if Dfexpr:
            DfFS = VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())
            Df = interpolate(Dfexpr, DfFS)
        return cls.from_function(f, Df)

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
        if not hasattr(self, 'fFS'):
            self.fFS = self.f.function_space()
        return self.fFS

    @property
    def basis(self):
        """Return FEniCSBasis"""
        if not hasattr(self, 'fbasis'):
            self.fbasis = FEniCSBasis(functionspace = self.functionspace)
        return self.fbasis

    @property
    def coeffs(self):
        """Return fenics coefficient vector (assignable with numpy arrays)"""
        return self.f.vector()
