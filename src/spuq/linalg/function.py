"""generic function interface, a simple function class and a const function class"""

from __future__ import division
from numpy import ones, infty
from abc import ABCMeta, abstractmethod
from spuq.utils.decorators import copydocs
from spuq.utils.type_check import *

class GenericFunction(object):
    """generic function interface
    
        allowed operations with scalar values or generic functions are
        +,-,*,/,**
        <<            composition
        %             tensorisation"""

    __metaclass__ = ABCMeta

    def __init__(self, domain_dim=1, codomain_dim=1, factor=1, domain=(-infty, infty)):
        self.domain_dim = domain_dim
        self.codomain_dim = codomain_dim
        self.factor = factor
        self._domain = [domain for _ in range(self.domain_dim)]

    def __call__(self, *x):
        if len(x) == 1 and isinstance(x[0], GenericFunction):
            return _compose(self, x[0])
        elif len(x) == 1 and isinstance(x[0], (tuple, list)):
            assert len(x[0]) == self.domain_dim
            return self.eval(*x[0])
        else:
            assert len(x) == self.domain_dim
            return self.eval(*x)

    @abstractmethod
    def eval(self, *x):
        """function evaluation
           return evaluated function at x or function string if x in not set"""
        return NotImplemented

    @property
    def D(self):
        "return derivative Df = f.D() = f.diff()"
        if not hasattr(self, "_diff"):
            self._diff = self.diff()
        return self._diff

    def diff(self):
        """return derivative GenericFunction"""
        return NotImplemented

    @property
    def domain(self):
        return self._domain

    def __add__(self, g):
        return _BinaryFunction(self, g, "add")

    def __radd__(self, g):
        return _BinaryFunction(g, self, "add")

    def __sub__(self, g):
        return _BinaryFunction(self, g, "sub")

    def __rsub__(self, g):
        return _BinaryFunction(g, self, "sub")

    def __pos__(self):
        return self

    def __neg__(self):
        return _BinaryFunction(0, self, "sub")

    def __mul__(self, g):
        return _BinaryFunction(self, g, "mul")

    def __rmul__(self, g):
        return _BinaryFunction(g, self, "mul")

    #def __div__(self, g):
    #    return _BinaryFunction(self, g, "div")
    def __truediv__(self, g):
        return _BinaryFunction(self, g, "div")

    #def __rdiv__(self, g):
    #    return _BinaryFunction(g, self, "div")
    def __rtruediv__(self, g):
        return _BinaryFunction(g, self, "div")

    def __pow__(self, g):
        return _BinaryFunction(self, g, "pow")

    def __rpow__(self, g):
        return _BinaryFunction(g, self, "pow")

    def __mod__(self, g):
        return _tensorise(self, g)

    def __lshift__(self, g):
        return _compose(self, g)


def _make_const_func(c, f):
    """Makes a ConstantFunction from c matching the domain and codomain of f"""
    return ConstFunction(c, f.domain_dim, f.codomain_dim)


def _make_matching_funcs(f, g):
    """Makes a ConstantFunction from c matching f, if c is not a GenericFunction; otherwise returns c unchanged"""
    if isinstance(f, GenericFunction):
        if isinstance(g, GenericFunction):
            return (f, g)
        else:
            return (f, _make_const_func(g, f))
    else:
        assert isinstance(g, GenericFunction)
        return (_make_const_func(f, g), g)


class _BinaryFunction(GenericFunction):
    def __init__(self, f, g, op):
        (f, g) = _make_matching_funcs(f, g)
        assert f.domain_dim == g.domain_dim
        assert f.codomain_dim == g.codomain_dim
        GenericFunction.__init__(self, f.domain_dim, f.codomain_dim)
        self.f = f
        self.g = g
        self.def_eval(op)

    def eval(self, *x):
        return self._eval(*x)

    def _fadd(self, *x):
        return self.f(*x) + self.g(*x)
    def _Dfadd(self):
        return self.f.diff() + self.g.diff()

    def _fsub(self, *x):
        return self.f(*x) - self.g(*x)
    def _Dfsub(self):
        return self.f.diff() - self.g.diff()

    def _fmul(self, *x):
        return self.f(*x) * self.g(*x)
    def _Dfmul(self):
        return self.f.diff() * self.g + self.f * self.g.diff()

    def _fdiv(self, *x):
        return self.f(*x) / (self.g(*x))
    def _Dfdiv(self):
        return (self.f.diff() * self.g - self.f * self.g.diff()) / (self.g ** 2)

    def _fpow(self, *x):
        return self.f(*x) ** self.g(*x)
    def _Dfpow(self, *x):
        return self._Dpowconst() + self._Dconstpow()
    def _Dfpowconst(self):
        return self.g * self.f.D * (self.f ** (self.g - 1))
    def _Dfconstpow(self):
        # log of function not yet defined
        assert False
        #return self.g.D * log(self.f) * (self.f ** self.g)

    def def_eval(self, op):
        if op == "add":
            self._eval = self._fadd
            self.diff = self._Dfadd
        elif op == "sub":
            self._eval = self._fsub
            self.diff = self._Dfsub
        elif op == "mul":
            self._eval = self._fmul
            self.diff = self._Dfmul
        elif op == "div":
            self._eval = self._fdiv
            self.diff = self._Dfdiv
        elif op == "pow":
            self._eval = self._fpow
            if isinstance(self.f, ConstFunction):
                self.diff = self._Dfconstpow
            elif isinstance(self.g, ConstFunction):
                self.diff = self._Dfpowconst
            else:
                self.diff = self._Dfpow
        else:
            assert False


def _compose(f, g):
    class ComposedFunction(GenericFunction):
        def __init__(self, f, g):
            assert g.codomain_dim == f.domain_dim
            GenericFunction.__init__(self, g.domain_dim, f.codomain_dim)
            self.f = f
            self.g = g
        def eval(self, *x):
            return self.f(self.g(x))
        def diff(self):
            return self.f.D(self.g) * self.g.D

    return ComposedFunction(f, g)

def _tensorise(f, g):
    class TensorisedFunction(GenericFunction):
        def __init__(self, f, g):
            assert f.codomain_dim == g.codomain_dim
            GenericFunction.__init__(self, f.domain_dim + g.domain_dim,
                                     f.codomain_dim)
            self.f = f
            self.g = g
        def eval(self, *x):
            x1 = x[:self.f.domain_dim]
            x2 = x[self.f.domain_dim:]
            return self.f(*x1) * self.g(*x2)
        def diff(self):
            return TensorisedFunction(self.f.diff(), self.g.diff())

    return TensorisedFunction(f, g)


class SimpleFunction(GenericFunction):
    """python function interface"""

    def __init__(self, f, Df=None, domain_dim=1, codomain_dim=1):
        GenericFunction.__init__(self, domain_dim=domain_dim,
                                 codomain_dim=codomain_dim)
        self._f = f
        self._Df = Df

    def eval(self, *x):
        return self._f(*x)

    def diff(self):
        assert self._Df != None
        return SimpleFunction(self._Df, domain_dim=self.domain_dim,
                                  codomain_dim=self.codomain_dim * self.domain_dim)


class ConstFunction(GenericFunction):
    def __init__(self, const=1, domain_dim=1, codomain_dim=1):
        GenericFunction.__init__(self, domain_dim=domain_dim, codomain_dim=codomain_dim)
        self.const = const

    def eval(self, *x):
        if self.codomain_dim > 1:
            return self.const * ones((self.codomain_dim, 1))
        else:
            return self.const

    def diff(self):
        return ConstFunction(const=0, domain_dim=self.domain_dim, codomain_dim=self.domain_dim * self.codomain_dim)
