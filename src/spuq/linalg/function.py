"""generic function interface and a simple function class"""

from abc import ABCMeta, abstractmethod
from types import MethodType
from spuq.utils.decorators import copydocs
from spuq.utils.type_check import *

class GenericFunction(object):
    """generic function interface
    
        allowed operations with scalar values or generic functions are
        +,-,*,/,**
        <<            composition
        %             tensorisation"""

    __metaclass__ = ABCMeta

    def __init__(self, domain_dim=1, codomain_dim=1, factor=1):
        self.domain_dim = domain_dim
        self.codomain_dim = codomain_dim
        self.factor = factor

    def __call__(self, *x):
        if len(x)==1 and isinstance(x[0], GenericFunction):
            return self._compose(self, x[0])
        elif len(x)==1 and isinstance(x[0], (tuple, list)):
            assert len(x[0])==self.domain_dim
            return self.eval(*x[0])
        else:
            assert len(x)==self.domain_dim
            return self.eval(*x)

    @abstractmethod
    def eval(self, *x):
        """function evaluation
           return evaluated function at x or function string if x in not set"""
        return NotImplemented
        
    @property
    def D(self):
        "return derivative Df = f.D() = f.diff()"
        if not hasattr(self,"_diff"):
            self._diff = self.diff()
        return self._diff

    def diff(self):
        """return derivative GenericFunction"""
        return NotImplemented
        
    def __add__(self, g):
        return _add(self, g)

    def __sub__(self, g):
        return _add(self, g, sign=-1)

    def __mul__(self, g):
        return _mul(self, g)

    def __div__(self, g):
        return _mul(self, g, dodiv=True)

    def __pow__(self, g):
        return _pow(self, g)
        
    def __mod__(self, g):
        return _tensorise(self, g)
        
    def __lshift__(self, g):
        return _compose(self, g)


def _add(f, g, sign=1):
    class AddedFunction(GenericFunction):
        def __init__(self, f, g):
            assert f.domain_dim == g.domain_dim
            assert f.codomain_dim == g.codomain_dim
            GenericFunction.__init__(self, f.domain_dim, f.codomain_dim)
            self.f = f
            self.g = g
    def _fadd(self, *x):
        return self.f(*x) + self.g(*x)
    def _Dfadd(self):
        return self.f.diff() + self.g.diff()
    def _fsub(self, *x):
        return self.f(*x) + self.g(*x)
    def _Dfsub(self):
        return self.f.diff() + self.g.diff()

    AF = AddedFunction(f, g)
    assert abs(sign)==1
    if sign == 1:
        AF.eval = MethodType(_fadd, AF, AddedFunction)
        AF.diff = MethodType(_Dfadd, AF, AddedFunction)
    else:
        AF.eval = MethodType(_fsub, AF, AddedFunction)
        AF.diff = MethodType(_Dfsub, AF, AddedFunction)
    return AF
    
def _mul(f, g, dodiv=False):
    class MultipliedFunction(GenericFunction):
        def __init__(self, f, g):
            assert f.domain_dim == g.domain_dim
            assert f.codomain_dim == g.codomain_dim
            GenericFunction.__init__(self, f.domain_dim, f.codomain_dim)
            self.f = f
            self.g = g
    def _fdiv(self, *x):
        return self.f(*x)/self.g(*x)
    def _Dfdiv(self):
        return (self.f.diff()*self.g + self.f*self.g.diff())/(self.g**2)
    def _fmul(self, *x):
        return self.f(*x)*self.g(*x)
    def _Dfmul(self):
        return self.f.diff()*self.g + self.f*self.g.diff()
    
    MF = MultipliedFunction(f, g)
    if dodiv:
        MF.eval = MethodType(_fdiv, MF, MultipliedFunction)
        MF.diff = MethodType(_Dfdiv, MF, MultipliedFunction)
    else:
        MF.eval = MethodType(_fmul, MF, MultipliedFunction)
        MF.diff = MethodType(_Dfmul, MF, MultipliedFunction)
    return MF        

def _pow(f, g):
    class PowerFunction(GenericFunction):
        def __init__(self, f, g):
            assert isinstance(g, (int,float)) or f.codomain_dim == g.codomain_dim
            GenericFunction.__init__(self, f.domain_dim, f.codomain_dim)
            self.f = f
            self.g = g
    def _powconst(self, *x):
        return self.f(*x)**self.g
    def _Dpowconst(self, *x):
        return self
    def _powfunc(self, *x):
        return self.f(x[:self.f.domain_dim])**self.g(x[-self.g.domain_dim:])            
    def _Dpowfunc(self, *x):
        return self.g.diff()*self
    
    PF = PowerFunction(f, g)
    if isinstance(g, (int,float)):
        PF.eval = MethodType(_powconst, PF, PowerFunction)
        PF.diff = MethodType(_Dpowconst, PF, PowerFunction)
    else:
        PF.eval = MethodType(_powfunc, PF, PowerFunction)
        PF.diff = MethodType(_Dpowfunc, PF, PowerFunction)
    return PF

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
            return self.f.D(self.g)*self.g.D
        
    return ComposedFunction(f,g)

def _tensorise(f, g):
    class TensorisedFunction(GenericFunction):
        def __init__(self, f, g):
            assert f.codomain_dim == g.codomain_dim
            GenericFunction.__init__(self, f.domain_dim+g.domain_dim, 
                                     f.codomain_dim)
            self.f = f
            self.g = g
        def eval(self, *x):
            x1 = x[:self.f.domain_dim]
            x2 = x[self.f.domain_dim:]
            return self.f(*x1)*self.g(*x2)
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
                                  codomain_dim=self.codomain_dim*self.domain_dim)
