"""generic function interface and sympy function wrapper"""

from abc import ABCMeta, abstractproperty, abstractmethod
from spuq.utils.type_check import *

__all__ = ["GenericFunction", "SimpleFunction"]

class GenericFunction(object):
    """generic function interface"""

    __metaclass__ = ABCMeta

    def __init__(self, domain_dim=1, codomain_dim=1):
        self.domain_dim = domain_dim
        self.codomain_dim = codomain_dim


    def __call__(self, *x):
        if len(x)==1 and isinstance(x[0], GenericFunction):
            return _compose(self, x[0])
        elif len(x)==1 and isinstance(x[0], (tuple, list)):
            assert len(x[0])==self.domain_dim
            return self.eval(*x[0])
        else:
            assert len(x)==self.domain_dim
            return self.eval(*x)

    
    def __pow__(self, g):
        return  _tensorise(self, g)

    @abstractmethod
    def eval(self, *x):
        """function evaluation
        
            return evaluated function at x or function string if x in not set"""
        return NotImplemented
        
    @property
    def D(self):
        "df = f.D()"
        if not hasattr(self,"_diff"):
            self._diff = self.diff()
        return self._diff

    def diff(self):
        """return derivative
        return GenericFunction()"""
        return NotImplemented

def _compose(f, g):
    class ComposedFunction(GenericFunction):
        def __init__(self, f, g, **kwargs):
            GenericFunction.__init__(self, g.domain_dim, f.codomain_dim, 
                                     **kwargs)
            assert g.codomain_dim==f.domain_dim
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
            GenericFunction.__init__(self, f.domain_dim+g.domain_dim, 
                                     f.codomain_dim)
            assert f.codomain_dim==g.codomain_dim
            self.f = f
            self.g = g
        def eval(self, *x):
            x1 = x[:self.f.domain_dim]
            x2 = x[self.f.domain_dim:]
            return self.f(*x1)*self.g(*x2)
        def diff(self):
            return NotImplemented
    return TensorisedFunction(f, g)


class SimpleFunction(GenericFunction):
    def __init__(self, f, Df=None, domain_dim=1, codomain_dim=1):
        GenericFunction.__init__(self, domain_dim=domain_dim, 
                                 codomain_dim=codomain_dim)
        self._f = f
        self._Df = Df
    def eval(self, *x):
        return self._f(*x)
    def diff(self):
        return SimpleFunction(Df, 
                              domain_dim=self.domain_dim, 
                              codomain_dim=self.codomain_dim*self.domain_dim)

f = SimpleFunction(f=lambda x: x**2)
g = SimpleFunction(f=lambda x: 2*x)
print f(3)
print f((3))
print f(f)(3)
print f(f(f))(3)

print f(g)(3), 36
print g(f)(3), 18

print (f**g)(5,7), 350
print (f**g)(7,5), 490
h1=f**g
h2=g**f
print (h1**h2)(5,7,5,7), 350*490


#f = GenericFunction(domain_dim=3)
#f(1,2,3)
#f((1,2,3))
#f(f)

#f = GenericFunction()
