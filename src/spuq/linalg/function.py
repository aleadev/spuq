"""generic function interface and sympy function wrapper"""

from abc import ABCMeta, abstractproperty, abstractmethod
from spuq.utils.decorators import copydocs
from spuq.utils.type_check import *

class GenericFunction(object):
    """generic function interface"""

    __metaclass__ = ABCMeta

    def __init__(self, domain_dim=1, codomain_dim=1):
        self.domain_dim = domain_dim
        self.codomain_dim = codomain_dim

    def compose(self, g):
        class ComposedFunction(GenericFunction):
            def eval(self, *x):
                return self.f(self.g(x))
            def diff(self):
                return self.f.D(self.g)*self.g.D
        c=ComposedFunction()
        c.f=self
        c.g=g
        return c

    def tensorise(self, g):
        class TensorisedFunction(GenericFunction):
            def eval(self, *x):
                return self.f(*x)*self.g(*x))
            def diff(self):
                return self.f.D(self.g)*self.g.D
        c=ComposedFunction()
        c.f=self
        c.g=g
        return c

    def add(self, g):
        class ComposedFunction(GenericFunction):
            def eval(self, *x):
                return self.f(self.g(*x))
            def diff(self):
                return self.f.D(self.g)*self.g.D
        c=ComposedFunction()
        c.f=self
        c.g=g
        return c

    def __call__(self, *x):
        if len(x)==1 and isinstance(x[0], GenericFunction):
            return self.compose(x[0])
        elif len(x)==1 and isinstance(x, (tuple, list)):
            return self.eval(*x[0])
        else:
            return self.eval(*x)
        

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


# f, g #1->1
# f(x), g(y)
# h=tensorize(f,g)
# h((x,y)) = f(x)*g(y)
# h(x,y) = f(x) g(y)
# k = tensorise(h,h)
# k = h**h
# k(x,y,z,w)
# f+g
# (f*g)(x)=f(x)*g(x)
# f(g)
# h+




class SympyFunction2D(GenericFunction):
    """wrapper for sympy function"""
    
    @takes(str)
    def __init__(self, s, dimin=2, dimout=1):
        assert dimin==2, 'dimension <> 2 not supported yet'
        super().__init__(dimin, dimout)
        
        from sympy import *
        x, y, z = symbols('x y')
        self.f = eval(s.replace('x[0]','x').replace('x[1]','y'))
            
    def eval(self, x=None):
        if x:
            return self.f.subs().evalf()
        else:
            return self.str() 
        
    def diff(self, x=None, order=1):
        if x:
            return self.f.diff(x, order)
        else:
            return str(diff(self.f)) 
        
    def __repr__(self):
        return 'SympyFunction(\"'+str(self.f)+','+str(self.dimin)+','+str(self.dimout)+')'
    
    def __str__(self):
        return str(self.f)
    
