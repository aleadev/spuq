"""generic function interface and sympy function wrapper"""

from abc import ABCMeta, abstractproperty, abstractmethod
from spuq.utils.decorators import copydocs
from spuq.utils.type_check import *

class GenericFunction(object):
    """generic function interface"""

    __metaclass__ = ABCMeta

    def __init__(self, dimin=2, dimout=1):
        self.dimin = dimin
        self.dimout = dimout

    def __call__(self, x):
        return self.eval(x)

    @abstractmethod    
    def eval(self, x=None):
        """function evaluation
        
            return evaluated function at x or function string if x in not set"""
        return NotImplemented
        
    @abstractmethod
    def diff(self, x=None, order=1):
        """return derivative
        
            return derivative at x or function string of derivative if x is not set"""
        return NotImplemented


class SympyFunction(GenericFunction):
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
            return self._f.subs().evalf()
        else:
            return self.str() 
        
    def diff(self, x=None, order=1):
        if x:
            return self._f.diff(x, order)
        else:
            return str(diff(self.f)) 
        
    def __repr__(self):
        return 'SympyFunction(\"'+str(self.f)+','+str(self.dimin)+','+str(self.dimout)+')'
    
    def __str__(self):
        return str(self.f)
    