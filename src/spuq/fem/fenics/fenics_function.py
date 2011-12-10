"""FEniCS discrete function wrapper"""

from dolfin import Function, Expression, interpolate
from spuq.utils.type_check import *
from spuq.linalg.function import GenericFunction, SympyFunction

class FenicsFunction(GenericFunction):
    """Wrapper for discrete FEniCS Function"""
    
    @takes(string, Dfstr=optional((list_of(string),tuple_of(string))), FS=optional())
    def __init__(self, fstr, Dfstr=None, FS=None, dimin=2, dimout=1):
        """Initialise (discrete) function.
        
        In case some function space is provided, the discrete interpolation of the function (given as string with 'x[0]' and 'x[1]') is constructed.
        Otherwise, the analytical representation is kept.
        If the two derivatives of fstr are not passed in Dfstr, SympyFunction is used to analytically determine the gradient.
        """
        super().__init__(dimin, dimout)
        if Dfstr:
            self._exf = Expression(fstr)
            self._exDf = Expression(Dfstr)
        else:
            F = SympyFunction(fstr)
            self._exf = Expression(F.eval().replace('x','x[0]').replace('y','x[1]'))
            self._exDf = Expression(F.diff().replace('x','x[0]').replace('y','x[1]'))
        
        if FS:
            self._FS = FS
            self._f = interpolate(self._exf, self._FS)
            self._Df = interpolate(self._exDf, self._FS)

        
    def eval(self, x=None):
        """Function evaluation.
        
            Return evaluated function at x or function string if x in not set"""
        if x:
            if self._FS:
                return self._f(x)
            else:
                return self._exf(x)
        else:
            return self._exf

        
    def diff(self, x=None, order=1):
        """Return derivative.
        
            Return derivative at x or function string of derivative if x is not set"""
        assert order == 1, "only first derivative supported"
        if x:
            if self._FS:
                return self._Df(x)
            else:
                return self._exDf(x)
        else:
            return FenicsFunction(self._exDf)

        
    @property
    def discrete(self):
        return bool(self._FS)
