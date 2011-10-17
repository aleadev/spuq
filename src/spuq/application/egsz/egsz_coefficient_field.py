"""expansion of a coefficient field"""

from spuq.utils.type_check import *
from spuq.linalg.function import GenericFunction
from spuq.stochastics.random_variable import RandomVariable

class CoefficientField(object):
    """expansion of a coefficient field according to EGSZ (1.2)"""
    
    @takes((list_of(GenericFunction),tuple_of(GenericFunction)), (list_of(RandomVariable)))
    def __init__(self, funcs, rvs):
        """initialise with list of functions and list of random variables
        
            The first function is the mean field for which no random variable is required, i.e. len(funcs)=len(rvs)+1.
            Alternatively, just one random variable can be provided for all expansion coefficients.
        """
        assert len(rvs) == 1 or len(funcs)-1 == len(rvs)
        self._funcs = funcs
        self._rvs = rvs
        
    def coefficients(self):
        """return expansion iterator for (Function,RV) pairs"""
        def coeff_iter(self):
            """expansion iterator"""
            assert bool(self._funcs) and bool(self._rvs)
            yield self._funcs[0], None                  # first function is deterministic mean field
            for i in xrange(len(self._funcs)-1):
                if len(self._rvs) == 1:
                    yield self._funcs[i+1], self._rvs
                else:
                    yield self._funcs[i+1], self._rvs[i]
                
        return coeff_iter

    def __getitem__(self, i):
        assert i < len(self._funcs), "invalid index"
        if i == 0:
            return self._funcs[0], None
        else:
            if len(self._rvs) == 1:
                return self._funcs[i], self._rvs
            else:
                return self._funcs[i], self._rvs[i-1]

    def __repr__(self):
        return "CoefficientField(funcs={0},rvs={1})".format(self._funcs,self._rvs)
