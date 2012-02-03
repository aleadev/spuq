"""The expansion of a coefficient field."""

from spuq.utils.type_check import *
from spuq.linalg.function import GenericFunction
from spuq.stochastics.random_variable import RandomVariable

class CoefficientField(object):
    """expansion of a coefficient field according to EGSZ (1.2)"""
    
#    @takes((list_of(FEniCSFunction),tuple_of(FEniCSFunction),list_of(FEniCSExpression),tuple_of(FEniCSExpression)), (list_of(RandomVariable),tuple_of(RandomVariable)))
    def __init__(self, funcs, rvs):
        """initialise with list of functions and list of random variables
        
        The first function is the mean field for which no random
        variable is required, i.e. len(funcs)=len(rvs)+1.

        Alternatively, just one random variable can be provided for
        all expansion coefficients.

        Usually, the functions should be wrapped FEniCS Expressions or
        Functions, i.e. FEniCSExpression or FEniCSFunction.
        """
        assert len(funcs)-1 == len(rvs), (
            "Need one more function than random variable (for the deterministic case)")
        self._funcs = list(funcs)
        # first function is deterministic mean field
        self._funcs.insert(0, None)
        self._rvs = rvs

    @classmethod
    def createWithIidRVs(cls, func, rv):
        rvs = [rv] * (len(func)-1)
        return cls(func, rv)
        
    def coefficients(self):
        """return expansion iterator for (Function,RV) pairs"""
        def coeff_iter(self):
            """expansion iterator"""
            assert bool(self._funcs) and bool(self._rvs)
            for i in xrange(len(self._funcs)):
                yield self._funcs[i], self._rvs[i]
                
        return coeff_iter

    def __getitem__(self, i):
        assert i < len(self._funcs), "invalid index"
        return self._funcs[i], self._rvs[i]

    def __repr__(self):
        return "CoefficientField(funcs={0},rvs={1})".format(self._funcs[1:],self._rvs)

    def __len__(self):
        """Length of coefficient field expansion"""
        return len(self._funcs)
