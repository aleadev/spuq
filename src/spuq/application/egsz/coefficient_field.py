"""The expansion of a coefficient field."""

from spuq.utils.type_check import takes, anything, sequence_of
from spuq.linalg.function import GenericFunction
from spuq.stochastics.random_variable import RandomVariable, DeterministicPseudoRV
from spuq.utils import strclass

class CoefficientField(object):
    """expansion of a coefficient field according to EGSZ (1.2)"""

    @takes(anything, sequence_of(GenericFunction), sequence_of(RandomVariable))
    def __init__(self, funcs, rvs):
        """initialise with list of functions and list of random variables
        
        The first function is the mean field for which no random
        variable is required, i.e. len(funcs)=len(rvs)+1.
        DeterministicPseudoRV is associated with the mean field implicitly.

        Alternatively, just one random variable can be provided for
        all expansion coefficients with method createWithIidRVs.
        """
        assert len(funcs) == len(rvs) + 1, (
            "Need one more function than random variable (for the deterministic case)")
        self._funcs = list(funcs)
        # first function is deterministic mean field
        self._rvs = list(rvs)
        self._rvs.insert(0, DeterministicPseudoRV)

    @classmethod
    @takes(anything, sequence_of(GenericFunction), RandomVariable)
    def createWithIidRVs(cls, funcs, rv):
        # random variables are stateless, so we can just use the same n times 
        rvs = [rv] * (len(funcs) - 1)
        return cls(funcs, rvs)

    def coefficients(self):
        """return expansion iterator for (Function,RV) pairs"""
        for i in xrange(len(self._funcs)):
            yield self._funcs[i], self._rvs[i]

    def __getitem__(self, i):
        assert i < len(self._funcs), "invalid index"
        return self._funcs[i], self._rvs[i]

    def __repr__(self):
        return "<%s funcs=%s, rvs=%s>" % \
               (strclass(self.__class__), self._funcs[1:], self._rvs)

    def __len__(self):
        """Length of coefficient field expansion"""
        return len(self._funcs)
