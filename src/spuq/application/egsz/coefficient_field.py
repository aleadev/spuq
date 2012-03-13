"""The expansion of a coefficient field."""

from types import GeneratorType
from itertools import count
from numpy import infty

from spuq.utils.type_check import takes, anything, sequence_of
from spuq.linalg.function import GenericFunction
from spuq.stochastics.random_variable import RandomVariable, DeterministicPseudoRV
from spuq.utils import strclass

class CoefficientField(object):
    """Expansion of a coefficient field according to EGSZ (1.2)."""

#    @takes(anything, sequence_of(GenericFunction), sequence_of(RandomVariable))
    @takes(anything, sequence_of(anything), sequence_of(RandomVariable))
    def __init__(self, funcs, rvs):
        """Initialise with list of functions and list of random variables.
        
        The first function is the mean field for which no random
        variable is required, i.e. len(funcs)=len(rvs)+1.
        DeterministicPseudoRV is associated with the mean field implicitly."""
        assert len(funcs) == len(rvs) + 1, (
            "Need one more function than random variable (for the deterministic case)")
        self._funcs = list(funcs)
        # first function is deterministic mean field
        self._rvs = list(rvs)
        self._rvs.insert(0, DeterministicPseudoRV)

    @classmethod
    @takes(anything, sequence_of(GenericFunction), RandomVariable)
    def createWithIidRVs(cls, funcs, rv):
        """Create coefficient field where all expansion terms have the identical random variable."""
        # random variables are stateless, so we can just use the same n times 
        rvs = [rv] * (len(funcs) - 1)
        return cls(funcs, rvs)

    def coefficients(self):
        """Return expansion iterator for (Function,RV) pairs."""
        for i in xrange(len(self._funcs)):
            yield self._funcs[i], self._rvs[i]

    def __getitem__(self, i):
        assert i < len(self._funcs), "invalid index"
        return self._funcs[i], self._rvs[i]

    def __repr__(self):
        return "<%s funcs=%s, rvs=%s>" % \
               (strclass(self.__class__), self._funcs[1:], self._rvs)

    @property
    def length(self):
        return len(self._funcs)

    def __len__(self):
        """Length of coefficient field expansion."""
        return self.length


class ParametricCoefficientField(CoefficientField):
    """Expansion of a coefficient field according to EGSZ (1.2)."""

    @takes(anything, GeneratorType, GeneratorType)
    def __init__(self, func_gen, rv_gen):
        """Initialise with function and random variable generators.
        
        The first function is the mean field with which a
        DeterministicPseudoRV is associated implicitly."""
        self._func_gen = func_gen
        self._rvs_gen = rv_gen
        self._funcs = list()
        self._rvs = list()
        # first function is deterministic mean field
        self._funcs.append(self._func_gen.next())
        self._rvs.append(DeterministicPseudoRV)

    @classmethod
    @takes(anything, GeneratorType, RandomVariable)
    def createWithIidRVs(cls, func_gen, rv):
        """Create coefficient field where all expansion terms have the identical random variable."""
        # random variables are stateless, so we can just use the same n times 
        return cls(func_gen, (rv for _ in count()))

    def coefficients(self):
        """Return expansion iterator for (Function,RV) pairs."""
        for i in count():
            yield self[i]

    def __getitem__(self, i):
        if i >= len(self._funcs):
            for _ in range(i - len(self._funcs) + 1):
                self._funcs.append(self._func_gen.next())
                self._rvs.append(self._rvs_gen.next())
        return self._funcs[i], self._rvs[i]

    @property
    def length(self):
        return infty

    def __len__(self):
        """Length of coefficient field expansion."""
        # TODO: how shall we handle this?
        raise Exception("length of ParametricCoefficientField is infinite")
