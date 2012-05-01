"""The expansion of a coefficient field."""

import sys
from types import GeneratorType

from spuq.utils.type_check import takes, anything, sequence_of
from spuq.stochastics.random_variable import RandomVariable
from spuq.utils import strclass
from spuq.utils.parametric_array import ParametricArray
from abc import ABCMeta, abstractmethod, abstractproperty


class CoefficientField(object):
    """Expansion of a coefficient field according to EGSZ (1.2)."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def mean_func(self):
        pass

    @abstractproperty
    def funcs(self):
        pass

    @abstractproperty
    def rvs(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

    @abstractmethod
    def sample_rvs(self):
        pass

    def sample_realization(self, Delta):
        def prod(l):
            p = 1
            for f in l:
                if p is None:
                    p = f
                else:
                    p *= f
            return p
        RV_samples = self.sample_rvs()
        sample_map = {}
        for mu in Delta:
            sample_map[mu] = prod(self[m + 1][1].orth_polys[mu[m]](RV_samples[m + 1]) for m in range(len(mu)))
        return sample_map, RV_samples



class ListCoefficientField(CoefficientField):
    """Expansion of a coefficient field according to EGSZ (1.2)."""

    #    @takes(anything, sequence_of(GenericFunction), sequence_of(RandomVariable))
    @takes(anything, anything, sequence_of(anything), sequence_of(RandomVariable))
    def __init__(self, mean_func, funcs, rvs):
        """Initialise with list of functions and list of random variables.
        
        The first function is the mean field for which no random
        variable is required, i.e. len(funcs)=len(rvs)+1.
        DeterministicPseudoRV is associated with the mean field implicitly."""
        assert len(funcs) == len(rvs)
        self._mean_func = mean_func
        self._funcs = list(funcs)
        self._rvs = list(rvs)

    @classmethod
    @takes(anything, anything, sequence_of(anything), RandomVariable)
    def create_with_iid_rvs(cls, mean_func, funcs, rv):
        """Create coefficient field where all expansion terms have the identical random variable."""
        # random variables are stateless, so we can just use the same n times 
        rvs = [rv] * len(funcs)
        return cls(mean_func, funcs, rvs)

    @property
    def mean_func(self):
        return self._mean_func

    @property
    def funcs(self):
        return self._funcs

    @property
    def rvs(self):
        return self._rvs

    def __len__(self):
        """Length of coefficient field expansion."""
        return len(self._funcs)

    def __getitem__(self, i):
        assert i < len(self._funcs), "invalid index"
        return self._funcs[i], self._rvs[i]

    def __repr__(self):
        return "<%s mean=%s funcs=%s, rvs=%s>" % \
               (strclass(self.__class__), self.mean_func, self._funcs, self._rvs)

    def sample_rvs(self):
        return (float(self._rvs[i].sample(1)) for i in len(self._rvs))


class ParametricCoefficientField(CoefficientField):
    """Expansion of a coefficient field according to EGSZ (1.2)."""

    @takes(anything, anything, (callable, GeneratorType), (callable, GeneratorType))
    def __init__(self, mean_func, func_func, rv_func):
        """Initialise with function and random variable generators.

        The first function is the mean field with which a
        DeterministicPseudoRV is associated implicitly."""
        self._mean_func = mean_func
        self._funcs = ParametricArray(func_func)
        self._rvs = ParametricArray(rv_func)

    @classmethod
    @takes(anything, anything, callable, RandomVariable)
    def create_with_iid_rvs(cls, mean_func, func_func, rv):
        """Create coefficient field where all expansion terms have the identical random variable."""
        # random variables are stateless, so we can just use the same n times
        return cls(mean_func, func_func, lambda i: rv)

    @property
    def mean_func(self):
        return self._mean_func

    @property
    def funcs(self):
        return self._funcs

    @property
    def rvs(self):
        return self._rvs

    def __len__(self):
        """Length of coefficient field expansion."""
        return sys.maxint
        # Note: infty doesn't work here because it cannot be converted to
        # integer (that's what len() does to the value returned from here)

    def __getitem__(self, i):
        return self._funcs[i], self._rvs[i]

    def sample_rvs(self):
        return ParametricArray(lambda i: float(self._rvs[i].sample(1)))
