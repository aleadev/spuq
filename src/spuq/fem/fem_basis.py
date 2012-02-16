from abc import ABCMeta, abstractmethod
from spuq.linalg.basis import FunctionBasis

class FEMBasis(FunctionBasis):
    """"FEM basis"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def refine(self, cells):
        """Refine mesh of basis uniformly or wrt cells, returns
        (prolongate,restrict,...)."""
        raise NotImplementedError

    @abstractmethod
    def project_onto(self, vec):
        """Project coefficient vector to FEMBasis.
        
        vec can either be a FEMVector or an array in which case a
        basis has to be passed as well. In the first case, a new
        FEMVector is returned, in the second case, a coefficient array
        and the new basis is returned."""
        raise NotImplementedError
