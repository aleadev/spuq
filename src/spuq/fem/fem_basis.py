from abc import *
from spuq.linalg.basis import FunctionBasis

class FEMBasis(FunctionBasis):
    ''' '''
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def refine(self, cells):
        """Refine mesh of basis uniformly or wrt cells, returns
        (prolongate,restrict,...)."""
        return NotImplemented

    @abstractmethod
    def project(self, vec, vecbasis, ptype):
        """Project coefficient vector to FEMBasis.
        
        vec can either be a FEMVector or an array in which case a
        basis has to be passed as well.  in the first case, a new
        FEMVector is returned, in the second case, a coefficient array
        and the new basis is returned."""
        return NotImplemented
