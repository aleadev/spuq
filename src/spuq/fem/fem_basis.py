import abc
from spuq.linalg.basis import FunctionBasis

class FEMBasis(FunctionBasis):
    ''' '''
    
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def refine(self, cells):
        '''refines mesh of basis uniformly or wrt cells, returns (prolongate,restrict,...).''' 
        return NotImplemented

    @abc.abstractmethod
    def project(self, vec, vecbasis, ptype):
        '''project coefficient vector to FEMBasis.
        
        vec can either be a FEMVector or an array in which case a basis has to be passed as well.
        in the first case, a new FEMVector is returned, in the second case, a coefficient array and the new basis is returned.''' 
        return NotImplemented

    @abc.abstractproperty
    def basis(self):
        return NotImplemented
    
    @basis.setter
    def basis(self, val):
        return NotImplemented
    
    @abc.abstractproperty
    def mesh(self):
        return NotImplemented
    
    @mesh.setter
    def mesh(self, val):
        return NotImplemented
