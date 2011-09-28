import abc
from spuq.linalg.vector import FlatVector

class FEMVector(FlatVector):
    '''ABC FEM vector which contains a coefficient vector and a discrete basis.'''
    
#    def __init__(self, coeff, basis):
#        assert isinstance(basis, FEMBasis)
        #super(self,FEMVector).__init__(coeff, basis)

    @abc.abstractmethod
    def evaluate(self, x):
        return NotImplemented

#    def project(self, basis, ptype=FEMBasis.PROJECTION.INTERPOLATION, inverse=False):
#        assert isinstance(basis, FEMBasis)
#        newcoeff = FEMBasis.project(self.coeff, self.basis, basis, ptype, inverse)
#        return FEMVector(newcoeff, basis)
