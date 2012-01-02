"""Interface for spatial FEM discretisation"""

from abc import ABCMeta, abstractproperty, abstractmethod
from spuq.utils.type_check import *
from spuq.fem.fem_basis import FEMBasis

class FEMDiscretisation(object):
    """FEM discretisation interface"""
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    @takes(anything, dict, FEMBasis)
    def assemble_operator(self, data, basis):
        """Evaluate discrete operator"""
        return NotImplemented
    
