"""Interface for spatial FEM discretisation"""

from abc import ABCMeta, abstractmethod
from spuq.utils.type_check import takes, anything
from spuq.fem.fem_basis import FEMBasis

class FEMDiscretisation(object):
    """FEM discretisation interface"""

    __metaclass__ = ABCMeta

    @abstractmethod
    @takes(anything, anything, FEMBasis)
    def assemble_operator(self, data, basis):
        """Evaluate discrete operator"""
        raise NotImplementedError
