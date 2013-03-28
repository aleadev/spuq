from abc import ABCMeta, abstractmethod
from spuq.linalg.basis import FunctionBasis

class FEMBasis(FunctionBasis):
    """"FEM basis"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def refine(self, cell_ids=None):
        """Refine mesh of basis uniformly or with respect to cell ids, returns
        (new_basis, prolongate, restrict,)."""
        raise NotImplementedError

    @abstractmethod
    def project_onto(self, vec):
        """Project vector onto own FEMBasis."""
        raise NotImplementedError

    @abstractmethod
    def get_dof_coordinates(self):
        """Return coordinates of degrees of freedom."""
        raise NotImplementedError
    