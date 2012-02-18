from abc import abstractproperty, abstractmethod
from spuq.linalg.vector import FlatVector

class FEMVector(FlatVector):
    '''ABC FEM vector which contains a coefficient vector and a discrete basis.'''

    @abstractmethod
    def eval(self, x):
        raise NotImplementedError

    @abstractproperty
    def basis(self):
        '''return FEMBasis'''
        raise NotImplementedError

    @abstractproperty
    def coeffs(self):
        '''return coefficient vector'''
        raise NotImplementedError

    @coeffs.setter
    def coeffs(self, val):
        '''set coefficient vector'''
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        """Compare vectors for equality.

        Note that vectors are only considered equal when they have
        exactly the same type."""
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError

    @abstractmethod
    def __iadd__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __isub__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __imul__(self, other):
        raise NotImplementedError
