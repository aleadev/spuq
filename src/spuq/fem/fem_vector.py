import abc
from spuq.linalg.vector import FlatVector

class FEMVector(FlatVector):
    '''ABC FEM vector which contains a coefficient vector and a discrete basis.'''

    @abc.abstractmethod
    def eval(self, x):
        raise NotImplementedError
