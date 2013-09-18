import logging
import numpy as np
from spuq.linalg.operator import ComponentOperator
from spuq.linalg.tensor_basis import TensorBasis
from spuq.linalg.tensor_vector import FullTensor
from spuq.utils.type_check import takes, sequence_of, anything

class CPTensor(TensorVector):
    @takes(anything, sequence_of(np.ndarray), TensorBasis)
    def __init__(self, X, basis):
        super(TensorVector, self).__init__(basis)
        self._X = X

    @takes(anything, ComponentOperator, int)
    def apply_to_dim(self, A, axis):
        Y = [x for x in self._X]
        Y[axis] = A.apply_to_matrix(self._X[axis])
        return self.__class__(Y, self._basis)

    def __eq__(self, other):  # pragma: no cover
        """Test whether vectors are equal."""
        return (
            self._basis == other._basis and
            all([np.all(x1 == x2) for x1, x2 in zip(self._X, other._X)])
        )

    def __neg__(self):  # pragma: no cover
        """Compute the negative of this vector."""
        Y = [x for x in self._X]
        Y[0] *= -1
        return CPTensor(Y, self.basis)

    def __iadd__(self, other):  # pragma: no cover
        """Add another vector to this one."""
        self._X = [np.hstack([x1, x2]) for x1, x2 in zip(self.X, other.X)]
        return self

    def __imul__(self, other):  # pragma: no cover
        """Multiply this vector with a scalar."""
        if isinstance(other, Scalar):
            self._X[0] *= -other
            return self
        else:
            raise TypeError

    def __inner__(self, other):
        raise NotImplementedError

    @property
    def order(self):
        return len(self._X)

    def to_full(self):
        assert self.order == 2
        return FullTensor(self._X[0]*self._X[1].T, self._basis)
