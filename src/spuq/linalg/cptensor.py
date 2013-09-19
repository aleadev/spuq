import logging
import numpy as np
from spuq.linalg.basis import CanonicalBasis
from spuq.linalg.vector import Scalar
from spuq.linalg.operator import ComponentOperator
from spuq.linalg.tensor_basis import TensorBasis
from spuq.linalg.tensor_vector import FullTensor, TensorVector
from spuq.utils.type_check import takes, sequence_of, anything

class CPTensor(TensorVector):
    @takes(anything, sequence_of(np.ndarray), TensorBasis)
    def __init__(self, X, basis):
        self._basis = basis
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
            np.all(self.flatten().as_array() == other.flatten().as_array())
            # TODO: the previous line is not really efficient
            #all([np.all(x1 == x2) for x1, x2 in zip(self._X, other._X)])
        )

    def __neg__(self):  # pragma: no cover
        """Compute the negative of this vector."""
        Y = [x for x in self._X]
        Y[0] = -Y[0]
        return CPTensor(Y, self.basis)

    def __iadd__(self, other):  # pragma: no cover
        """Add another vector to this one."""
        self._X = [np.hstack([x1, x2]) for x1, x2 in zip(self._X, other._X)]
        return self

    def __imul__(self, other):  # pragma: no cover
        """Multiply this vector with a scalar."""
        if isinstance(other, Scalar):
            self._X[0] = other * self._X[0]
            return self
        else:
            raise TypeError

    def __inner__(self, other):
        assert self.order == 2
        X = self._X
        Y = other._X
        return np.sum(np.dot(X[0].T, Y[0]) * np.dot(X[1].T, Y[1]))

        raise NotImplementedError

    @property
    def order(self):
        return len(self._X)

    @property
    def rank(self):
        return self._X[0].shape[1]

    def truncate(self, R):
        assert self.order == 2
        R = min(R, self.rank)
        Q1, R1 = np.linalg.qr(self._X[0])
        Q2, R2 = np.linalg.qr(self._X[1])
        U, s, V_T = np.linalg.svd(np.dot(R1, R2.T), full_matrices=False)
        V = V_T.T
        U = np.dot(U, np.diag(s))
        X1 = np.dot(Q1, U)[:,:R]
        X2 = np.dot(Q2, V)[:,:R]
        return CPTensor([X1, X2], self.basis)

    def flatten(self):
        # TODO: implement for higher-order tensors
        assert len(self._X) == 2
        Y = np.dot(self._X[0], self._X[1].T)
        return FullTensor(Y, self._basis)

    def to_full(self):
        return self.flatten()
