import logging
import numpy as np
import scipy.sparse as sps

from spuq.linalg.vector import Vector, Scalar
from spuq.linalg.operator import Operator, ComponentOperator
from spuq.linalg.basis import Basis, CanonicalBasis
from spuq.utils.type_check import takes, anything

logger = logging.getLogger(__name__)

class TensorBasis(CanonicalBasis):
    def __init__(self):
        super(TensorBasis, self).__init__(-1)


class TensorVector(Vector):
    def __init__(self, basis):
        self._basis = basis

    @property
    def basis(self):
        return self._basis


class MatrixTensorVector(TensorVector):

    @takes(anything, np.ndarray, TensorBasis)
    def __init__(self, X, basis):
        """Initialise with list of vectors.
        @param X:
        @param basis:
        """
        TensorVector.__init__(self, basis)
        self.X = X
    
    def dim(self):  # pragma: no cover
        """Return dimension of this vector."""
        return self.X.shape

    def copy(self):  # pragma: no cover
        return self.__class__(self.X.copy(), self.basis)

    def flatten(self):
        return self.X.ravel()

    def as_matrix(self):
        return self.X

    @takes(anything, Operator, int)
    #@takes(anything, (np.ndarray, sps.spmatrix), int)
    def apply_matrix_old(self, op, axis):
        A = op.as_matrix()
        X = np.matrix(self.X)
        if axis == 0:
            Y = A * X
        elif axis == 1:
            Y = (A * X.T).T
        return self.__class__(Y, self._basis)

    @takes(anything, ComponentOperator, int)
    def apply_to_dim(self, A, axis):
        X = np.matrix(self.X)
        if axis == 0:
            Y = A.apply_to_matrix(X)
        elif axis == 1:
            Y = (A.apply_to_matrix(X.T)).T
        return self.__class__(Y, self._basis)

    @property
    def coeffs(self):
        return self.flatten()

    def __eq__(self, other):  # pragma: no cover
        """Test whether vectors are equal."""
        return np.all(self.X == other.X)
 
    def __neg__(self):  # pragma: no cover
        """Compute the negative of this vector."""
        return MatrixTensorVector(-self.X, self.basis)
 
    def __iadd__(self, other):  # pragma: no cover
        """Add another vector to this one."""
        self.X += other.X
        return self
 
    def __imul__(self, other):  # pragma: no cover
        """Multiply this vector with a scalar."""
        if isinstance(other, Scalar):
            self.X *= other
            return self
        else:
            raise TypeError

    def __inner__(self, other):
        assert isinstance(other, MatrixTensorVector)
        return np.sum(np.multiply(self.X, other.X))

    @classmethod
    def from_list(cls, L):
        X = np.vstack(L).T
        return cls(X, TensorBasis())