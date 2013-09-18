import logging

import numpy as np
import scipy.sparse as sps
import scipy.linalg as la
import scipy.sparse.linalg as spsla

from spuq.linalg.basis import Basis
from spuq.linalg.vector import Vector
from spuq.linalg.operator import Operator, BaseOperator, ComponentOperator
from spuq.utils.type_check import takes, anything

logger = logging.getLogger(__name__)


class ScipyOperatorBase(BaseOperator, ComponentOperator):
    @takes(anything, sps.spmatrix, Basis)
    def __init__(self, matrix, domain, codomain):
        self._matrix = matrix
        super(ScipyOperatorBase, self).__init__(domain, codomain)

    @property
    def matrix(self):
        return self._matrix




class ScipyOperator(ScipyOperatorBase):
    @takes(anything, Vector)
    def apply(self, vec):
        # TODO: check basis
        new_vec = vec.copy()
        new_vec.coeffs = self._matrix * new_vec.coeffs
        return new_vec

    def as_matrix(self):
        return self._matrix

    @takes(anything, np.ndarray)
    def apply_to_matrix(self, X):
        return self._matrix * X


class ScipySolveOperator(ScipyOperatorBase):
    @takes(anything, Vector)
    def apply(self, vec):
        # TODO: check basis
        new_vec = vec.copy()
        new_vec.coeffs = spsla.spsolve(self._matrix, vec.coeffs)
        return new_vec

    def as_matrix(self):
        # TODO: compute inverse (issue warning?)
        logger.warning("computing the inverse of a sparse matrix")
        return la.inv(self._matrix.toarray())

    @takes(anything, np.ndarray)
    def apply_to_matrix(self, X):
        Y = np.zeros_like(X)
        for i in range(X.shape[1]):
            Y[:,i] = spsla.spsolve(self._matrix, X[:,i])[:,np.newaxis]
        return Y

