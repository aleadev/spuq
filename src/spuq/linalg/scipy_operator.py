import logging

import numpy as np
import scipy.sparse as sps
import scipy.linalg as la

from spuq.linalg.basis import Basis
from spuq.linalg.vector import Vector
from spuq.linalg.operator import Operator
from spuq.utils.type_check import takes, anything

logger = logging.getLogger(__name__)


class ScipyOperatorBase(Operator):
    @takes(anything, sps.spmatrix, Basis)
    def __init__(self, matrix, domain, codomain):
        self._matrix = matrix
        self._domain = domain
        self._codomain = codomain

    @property
    def matrix(self):
        return self._matrix

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain



class ScipyOperator(ScipyOperatorBase):
    @takes(anything, Vector)
    def apply(self, vec):
        # TODO: check basis
        new_vec = vec.copy()
        new_vec.coeffs = self._matrix * new_vec.coeffs
        return new_vec

    def as_matrix(self):
        return self._matrix


class ScipySolveOperator(ScipyOperatorBase):
    @takes(anything, Vector)
    def apply(self, vec):
        # TODO: check basis
        new_vec = vec.copy()
        from scipy.sparse.linalg import spsolve
        new_vec.coeffs = spsolve(self._matrix, vec.coeffs)
        return new_vec

    def as_matrix(self):
        # TODO: compute inverse (issue warning?)
        logger.warning("computing the inverse of a sparse matrix")
        return la.inv(self._matrix.toarray())


