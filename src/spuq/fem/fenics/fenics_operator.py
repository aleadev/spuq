import numpy as np
import scipy.sparse as sps

import dolfin

from spuq.linalg.operator import Operator
from spuq.linalg.scipy_operator import ScipyOperator, ScipySolveOperator
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.utils.type_check import takes, anything


class FEniCSOperatorBase(Operator):
    @takes(anything, dolfin.Matrix, FEniCSBasis)
    def __init__(self, matrix, basis):
        self._matrix = matrix
        self._basis = basis

    @property
    def domain(self):
        return self._basis

    @property
    def codomain(self):
        return self._basis
    
    @property
    def dim(self):
        return self._basis.dim

    def _as_scipy_matrix(self):
        rows, cols, values = self._matrix.data()
        return sps.csr_matrix((values, cols, rows))


class FEniCSOperator(FEniCSOperatorBase):
    @takes(anything, dolfin.Matrix, FEniCSBasis, np.ndarray)
    def __init__(self, matrix, basis, mask=None):
        FEniCSOperatorBase.__init__(self, matrix, basis)
        self._mask = mask
        
    @takes(anything, FEniCSVector)
    def apply(self, vec):
        new_vec = vec.copy()
        if self._mask is not None:
            new_vec.coeffs = new_vec.coeffs * self._mask
        new_vec.coeffs = self._matrix * new_vec.coeffs
        if self._mask is not None:
            new_vec.coeffs = new_vec.coeffs * self._mask
        return new_vec

    def as_scipy_operator(self):
        matrix = self._as_scipy_matrix()
        basis = self._basis.as_canonical_basis()
        return ScipyOperator(matrix, domain=basis, codomain=basis)


class FEniCSSolveOperator(FEniCSOperatorBase):
    @takes(anything, FEniCSVector)
    def apply(self, vec):
        new_vec = vec.copy()
        dolfin.solve(self._matrix, new_vec.coeffs, vec.coeffs)
        return new_vec
    def as_scipy_operator(self):
        matrix = self._as_scipy_matrix()
        basis = self._basis.as_canonical_basis()
        return ScipySolveOperator(matrix, domain=basis, codomain=basis)
