# Karhunen-Loeve expansion related functions

import numpy as np
import itertools as it
from scipy.linalg import eigh
# from scipy.sparse.linalg import eigsh
# np.set_printoptions(suppress=True)
from spuq.linalg.operator import MatrixOperator

def solve_KL_eigenvalue_problem(cov, basis, M):
    c4dof = basis.get_dof_coordinates()
    N = c4dof.shape[0]
    C = np.ndarray((N, N))
    # evaluate covariance matrix
    for i, j in it.product(range(N), repeat=2):
        C[i, j] = cov(c4dof[i], c4dof[j])
    # get Gram matrix
    G = basis.gramian
    W = G * MatrixOperator(C) * G
    # evaluate M largest eigenpairs of symmetric eigenvalue problem
    Gmat = G.as_matrix()
    J = Gmat.shape[0]
    evals, evecs = eigh(W.as_matrix(), Gmat, eigvals=(J - M, J - 1))
    return evals, evecs.T


class KLexpansion(object):
    # M term KL expansion of covariance with given discrete spatial basis
    def __init__(self, cov, basis, M):
        self.cov = cov
        self.basis = basis
        self.M = M
        self.evals, self.evecs = solve_KL_eigenvalue_problem(cov, basis, M)
        self._init_funcs()
        print "===VALS", self.evals
        print "===VECS", self.evecs

    def _init_funcs(self):
        self.evfuncs = []
        for i in range(self.M):
            ef = self.basis.new_vector()
            ef.coeffs = self.evecs[i]
            self.evfuncs.append(ef)

    def __getitem__(self, i):
        assert i < self.M
        return self.evals[i], self.evfuncs[i]

    def g(self, x):
        # evaluate all eigenfunctions at x
        v = [g(x) for g in self.evfuncs]
        return np.array(v)
    
