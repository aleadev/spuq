# Karhunen-Loeve expansion related functions

import numpy as np
import itertools as it
from scipy.linalg import eigh
# from scipy.sparse.linalg import eigsh
# np.set_printoptions(suppress=True)
from spuq.linalg.operator import MatrixOperator


class KLexpansion(object):
    # M term KL expansion of covariance with given discrete spatial basis
    def __init__(self, cov, basis, M, evals = None, evecs = None, KLtype = 'L2'):
        self.cov = cov
        self.basis = basis
        self.M = M
        if evals is None or evecs is None:
            self.evals, self.evecs = solve_KL_eigenvalue_problem(cov, basis, M, KLtype)
        else:
            self.evals, self.evecs = evals, evecs
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

    def g(self, x, basis = None):
        # evaluate all eigenfunctions at x
        if basis is None:
            evfuncs = self.evfuncs
        else:
            evfuncs = [basis.project_onto(g) for g in self.evfuncs]
        v = [g(x) for g in evfuncs]
        return np.array(v)


def solve_KL_eigenvalue_problem(cov, basis, M, KLtype = 'L2'):
    c4dof = basis.get_dof_coordinates()
    N = c4dof.shape[0]
    C = np.ndarray((N, N))
    # evaluate covariance matrix
    for i, j in it.product(range(N), repeat=2):
        C[i, j] = cov(c4dof[i], c4dof[j])
    if KLtype == 'L2':
        # get Gram matrix
        G = basis.gramian
    else:
        assert KLtype == 'H1'
        # get stiffness matrix
        G = basis.stiffness
    W = G * MatrixOperator(C) * G
    # evaluate M largest eigenpairs of symmetric eigenvalue problem
    Gmat = G.as_matrix()
    J = Gmat.shape[0]
    evals, evecs = eigh(W.as_matrix(), Gmat, eigvals=(J - M, J - 1))
    return evals, evecs.T
