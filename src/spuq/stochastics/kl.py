# Karhunen-Loeve expansion related functions

import numpy as np
import itertools as it

from scipy.linalg import eigh
# from scipy.sparse.linalg import eigsh
# np.set_printoptions(suppress=True)

def solve_KL_eigenvalue_problem(cov, basis, M):
    c4dof = basis.get_dof_coordinates()
    N = c4dof.shape[0]
    C = np.ndarray(N, N)
    # evaluate covariance matrix
    for i, j in it.product(range(N), repeat=2):
        C[i,j] = cov(c4dof[i],c4dof[j]) 
    # get Gram matrix
    G = basis.gramian()
    W = G*C*G
    # evaluate M largest eigenpairs of symmetric eigenvalue problem
    evals, evecs = eigh(W, M, which='LM')
    return (evals, evecs)
