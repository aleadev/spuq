# Karhunen-Loeve expansion related functions

import numpy as np
import scipy

def solve_KL_eigenvalue_problem(cov, basis, M):
    c4dof = basis.get_dof_coordinates()
    N = c4dof.shape[0]
    C = np.ndarray(N, N)
    # evaluate covariance matrix
    ...
    C[i,j] = cov(c4dof[i],c4dof[j]) 
    ...
    # get Gram matrix
    G = basis.gramian()
    W = M*C*M
    # evaluate M eigenpairs of symmetric eigenvalue problem
    scipy.eigs(W)
    ...