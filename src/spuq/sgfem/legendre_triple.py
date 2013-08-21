from __future__ import division
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, csc_matrix, csr_matrix

def evaluate_Legendre_triple(I_a, I_b, I_c):
    # get dimensions
    M = I_a.shape[0]
    N = I_b.shape[0]
    K = I_c.shape[0]
    # prepare sparse matrices
    L = [dok_matrix(M, N) for _ in range(K)]
    for k, muk in I_c.iteritems():
        for i, mui in I_a.iteritems():
            for j, muj in I_b.iteritems():
                val = 0.
                # compare indices
                cmp_ij = mui.cmp_indices(muj)
                # test for equality except kth component
                if np.all(np.append(cmp_ij[:k], cmp_ij[k:])):
                    # TODO: evaluate coefficient
                    val = 1
                    L[k][i, j] = val
    # convert to efficient sparse format
    L = [l.toformat('csr') for l in L] 
    return L
