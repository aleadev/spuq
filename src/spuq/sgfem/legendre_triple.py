from __future__ import division
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, csc_matrix, csr_matrix
import collections

def evaluate_triples(polysys, I_a, I_b):
    # get dimensions
    M = I_a.shape[0]
    N = I_b.shape[0]
    K = I_a.shape[1]
    
    if not isinstance(polysys, collections.Sequence):
        polysys = [polysys] * K
    # prepare sparse matrices
    L = [dok_matrix((M, N)) for _ in range(K)]
    for k in range(K):
        for i, mui in enumerate(I_a):
            for j, muj in enumerate(I_b):
                # compare indices
                cmp_ij = mui == muj
                # test for equality except kth component
                if np.all(cmp_ij[:k]) and np.all(cmp_ij[k+1:]):
                    ak, bk = mui[k], muj[k]
                    delta = ak - bk
                    if -1 <= delta <= 1:
                        beta = polysys[k].get_beta(bk)
                        val = beta[delta]
                    
                        if val <> 0.0:
                            L[k][i, j] = val
    # convert to efficient sparse format
    L = [l.asformat('csr') for l in L] 
    return L
