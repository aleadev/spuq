from __future__ import division

import collections
import numpy as np
import scipy.sparse as sps

from spuq.linalg.scipy_operator import ScipyOperator
from spuq.linalg.basis import CanonicalBasis


def evaluate_triples(polysys, I_a, I_b):
    # get dimensions
    M = I_a.shape[0]
    N = I_b.shape[0]
    K = I_a.shape[1]
    
    if not isinstance(polysys, collections.Sequence):
        polysys = [polysys] * K
    # prepare sparse matrices
    L = [sps.dok_matrix((M, N)) for _ in range(-1,K)]
    for k in range(-1, K):
        for i, mui in enumerate(I_a):
            for j, muj in enumerate(I_b):
                # compare indices
                cmp_ij = mui == muj
                # test for equality except kth component
                if np.all(cmp_ij[:k]) and np.all(cmp_ij[k+1:]):
                    ak, bk = mui[k], muj[k]
                    delta = ak - bk
                    if -1 <= delta <= 1:
                        if k != -1:
                            beta = polysys[k].get_beta(bk)
                            val = beta[delta]
                        else:
                            val = 1
                    
                        if val <> 0.0:
                            L[k+1][i, j] = val
    # convert to efficient sparse format
    L = [ScipyOperator(l.asformat('csr'), domain=CanonicalBasis(N), codomain=CanonicalBasis(M)) for l in L]
    return L
