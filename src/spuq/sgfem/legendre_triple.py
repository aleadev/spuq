from __future__ import division
import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, csc_matrix, csr_matrix
from spuq.polyquad.polynomials import LegendrePolynomials

def cmp_by_index(a1, a2):
    assert a1.shape == a2.shape
#     l1 = len(a1)
#     l2 = len(a2)
#     l = max(l1,l2)
#     a1 = np.append(a1, [0]*(l-l1))
#     a2 = np.append(a2, [0]*(l-l2))
    return [cmp(v1,v2) for v1,v2 in zip(a1,a2)]

def evaluate_Legendre_triple(I_a, I_b):
    # get dimensions
    M = I_a.shape[0]
    N = I_b.shape[0]
    K = I_a.shape[1]
    # create polynomial instance
    lp = LegendrePolynomials()
    # prepare sparse matrices
    L = [dok_matrix((M, N)) for _ in range(K)]
    for k in range(K):
        for i, mui in enumerate(I_a):
            for j, muj in enumerate(I_b):
                # compare indices
                cmp_ij = np.array(cmp_by_index(mui, muj))
                # test for equality except kth component
                if np.all(np.append(cmp_ij[:k-1], cmp_ij[k:]) == 0):
                    val = 0.0
                    ak, bk = mui[k], muj[k]
                    a, b, c = lp.get_beta(bk)
                    if ak == bk+1:
#                        print "A", a
                        val += a
                    if ak == bk:
                        val -= b
                    if ak == bk-1:
#                        print "C", c
                        val -= c
                    if val <> 0.0:
                        L[k][i, j] = val
    # convert to efficient sparse format
    L = [l.asformat('csr') for l in L] 
    return L
