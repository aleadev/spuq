from scipy.linalg import norm
import numpy as np
import scipy.sparse as sps

from spuq.linalg.tensor_operator import TensorOperator
from spuq.linalg.tensor_vector import FullTensor
from spuq.linalg.scipy_operator import ScipyOperator
from spuq.linalg.basis import CanonicalBasis


def construct_data(k, d, m):
    # prepare "deterministic" matrices
    K = [sps.csr_matrix(np.random.rand(k,k)) for _ in range(m)]
    K = [ScipyOperator(K_.asformat('csr'), domain=CanonicalBasis(K_.shape[0]), codomain=CanonicalBasis(K_.shape[1])) for K_ in K]

    # prepare "stochastic" matrices
    D = [sps.csr_matrix(np.random.rand(d,d)) for _ in range(m)]
    # convert to efficient sparse format
    D = [ScipyOperator(D_.asformat('csr'), domain=CanonicalBasis(D_.shape[0]), codomain=CanonicalBasis(D_.shape[1])) for D_ in D]

    # prepare vector
    u = [np.random.rand(k) for _ in range(d)]
    return K, D, u


# test tensor operator application
# ================================
I, J = 100, 15
for M in [1,2,5]:
    # prepare data
    K, D, u = construct_data(I, J, M)
    A = TensorOperator(K, D)
    u = FullTensor.from_list(u)
    # print matricisation of tensor operator
    Amat = A.as_matrix()
    print Amat.shape

    # compare with numpy kronecker product
    M2 = [sps.kron(K_.matrix, D_.matrix) for K_, D_ in zip(K, D)]
    M2 = np.sum(M2)

    print "error norm: ", norm(Amat-M2.todense()), " == ", norm(Amat-M2)

    # test application of operator
    w = A * u
