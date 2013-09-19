from scipy.linalg import norm
import numpy as np
import scipy.sparse as sps
from spuq.linalg.cptensor import CPTensor
from spuq.linalg.tensor_basis import TensorBasis

from spuq.linalg.tensor_operator import TensorOperator
from spuq.linalg.tensor_vector import FullTensor
from spuq.linalg.scipy_operator import ScipyOperator
from spuq.linalg.basis import CanonicalBasis


def construct_operator(n1, n2, R):
    # prepare "deterministic" matrices
    basis1 = CanonicalBasis(n1)
    basis2 = CanonicalBasis(n2)

    A1 = [ScipyOperator(sps.csr_matrix(np.random.rand(n1, n1)),
                        domain=basis1, codomain=basis1)
          for _ in range(R)]
    A2 = [ScipyOperator(sps.csr_matrix(np.random.rand(n2, n2)),
                        domain=basis2, codomain=basis2)
          for _ in range(R)]

    basis = TensorBasis([basis1, basis2])
    return TensorOperator(A1, A2, domain=basis, codomain=basis)

def construct_cptensor(n1, n2, R):
    # prepare vector
    basis1 = CanonicalBasis(n1)
    basis2 = CanonicalBasis(n2)
    basis = TensorBasis([basis1, basis2])
    X1 = np.random.rand(n1, R)
    X2 = np.random.rand(n2, R)
    return CPTensor([X1, X2], basis)


# test tensor operator application
# ================================
n1 = 100
n2 = 15
for R in [1,2,5]:
    # prepare data
    A = construct_operator(n1, n2, R)
    u = construct_cptensor(n1, n2, 5)

    f1 = (A * u).flatten().as_array()
    f2 = (A * u.flatten()).as_array()
    print np.linalg.norm(f1 - f2)


    print u.rank
    for r in range(6):
        v = u.truncate(r)
        u_mat = u.flatten().as_array()
        v_mat = v.flatten().as_array()
        print ">>>", v.rank, np.linalg.norm(u_mat - v_mat)


# implement rank method in cptensor
# look at rank increment
# implement rank reduction to fixed rank



    # test application of operator
    #w = A * u

