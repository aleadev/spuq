from scipy.linalg import norm
import numpy as np
import scipy.sparse as sps
from spuq.linalg.cptensor import CPTensor
from spuq.linalg.operator import MultiplicationOperator
from spuq.linalg.tensor_basis import TensorBasis

from spuq.linalg.tensor_operator import TensorOperator
from spuq.linalg.tensor_vector import FullTensor
from spuq.linalg.scipy_operator import ScipyOperator
from spuq.linalg.basis import CanonicalBasis
from test_zone.sgfem import pcg


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
R = 5

A = construct_operator(n1, n2, R)
u = construct_cptensor(n1, n2, 5)


def construct_preconditioner(A):
    basis = A.domain
    basis1 = basis[0]
    basis2 = basis[1]

    return TensorOperator(
        [MultiplicationOperator(1, domain=basis1)],
        [MultiplicationOperator(1, domain=basis2)],
        basis)


P = construct_preconditioner(A)
u_flat = u.flatten()
f_flat = A * u_flat

u_flat2 = pcg.pcg(A, f_flat, P, 0*u_flat)

print np.linalg.norm(u_flat.as_array() - u_flat2.as_array())
