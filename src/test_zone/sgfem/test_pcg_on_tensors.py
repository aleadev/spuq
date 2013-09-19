import logging

import numpy as np
import scipy.sparse as sps

from spuq.linalg.cptensor import CPTensor
from spuq.linalg.operator import MultiplicationOperator
from spuq.linalg.tensor_basis import TensorBasis
from spuq.linalg.tensor_operator import TensorOperator
from spuq.linalg.scipy_operator import ScipyOperator
from spuq.linalg.basis import CanonicalBasis
from test_zone.sgfem import pcg

logger = logging.getLogger(__name__)


def generate_matrix(n):
    A = np.random.rand(n, n) - 0.5
    return sps.csr_matrix(np.dot(A, A.T))


def construct_operator(n1, n2, R):
    # prepare "deterministic" matrices
    basis1 = CanonicalBasis(n1)
    basis2 = CanonicalBasis(n2)

    A1 = [ScipyOperator(generate_matrix(n1),
                        domain=basis1, codomain=basis1)
          for _ in range(R)]
    A2 = [ScipyOperator(generate_matrix(n2),
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


def construct_preconditioner(A):
    basis = A.domain
    basis1 = basis[0]
    basis2 = basis[1]

    return TensorOperator(
        [MultiplicationOperator(1, domain=basis1)],
        [MultiplicationOperator(1, domain=basis2)],
        basis)


def rank_truncate(R_max):
    """Create a truncation function that truncates by rank"""

    def do_truncate(X):
        return X.truncate(R_max)

    return do_truncate


def test_solve_pcg(A, P, u, f, **kwargs):
    """Solve the linear problem with given solution and show solve statistics"""
    [u2, _, numiter] = pcg.pcg(A, f, P, 0 * u, **kwargs)

    logger.info("error:  %s", np.linalg.norm(u.flatten().as_array() - u2.flatten().as_array()))
    logger.info("iter:   %s", numiter)
    logger.info("norm_u: %s", np.linalg.norm(u.flatten().as_array()))
    logger.info("norm_f: %s", np.linalg.norm((A * u).flatten().as_array()))

# test tensor operator application
# ================================
np.random.seed(10)

n1 = 100
n2 = 15
R_A = 5
R_u = 4

A = construct_operator(n1, n2, R_A)
P = construct_preconditioner(A)

u = construct_cptensor(n1, n2, R_u)
u_flat = u.flatten()

#pcg.logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

eps = 0.3
test_solve_pcg(A, P, u_flat, A * u_flat, eps=eps)
test_solve_pcg(A, P, u, A * u, truncate_func=rank_truncate(10), eps=eps)
