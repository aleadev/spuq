import numpy as np

from spuq.utils.testing import *

from spuq.application.egsz.pcg import pcg
from spuq.linalg.operator import MatrixOperator, MatrixSolveOperator, MultiplicationOperator, DiagonalMatrixOperator
from spuq.linalg.vector import FlatVector, inner
from spuq.linalg.basis import CanonicalBasis


def test_pcg_matrix():
    rand = np.random.mtrand.RandomState(1234).random_sample

    N = 7
    M = rand((N, N))
    A = MatrixOperator(np.dot(M, M.T))
    P = MultiplicationOperator(1, CanonicalBasis(N))
    x = FlatVector(rand((N,)))
    b = A * x
    print

    # solve with identity as preconditioner
    x_ap, zeta, iter = pcg(A, b, P, 0 * x, eps=0.00001)
    assert_array_almost_equal(x.coeffs, x_ap.coeffs)
    assert_true( iter <= N + 3)

    # solve with exact matrix inverse as preconditioner
    # should converge in one step
    P = MatrixSolveOperator(np.dot(M, M.T))
    x_ap, zeta, iter = pcg(A, b, P, 0 * x, eps=0.00001)
    assert_array_almost_equal(x.coeffs, x_ap.coeffs)
    assert_true( iter <= 2)


    P = MatrixOperator(P.as_matrix())
    x_ap, zeta, iter = pcg(A, b, P, 0 * x, eps=0.00001)
    #print iter
    assert_array_almost_equal(x.coeffs, x_ap.coeffs)

    P = DiagonalMatrixOperator(np.diag(A.as_matrix())).inverse()
    x_ap, zeta, iter = pcg(A, b, P, 0 * x, eps=0.00001)
    #print iter
    assert_array_almost_equal(x.coeffs, x_ap.coeffs)

    #print x
    #print x_ap


test_main()
