import numpy as np

from spuq.utils.testing import *

from spuq.application.egsz.pcg import pcg
from spuq.linalg.operator import MatrixOperator, MultiplicationOperator
from spuq.linalg.vector import FlatVector
from spuq.linalg.basis import CanonicalBasis


def test_pcg_matrix():
    N = 7
    M = np.random.random((N, N))
    A = MatrixOperator(np.dot(M, M.T))
    P = MultiplicationOperator(1, CanonicalBasis(N))
    x = FlatVector(np.random.random((N,)))
    b = A * x
    x_ap, zeta, iter = pcg(A, b, P, 0 * x, eps=0.00001)
    print iter
    assert_array_almost_equal(x.coeffs, x_ap.coeffs)

    #print x
    #print x_ap


test_main()
