import numpy as np

from spuq.utils.testing import *

from spuq.application.egsz.pcg import pcg
from spuq.linalg.operator import MatrixOperator
from spuq.linalg.vector import FlatVector

def test_pcg_matrix():
    A = MatrixOperator(np.random.random((4, 4)))
    x = FlatVector(np.random.random((4,)))
    b = A * x
    print A
    print x, b
    x_ap, foo = pcg(A, b, A, 0*x)
    print x_ap

