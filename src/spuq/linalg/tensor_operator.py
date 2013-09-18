import logging
import numpy as np

from spuq.linalg.operator import Operator, ComponentOperator
from spuq.utils.type_check import takes, anything, sequence_of
from spuq.linalg.tensor_vector import MatrixTensorVector

logger = logging.getLogger(__name__)

def _dim(A):
    try:
        m = A.shape[0]
        assert m == A.shape[1]
        return m
    except:
        return A.dim


class TensorOperator(Operator):
    """Tensor operator \sum_i A_i\otimes B_i."""
    #TODO: derive from BaseOperator

    @takes(anything, sequence_of(ComponentOperator), sequence_of(ComponentOperator))
    def __init__(self, A, B, domain=None, codomain=None):
        """Initialise with lists of operators."""
        assert len(A) == len(B)
        self.A, self.B = A, B
        self._domain = domain
        self._codomain = codomain
        self.I, self.J, self.M = self.A[0].domain.dim, self.B[0].domain.dim, len(A)

    @property
    def dim(self):
        return self.I, self.J, self.M

    def as_matrix(self):
        """Return matrix of operator if all included operators support this."""
        import itertools as iter
        I, J = self.I, self.J
        AB = np.ndarray((I*J, I*J))
        # TODO: contruct large matrix in i,j,value format
        for m in range(self.M):
            A = self.A[m].as_matrix().toarray()
            B = self.B[m].as_matrix().toarray()
            for xi, yi in iter.product(range(self.I), repeat=2):
                # add together
#                import ipdb; ipdb.set_trace()
                if m == 0:
                    AB[xi * J:(xi + 1) * J, yi * J:(yi + 1) * J] = A[xi, yi] * B
                else:
                    AB[xi * J:(xi + 1) * J, yi * J:(yi + 1) * J] += A[xi, yi] * B
        return AB

    def old_apply(self, vec):
        X = vec#.as_matrix()
        for m in range(self.M):
            # apply B
            BX = self.B[m].apply(X.T)
            # apply A
            AXB = self.A[m].apply(BX.T)
            # add together
            Y = AXB if m == 0 else Y + AXB
        return vec.__class__(Y)

    @takes(anything, MatrixTensorVector)
    def apply(self, vec):  # pragma: no cover
        """Apply operator to vector."""
        X = vec
        for m in range(self.M):
            # apply A
            AX = X.apply_to_dim(self.A[m], 0)
            # apply B
            ABX = AX.apply_to_dim(self.B[m], 1)
            # add together
            if m == 0:
                Y = ABX
            else:
                Y += ABX
        return Y


    def __call__(self, arg):
        """Operators have call semantics."""
        return self.apply(arg)
    
    @property
    def domain(self):
        """Return the basis of the domain."""
        return self._domain

    @property
    def codomain(self):
        """Return the basis of the codomain."""
        return self._codomain
