from spuq.linalg.operator import Operator
import numpy as np

import logging
logger = logging.getLogger(__name__)

def _dim(A):
    try:
        m, n = A.shape
        assert m == n
        return m
    except:
        return A.dim

class TensorOperator(Operator):
    """Tensor operator \sum_i A_i\otimes B_i."""
    
    def __init__(self, A, B, domain=None, codomain=None, reverse_kronecker=True):
        """Initialise with lists of operators."""
        assert len(A) == len(B)
        if not reverse_kronecker:
            self.A, self.B = A, B
        else:
            self.A, self.B = B, A
        self._domain = domain
        self._codomain = codomain
        self._reverse_kronecker = reverse_kronecker
        self.I, self.J, self.M = _dim(A[0]), _dim(B[0]), len(A)

    @property
    def dim(self):
        return self.I, self.J, self.M

    def as_matrix(self):
        """Return matrix of operator if all included operators support this."""
        import itertools as iter
        I, J, M = self.I, self.J, self.M
        AB = np.ndarray((I*J, I*J))
        for m in range(M):
            A, B = self.A[m], self.B[m]
            for xi, yi in iter.product(range(I), repeat=2):
                # create view
                ABij = AB[xi*J:(xi+1)*J, yi*J:(yi+1)*J]
                # add together
                ABij = A[xi,yi]*B.as_matrix() if m == 0 else ABij + A[xi,yi]*B.as_matrix()
        return AB

    def apply(self, vec):  # pragma: no cover
        """Apply operator to vector."""
        for m in range(self.M):
            # apply B to all components of vector
            Bv_m = [B.apply(v) for B, v in zip(self.B, vec)]
            # build outer product with A
            Vm = [ sum([ai*Bv_mi for ai, Bv_mi in zip(self.A[j,:], Bv_m)]) for j in range(self.I) ]
            # add together
            V = Vm if m == 0 else V + Vm
        return V            
    
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
