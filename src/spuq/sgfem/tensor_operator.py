from spuq.linalg.operator import Operator

import logging
logger = logging.getLogger(__name__)

class TensorOperator(Operator):
    """Tensor operator \sum_i A_i\otimes B_i."""
    
    def __init__(self, A, B, domain=None, codomain=None):
        """Initialise with lists of operators."""
        self.A = A
        self.B = B
        self._domain = domain
        self._codomain = codomain

    def as_matrix(self, reverse_kronecker = True):
        """Return matrix of operator if all included operators support this."""
        pass

    def apply(self, vec):  # pragma: no cover
        """Apply operator to vector."""
        pass
    
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
