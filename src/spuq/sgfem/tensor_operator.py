from spuq.linalg.operator import Operator

import logging
logger = logging.getLogger(__name__)

class TensorOperator(Operator):
    def __init__(self, A):
        """Initialise with single operator or list of operators."""
        pass

    def as_matrix(self):
        """Return matrix of operator if all included operators support this."""
        pass

    def apply(self, vec):  # pragma: no cover
        """Apply operator to vector."""
        raise NotImplementedException
    
    def __call__(self, arg):
        """Operators have call semantics."""
        return self.apply(arg)

    # TODO: multiplication with scalar and operator
    
