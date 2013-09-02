from spuq.linalg.vector import Vector, Scalar
import numpy as np

import logging
logger = logging.getLogger(__name__)

class TensorVector(Vector):
    def __init__(self, V):
        """Initialise with list of vectors."""
        self.V = V
        self.M = len(V)
        self.I = V[0].dim
    
    def dim(self):  # pragma: no cover
        """Return dimension of this vector"""
        return self.M*self.I

    def copy(self):  # pragma: no cover
        raise NotImplementedError

    def flatten(self):
        return np.hstack([v.array for v in self.V])

    def as_matrix(self):
        return np.vstack([v.array for v in self.V]).T

    def basis(self):
        return [v.basis for v in self.V]

    def __getitem__(self, i):
        return self.V[i]
    
    def __setitem__(self, i, val):
        self.V[i] = val

    def __len__(self):
        return len(self.V)
    
#     def transpose(self):
#         pass
# 
    def __eq__(self, other):  # pragma: no cover
        """Test whether vectors are equal."""
        return np.all([v1==v2 for v1, v2 in zip(self.V, other.V)])
 
    def __neg__(self, other):  # pragma: no cover
        """Compute the negative of this vector."""
        return TensorVector([-v for v in self.V])
 
    def __iadd__(self, other):  # pragma: no cover
        """Add another vector to this one."""
        for i in range(self.M):
            self.V[i] += other[i]
 
    def __imul__(self, other):  # pragma: no cover
        """Multiply this vector with a scalar."""
        for i in range(self.M):
            if isinstance(other, Scalar):
                self.V[i] *= other
            else:
                self.V[i] *= other[i]
 
#     def __inner__(self, other):  # pragma: no cover
#         """Scalar product of this vector with another vector."""
#         return NotImplemented
# 
#     def __rinner__(self, other):  # pragma: no cover
#         """Scalar product of this vector with another vector (reverse)."""
#         return NotImplemented
# 
#     def __add__(self, other):
#         """Add two vectors."""
#         return self.copy().__iadd__(other)
# 
#     def __radd__(self, other):  # pragma: no cover
#         """This happens only when other is not a vector."""
#         if isinstance(other, Scalar) and other == 0:
#             return self
#         return NotImplemented
# 
#     def __sub__(self, other):
#         """Subtract two vectors."""
#         if hasattr(self, "__isub__"):
#             return self.copy().__isub__(other)
#         return self +(-other)
# 
#     def __rsub__(self, other):  # pragma: no cover
#         """This happens only when other is not a vector."""
#         return NotImplemented
# 
#     def __mul__(self, other):
#         """Multiply a vector with a scalar from the right."""
#         if isinstance(other, Scalar):
#             return self.copy().__imul__(other)
#         return NotImplemented
# 
#     def __rmul__(self, other):
#         """Multiply a vector with a scalar from the left."""
#         if isinstance(other, Scalar):
#             return self.copy().__imul__(other)
#         return NotImplemented
# 
#     def __repr__(self):
#         return "<%s basis=%s, coeffs=%s>" % \
#                (strclass(self.__class__), self.basis, self.coeffs)
