import numpy as np

from spuq.fem.fem_basis import FEMBasis
from spuq.fem.fem_vector import FEMVector
from spuq.fem.fenics.fenics_mesh import FEniCSMesh

b0 = FEMBasis( FEniCSMesh() )
coeffs = np.zeros( 123 )
v0 = FEMVector( coeffs, b0 )
faces = [1,2,4] # marking_strategy( foo )
(b1, prol, rest) = b0.refine( faces )
v1 = prol( v0 )
assert v1.get_basis() == b1
assert v1.__class__ == v2.__class__
