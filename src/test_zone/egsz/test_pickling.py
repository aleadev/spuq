from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet

from dolfin import UnitSquare, FunctionSpace, Function, refine, Expression, plot, interactive
from math import pi
import pickle

mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(3, 1)]
N = len(mis)
meshes = [UnitSquare(10 + 2 * i, 10 + 2 * i) for i in range(N)]
functions = [Function(FunctionSpace(m, "CG", d + 1)) for d, m in enumerate(meshes)]
ex = Expression('sin(A*x[0])*sin(A*x[1])', A=1)
for i, f in enumerate(functions):
    ex.A = 2 * pi * (i + 1)
    f.interpolate(ex)

# test FEniCSVector
# =================
vectors = [FEniCSVector(f) for f in functions]
with open('test-vector.pkl', 'wb') as f:
    pickle.dump(vectors[1], f)

with open('test-vector.pkl', "rb") as f:
    v1 = pickle.load(f)


# test MultiVector
# ================
#mv1 = MultiVector()
mv1 = MultiVectorWithProjection()
for mu, v in zip(mis, vectors):
    mv1[mu] = v
    
with open('test-multivector.pkl', 'wb') as f:
    pickle.dump(mv1, f)

with open('test-multivector.pkl', "rb") as f:
    mv2 = pickle.load(f)

for i, mu in enumerate(mv2.active_indices()):
    plot(mv2[mu]._fefunc, title="vector " + str(mu))
interactive()
