from spuq.sgfem.tensor_operator import TensorOperator
from spuq.sgfem.tensor_vector import TensorVector
from spuq.sgfem.hermite_triple import evaluate_Hermite_triple
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.application.egsz.sample_problems2 import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.fem_discretisation import FEMPoisson
from spuq.fem.fenics.fenics_basis import FEniCSBasis

def prepare_deterministic_operators(pde, coeff, M, mesh, degree):
    fs = pde.function_space(mesh, degree=degree)
    FS = FEniCSBasis(fs)
    am_f = [coeff.mean_func]
    am_f.extend([coeff_field[m][0] for m in range(M-1)])
    return [pde.assemble_operator(basis=FS, coeff=f) for f in am_f]

def prepare_stochastic_operators(N, p):
    I = MultiindexSet.createCompleteOrderSet(N, p).arr
    H = evaluate_Hermite_triple(I, I, I)
    print H.shape
    return [H[:,:,k] for k in range(H.shape[2])]

def prepare_vectors():
    pass

# def setupMultiVector(cls, mis, pde, mesh, degree):
#     fs = pde.function_space(mesh, degree=degree)
#     w = MultiVectorSharedBasis()
#     for mu in mis:
#         w[mu] = FEniCSVector(Function(fs))
#     return w


# A setup problem
# ===============

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(0, 1)]

# setup domain and meshes
mesh, boundaries, dim = SampleDomain.setupDomain("square", initial_mesh_N=5)
mesh = SampleProblem.setupMesh(mesh, num_refine=0)

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=2, gamma=0.9, freqscale=1, freqskip=0, rvtype="uniform", scale=1)

# setup boundary conditions and pde
pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(2, "square", 0, boundaries, coeff_field)


# B setup tensor operator
# =======================

# setup deterministic operators
M, degree = 5, 1
K = prepare_deterministic_operators(pde, coeff_field, M, mesh, degree)
print len(K)

# setup stochastic operators
N, p = 5, 2
D = prepare_stochastic_operators(M, p)
print len(D)

# construct combined tensor operator
A = TensorOperator(K, D)

# setup tensor vector
u = prepare_vectors(A.size)
u = TensorVector(u)


# test tensor operator
# ====================

# test application of operator
w = A*u

# print matricisation of tensor operator
M = A.as_matrix()
print A
