
from spuq.sgfem.tensor_operator import TensorOperator
from spuq.sgfem.tensor_vector import TensorVector
from spuq.sgfem.hermite_triple import evaluate_Hermite_triple
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.application.egsz.sample_problems2 import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.fem_discretisation import FEMPoisson

def prepare_deterministic_operators(pde, coeff, M):
    return [pde.assemble_operator(coeff=coeff[m]) for m in range(M)]

def prepare_stochastic_operators(N, p):
    I = MultiindexSet.createCompleteOrderSet(N, p).arr
    return evaluate_Hermite_triple(I, I, I)

def prepare_vectors():
    pass


# A setup problem
# ===============

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(0, 1)]

# setup domain and meshes
mesh0, boundaries, dim = SampleDomain.setupDomain("square", initial_mesh_N=5)
mesh0 = SampleProblem.setupMesh(mesh0, num_refine=0)

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=2, gamma=0.9, freqscale=1, freqskip=0, rvtype="uniform", scale=1)

# setup boundary conditions and pde
pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, "square", CONF_problem_type, boundaries, coeff_field)


# B setup tensor operator
# =======================
    
# prepare coefficient field

# setup deterministic operators

# setup stochastic operators

# construct combined tensor operator

# setup tensor vector


# test tensor operator
# ====================


# test application of operator

# print matricisation of tensor operator

