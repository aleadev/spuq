from scipy.linalg import norm

from spuq.linalg.tensor_operator import TensorOperator
from spuq.linalg.tensor_vector import FullTensor
from spuq.polyquad.structure_coefficients import evaluate_triples
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.application.egsz.sample_problems2 import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.polyquad.polynomials import LegendrePolynomials


# use uBLAS backend for conversion to scipy sparse matrices
from dolfin import parameters
parameters.linear_algebra_backend = "uBLAS"

def prepare_deterministic_operators(pde, coeff, M, mesh, degree):
    fs = pde.function_space(mesh, degree=degree)
    basis = FEniCSBasis(fs)
    am_f = [coeff.mean_func]
    am_f.extend([coeff_field[m][0] for m in range(M)])
    K = [pde.assemble_operator(basis=basis, coeff=f).as_scipy_operator() for f in am_f]
    return K, basis

def prepare_stochastic_operators(M, p1, p2=0, type='L'):
    I = MultiindexSet.createCompleteOrderSet(M, p1).arr
    if type == 'L':
        print "Legendre triple", I.shape
        polysys = LegendrePolynomials()
        L = evaluate_triples(polysys, I, I)
    elif type == 'H':
        assert False
        #J = MultiindexSet.createCompleteOrderSet(M, p2).arr
        #H = evaluate_Hermite_triple(I, I, J)
        #L = [L[:, :, k] for k in range(L.shape[2])]
    return L

def prepare_vectors(J, basis):
    return [basis.new_vector().array for _ in range(J)]


# A setup problem
# ===============

# setup domain and meshes
mesh, boundaries, dim = SampleDomain.setupDomain("square", initial_mesh_N=10)
mesh = SampleProblem.setupMesh(mesh, num_refine=0)

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=2, gamma=0.9, freqscale=1, freqskip=0, rvtype="uniform", scale=1)

# setup boundary conditions and pde
pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(2, "square", 0, boundaries, coeff_field)


# B setup tensor operator
# =======================

# setup deterministic operators
M, degree = 3, 1
K, FS = prepare_deterministic_operators(pde, coeff_field, M, mesh, degree)
print "K", len(K)

K0inv = pde.assemble_solve_operator(basis=FS, coeff=f).as_scipy_operator()

# setup stochastic operators
p1 = 2
D = prepare_stochastic_operators(M, p1, 'L')
print "D", len(D), D[0].domain.dim
print D[0].as_matrix()

# construct combined tensor operator
A = TensorOperator(K, D)
I, J, M = A.dim
print "TensorOperator A dim", A.dim

# setup tensor vector
u = prepare_vectors(J, FS)
u = FullTensor.from_list(u)
print "FullTensor u", u.dim()
print "u as matrix shape", u.as_matrix().shape, u.flatten().shape


# test tensor operator
# ====================

# test application of operator
w = A * u

# print matricisation of tensor operator
M = A.as_matrix()
print M.shape, norm(M)

# plot mesh
#import dolfin
#dolfin.plot(mesh, interactive=True)

# plot sparsity pattern
#fig = figure()
##spy(M)
#spy(K[0].as_matrix())
#show()

from spuq.application.egsz.pcg import pcg
P = TensorOperator([K0inv], [D[0]])
print 0*w
b = w.copy()
b.X += 1
v = pcg(A, b, P, 0*w )

print "v", v



if False:
    import numpy as np
    import scipy as sp
    import scipy.sparse as spa
    
    
    A=spa.csr_matrix(np.random.rand(7,3))
    B=np.random.rand(3,5)
    print A
    print B
    print A*B
    print A*B - (B.T * A.T).T
    #print np.array(sp.dot(A,B))
    print sp.dot(A,spa.csr_matrix(B)).toarray()

