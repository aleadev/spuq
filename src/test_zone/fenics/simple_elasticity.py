from dolfin import *

# Create mesh
N = 30
mesh = UnitSquare(N, N)
# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 2)
# Create test and trial functions, and source term
u, w = TrialFunction(V), TestFunction(V)
b = Constant((0.0, 0.0))
# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress
sigma = 2*mu*sym(grad(u)) + lmbda*tr(grad(u))*Identity(w.cell().d)
# Governing balance equation
F = inner(sigma, grad(w))*dx - dot(b, w)*dx
# Extract bilinear and linear forms from F
a, L = lhs(F), rhs(F)

# Dirichlet boundary condition on entire boundary
def uD_boundary1(x, on_boundary):
    return abs(x[0]) <= 0 and on_boundary
def uD_boundary2(x, on_boundary):
    return abs(x[0]) >= 1 and on_boundary

bc1 = DirichletBC(V, Constant((0,0)), uD_boundary1)
bc2 = DirichletBC(V, Constant((0,0.3)), uD_boundary2)

# Set up PDE and solve
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bcs=[bc1,bc2])
solver = LinearVariationalSolver(problem)
solver.parameters["symmetric"] = True
solver.solve()

plot(u,mode="displacement",interactive=True,wireframe=True)
