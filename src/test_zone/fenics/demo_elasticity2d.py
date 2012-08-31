"""This demo program solves the equations of static linear elasticity."""

from dolfin import *

# define mesh and vector function space
mesh = UnitSquare(20, 20)
degree = 1
V = VectorFunctionSpace(mesh, "CG", degree)

# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0] <= 0.0 and on_boundary

# Dirichlet boundary condition
def default_force(values, x):
    values[0], values[1] = -0.3, 0.0
 
class BCForce(Expression):
    def __init__(self, force=default_force):
        self.force = force
        
    def eval(self, values, x):
        self.force(values, x)

    def value_shape(self):
        return (2,)

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0] >= 1.0 and on_boundary

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0.0, 0.0))

E = 1e5
nu = 0.4
 # + Expression("0.249*sin(A*pi*x[0])", degree=5, A=4)

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
print "mu=%s, lambda=%s" % (mu, lmbda)
sigma = lambda v: 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(v.cell().d)

a = inner(sigma(u), grad(v)) * dx
L = inner(f, v) * dx

# Set up boundary condition at left end
c = Constant((0.0, 0.0))
bcl = DirichletBC(V, c, left)

# Set up boundary condition at right end
r = BCForce()
bcr = DirichletBC(V, r, right)

# Set up boundary conditions
bcs = [bcl, bcr]

if False:
    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={"symmetric": True})

    # Save solution to VTK format
    File("elasticity2d.pvd", "compressed") << u

    # Save colored mesh partitions in VTK format if running in parallel
    if MPI.num_processes() > 1:
        File("partitions2d.pvd") << CellFunction("uint", mesh, MPI.process_number())

    # Plot solution
    #plot(u, rescale=False, axes=True)
    plot(u, mode="displacement", mesh=mesh, wireframe=True, rescale=False, axes=True, interactive=True)



import numpy as np
import scipy.linalg as la

def msym(M):
    return 0.5*(M+M.T)

A = assemble(a)
b = assemble(L)
bcl.apply(A)
bcr.apply(A)
bcl.apply(b)
bcr.apply(b)
x1 = 0*b
solve(A, x1, b)

M=A.array()
#print M
#print type(M)
print la.norm(M-M.T), min(la.eigvals(msym(M)))



A, b = assemble_system(a, L, [bcl, bcr])
x2 = 0*b
solve(A, x2, b)

M=A.array()
#print M
#print type(M)
print la.norm(M-M.T), min(la.eigvals(msym(M)))


print la.norm(x1.array()-x2.array())



A, _ = assemble_system(a, L, [bcl, bcr])
_, b = assemble_system(a, L, [bcl, bcr])
x2 = 0*b
solve(A, x2, b)

M=A.array()
#print M
#print type(M)
print la.norm(M-M.T), min(la.eigvals(msym(M)))


print la.norm(x1.array()-x2.array())
