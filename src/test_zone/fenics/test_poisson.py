"""
Poisson equation with Dirichlet conditions.

-Laplace(u) = f on the unit square.
u = x*(x-1)*y*(y-1)
u = u0 = 0 on the boundary.
f = 2*x*(x-1) + 2*y*(y-1).
"""

from dolfin import *

# create mesh and define function space
mesh = UnitSquare(100, 100)
V = FunctionSpace(mesh, 'CG', 1)

# define boundary conditions
u0 = Constant(0.0)

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Expression('1')
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# compute solution
A = assemble(a)
b = assemble(L)
bc.apply(A, b)
u = Function(V)
solve(A, u.vector(), b)

# plot solution and mesh
plot(u)
plot(mesh)
interactive()

# assemble mass matrix and get numpy vector
M = assemble((u * v) * dx)
npM = M.array()
# print 'shape of mass matrix M is ', npM.shape

# interpolation and projection test
mesh2 = UnitSquare(40, 40)
V2 = FunctionSpace(mesh2, 'CG', 2)
f2 = Expression('sin(4*3.141*x[0]*x[1])+x[0]*x[1]')
f2_V2 = interpolate(f2, V2)
pf2_V1 = project(f2_V2, V)

plot(f2_V2)
plot(pf2_V1)
interactive()
