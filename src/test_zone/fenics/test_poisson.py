"""
Poisson equation with Dirichlet conditions.

-Laplace(u) = f on the unit square.
u = x*(x-1)*y*(y-1)
u = u0 = 0 on the boundary.
f = 2*x*(x-1) + 2*y*(y-1).
"""

from dolfin import *

# create mesh and define function space
mesh1 = UnitSquare(5, 5)
V1 = FunctionSpace(mesh1, 'CG', 1)
mesh2 = refine(mesh1)
V2 = FunctionSpace(mesh2, 'CG', 1)

# define boundary conditions
u0 = Constant(0.0)

def u0_boundary(x, on_boundary):
    return (x[1] <= 0 or x[1] >= 1) and on_boundary

bc1 = DirichletBC(V1, u0, u0_boundary)
bc2 = DirichletBC(V2, u0, u0_boundary)

# define variational problem
v1 = TestFunction(V1)
u1 = TrialFunction(V1)
v2 = TestFunction(V2)
u2 = TrialFunction(V2)
f = Expression('1')
a1 = inner(grad(u1), grad(v1)) * dx
L1 = f * v1 * dx
a2 = inner(grad(u2), grad(v2)) * dx
L2 = f * v2 * dx

# compute solution
A1 = assemble(a1)
b1 = assemble(L1)
bc1.apply(A1, b1)
u1 = Function(V1)
solve(A1, u1.vector(), b1)

A2 = assemble(a2)
b2 = assemble(L2)
bc2.apply(A2, b2)
u2 = Function(V2)
solve(A2, u2.vector(), b2)

# plot solution and mesh
plot(u1, title="u1")
plot(mesh1, title="mesh1")
plot(u2, title="u2")
plot(mesh2, title="mesh2")
#u12 = interpolate(u1, V2)
#err = Function(V2)
#err.vector()[:] = u12.vector() - u2.vector()
#print "H1 err =", norm(err, "H1")
#plot(err, title="err")
interactive()


if False:
    # assemble mass matrix and get numpy vector
    M = assemble((u1 * v1) * dx)
    npM = M.array()
    # print 'shape of mass matrix M is ', npM.shape
    
    # interpolation and projection test
    mesh2 = UnitSquare(40, 40)
    V2 = FunctionSpace(mesh2, 'CG', 2)
    f2 = Expression('sin(4*3.141*x[0]*x[1])+x[0]*x[1]')
    f2_V2 = interpolate(f2, V2)
    pf2_V1 = project(f2_V2, V1)
    
    #plot(f2_V2)
    #plot(pf2_V1)
    #interactive()
