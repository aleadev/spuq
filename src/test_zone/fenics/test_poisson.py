"""
Poisson equation with Dirichlet conditions.

-Laplace(u) = f on the unit square.
u = x*(x-1)*y*(y-1)
u = u0 = 0 on the boundary.
f = 2*x*(x-1) + 2*y*(y-1).
"""

from dolfin import *

# create mesh and define function space
N = 5
dim = 1
if dim == 1:
    mesh1 = UnitInterval(N)
#    mesh2 = UnitInterval(2 * N)
elif dim == 2:
    mesh1 = UnitSquare(N, N)
#    mesh2 = UnitSquare(2 * N, 2 * N)
else:
    assert not "unsupported dimension"
mesh2 = refine(mesh1)
V1 = FunctionSpace(mesh1, 'CG', 1)
V2 = FunctionSpace(mesh2, 'CG', 1)
mesh3 = refine(mesh1)
V3 = FunctionSpace(mesh3, 'CG', 1)

# define boundary conditions
u0 = Constant(0.0)

tol = 1e-14
if dim == 1:
    def u0_boundary(x, on_boundary):
        return (abs(x[0]) <= tol or abs(x[0] - 1) <= tol) and on_boundary
elif dim == 2:
    def u0_boundary(x, on_boundary):
        return (abs(x[1]) <= tol or abs(x[1] - 1) <= tol) and on_boundary

bc1 = DirichletBC(V1, u0, u0_boundary)
bc2 = DirichletBC(V2, u0, u0_boundary)
bc3 = DirichletBC(V3, u0, u0_boundary)

# define variational problem
v1 = TestFunction(V1)
u1 = TrialFunction(V1)
v2 = TestFunction(V2)
u2 = TrialFunction(V2)
v3 = TestFunction(V3)
u3 = TrialFunction(V3)
f = Constant(1)
a1 = inner(grad(u1), grad(v1)) * dx
L1 = f * v1 * dx
a2 = inner(grad(u2), grad(v2)) * dx
L2 = f * v2 * dx
a3 = inner(grad(u3), grad(v3)) * dx
L3 = f * v3 * dx

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

A3 = assemble(a3)
b3 = assemble(L3)
bc3.apply(A2, b3)
u3 = Function(V3)
solve(A3, u3.vector(), b3)

# plot solution and mesh
plot(u1, title="u1")
plot(u2, title="u2")
plot(u3, title="u3")
if dim > 1: 
    plot(mesh1, title="mesh1")
    plot(mesh2, title="mesh2")
#u12 = interpolate(u1, V2)
#err = Function(V2)
#err.vector()[:] = u12.vector() - u2.vector()
#print "H1 err =", norm(err, "H1")
#plot(err, title="err")
interactive()


if False and dim == 2:
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
