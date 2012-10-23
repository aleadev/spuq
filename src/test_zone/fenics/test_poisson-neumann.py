"""
Poisson equation with Dirichlet conditions.

-Laplace(u) = f on the unit square.
u = x*(x-1)*y*(y-1)
u = u0 = 0 on the boundary.
f = 2*x*(x-1) + 2*y*(y-1).
"""

from dolfin import *

# create mesh and define function space
N = 20
dim = 2
if dim == 1:
    mesh = UnitInterval(N)
#    mesh2 = UnitInterval(2 * N)
elif dim == 2:
    mesh = UnitSquare(N, N)
#    mesh2 = UnitSquare(2 * N, 2 * N)
else:
    assert not "unsupported dimension"
mesh2 = refine(mesh)
V = FunctionSpace(mesh, 'CG', 1)

# define boundary conditions
u0 = Constant(0.0)

tol = 1e-14
def u0_boundary(x, on_boundary):
    return abs(x[0]) <= tol and on_boundary
bc = DirichletBC(V, u0, u0_boundary)

# define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Constant(1)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# add Neumann part
bottom, top = compile_subdomains(['near(x[1], 0.) && on_boundary',
                                  'near(x[1], 1.) && on_boundary'])
g_parts = FacetFunction("uint", mesh, mesh.topology().dim() - 1)
g_parts.set_all(0)
bottom.mark(g_parts, 3)
top.mark(g_parts, 33)
# evaluate boundary flux terms
ds = Measure("ds")[g_parts]
L += Constant(-1) * v * ds(3) + Expression("sin(pi*2.*x[0])") * v * ds(33)

# compute solution
A = assemble(a)
b = assemble(L)
bc.apply(A, b)
u = Function(V)
solve(A, u.vector(), b)

# plot solution and mesh
plot(u, title="u")
plot(mesh, title="mesh")
interactive()
