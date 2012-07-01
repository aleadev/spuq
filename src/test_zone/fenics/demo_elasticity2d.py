"""This demo program solves the equations of static linear elasticity."""

from dolfin import *

# Load mesh and define function space
mesh = UnitSquare(10, 10)
degree = 1
V = VectorFunctionSpace(mesh, "CG", degree)

# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0] <= 0.0 and on_boundary

# Dirichlet boundary condition
def default_force(values, x):
    values[0], values[1] = 0.0, 1
 
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

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
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
