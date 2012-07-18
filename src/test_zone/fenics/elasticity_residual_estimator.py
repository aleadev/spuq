"""
Demo for adaptive linear elasticity using a residual-based energy-norm error
estimator

  eta_h**2 = sum_T eta_T**2

with

  eta_T**2 = h_T**2 ||R_T||_T**2 + h_T ||R_dT||_dT**2

where

  R_T = - (f + sigma(u_h))
  R_dT = jump(sigma(u_h) * n)

and

  sigma(v) = C\eps(v) = 2\mu\eps(v) + \lamda tr(\eps(v))I

and a 'maximal' marking strategy (refining those cells for
which the error indicator is greater than a certain fraction of the
largest error indicator)
"""

from dolfin import *
from numpy import array, sqrt

# error tolerance
tolerance = 0.1
max_iterations = 15

# set Neumann boundary condition on right edge
with_Neumann = True

# refinement fraction
Theta = 0.4

# create initial mesh
#mesh = UnitSquare(5, 5)
mesh = Mesh("lshape.xml")
maxx = 1
minx = -1
maxy = 1
miny = -1


## sub domain for clamp at left end
#def left(x, on_boundary):
#    return x[0] <= -1.0 and on_boundary
#
## sub domain for rotation at right end
#def right(x, on_boundary):
#    return x[0] >= 1.0 and on_boundary

llc, lrc, tlc, trc, top, bottom, left, right = compile_subdomains(['near(x[0], minx) && near(x[1], miny)',
                                                 'near(x[0], maxx) && near(x[1], miny)',
                                                 'near(x[0], minx) && near(x[1], maxy)',
                                                 'near(x[0], maxx) && near(x[1], maxy)',
                                                 'near(x[1], maxy) && on_boundary',
                                                 'near(x[1], miny) && on_boundary',
                                                 'near(x[0], minx) && on_boundary',
                                                 'near(x[0], maxx) && on_boundary'])
# the corners
llc.minx = minx
llc.miny = miny
lrc.maxx = maxx
lrc.miny = miny
tlc.minx = minx
tlc.maxy = maxy
trc.maxx = maxx
trc.maxy = maxy
# the edges
top.maxy = maxy
bottom.miny = miny
left.minx = minx
right.maxx = maxx

# Dirichlet boundary condition
def default_force(values, x):
    values[0], values[1] = 0.0, 0.0
 
class BCForce(Expression):
    def __init__(self, force=default_force):
        self.force = force
    def eval(self, values, x):
        self.force(values, x)
    def value_shape(self):
        return (2,)


# =============================
# =============================

E = 1e5
nu = 0.4    
mu = E / (2.0 * (1.0 + nu))
#lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
#print "mu, lambda =", mu, lmbda
import ufl
lmbda = Expression("100000.*(0.6+0.5*sin(2.*(x[0]+x[1])))", element=FiniteElement('Lagrange', ufl.triangle, 1))

for i in range(max_iterations):
    # Define variational problem
    V = VectorFunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant((0.0, 0.0))

    # mark Neumann boundary
    if with_Neumann:
#        Neumann_parts = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
        Neumann_parts = FacetFunction("uint", mesh, mesh.topology().dim() - 1)
        Neumann_parts.set_all(0)
        right.mark(Neumann_parts, 1)

    # define sigma    
    sigma = lambda v: 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(v.cell().d)

    # define bilinear and linear forms
    a = inner(sigma(u), sym(grad(v))) * dx
    L = inner(f, v) * dx
    if with_Neumann:
        # define boundary measures
        ds = Measure("ds")[Neumann_parts]
        # shear load (Neumann boundary)
        g = Constant((0.0, 1000.0))
#        g = Constant((0.0, 1.0))
        L -= dot(g, v) * ds(1)
    
    # setup boundary condition at left end
    l = Constant((0.0, 0.0))
    bcl = DirichletBC(V, l, left)
    bcs = [bcl]
    
#    # (set corner vertices) note use of Vector DirichletBC and "pointwise"
#    bc_tlc = DirichletBC(V, l, trc, "pointwise")
#    bc_llc = DirichletBC(V, l, llc, "pointwise")
##    bc_tlc = DirichletBC(V.sub(0), Constant(0.0), trc, "pointwise")
##    bc_llc = DirichletBC(V.sub(0), Constant(0.0), llc, "pointwise")
#    bcs = [bcl, bc_tlc, bc_llc]

    # setup boundary condition at right end
    if not with_Neumann:
        r = Constant((-0.1, 0.5))
        bcr = DirichletBC(V, r, right)
        bcs.append(bcr)

    # Compute solution
    u_h = Function(V)
#    b = assemble(L, exterior_facet_domains=Neumann_parts)
#    solve(a == b, u_h, bcs)
    solve(a == L, u_h, bcs)
    
    # Define cell and facet residuals
    n = FacetNormal(mesh)
    R_T = -(f + div(sigma(u_h)))
    R_dT = dot(sigma(u_h), n)

    # Will use space of constants to localize indicator form
    Constants = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(Constants)
    h = CellSize(mesh)

    # Define form for assembling error indicators
    form = h ** 2 * dot(R_T, R_T) * w * dx + avg(h) * dot(avg(R_dT), avg(R_dT)) * 2 * avg(w) * dS
    if with_Neumann:
        sigmanerr = g + dot(sigma(u_h), n)
        R_NdT = dot(sigmanerr, sigmanerr) 
        form += h * R_NdT * w * ds(1)

    # Assemble error indicators
    indicators = assemble(form)

    # Calculate error
    error_estimate = sqrt(sum(i for i in indicators.array()))
    print "error_estimate = ", error_estimate

    # Take sqrt of indicators
    indicators = array([sqrt(i) for i in indicators])

    # Stop if error estimates is less than tolerance
    if error_estimate < tolerance:
        print "\nEstimated error < tolerance: %s < %s" % (error_estimate,
                                                        tolerance)
        break

    # Mark cells for refinement based on maximal marking strategy
    largest_error = max(indicators)
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
        cell_markers[c] = indicators[c.index()] > (Theta * largest_error)

    # Refine mesh
    mesh = refine(mesh, cell_markers)

# Plot final solution
#plot(u_h, mode="displacement", mesh=mesh, wireframe=False, rescale=False, axes=True)
plot(u_h, mode="displacement", mesh=mesh, wireframe=True, rescale=False, axes=True)
plot(mesh)

# Hold plots
interactive()
