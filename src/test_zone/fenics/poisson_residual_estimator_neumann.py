"""
Demo for adaptive Poisson using a residual-based energy-norm error
estimator

  eta_h**2 = sum_T eta_T**2

with

  eta_T**2 = h_T**2 ||R_T||_T**2 + h_T ||R_dT||_dT**2 + h_T ||R_NT||_NT**2

where

  R_T = - (f + div grad u_h)
  R_dT = jump(grad u_h * n)
  R_NT = g - grad u_h * n

and a 'maximal' marking strategy (refining those cells for
which the error indicator is greater than a certain fraction of the
largest error indicator)
"""

from dolfin import *
from numpy import array, sqrt

# Error tolerance
tolerance = 0.1
max_iterations = 1

# Refinement fraction
fraction = 0.4

# Create initial mesh
N = 5
mesh = UnitSquareMesh(N, N)
maxx, minx, maxy, miny = 1, 0, 1, 0
# setup boundary parts
top, bottom, left, right = compile_subdomains([  'near(x[1], 1.) && on_boundary',
                                                 'near(x[1], 0.) && on_boundary',
                                                 'near(x[0], 0.) && on_boundary',
                                                 'near(x[0], 1.) && on_boundary'])
top.maxy = maxy
bottom.miny = miny
left.minx = minx
right.maxx = maxx
boundaries = {'top':top, 'bottom':bottom, 'left':left, 'right':right}

# set all but left boundary as Neumann boundary
Neumann_parts = FacetFunction("size_t", mesh, 0)
top.mark(Neumann_parts, 1)
right.mark(Neumann_parts, 2)
bottom.mark(Neumann_parts, 3)
# define boundary measures
ds = Measure("ds")[Neumann_parts]
# define Neumann fluxes
g = (Constant(0.0), Constant(1.0), Constant(0.0))

# adaptation loop
for i in range(max_iterations):

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant("1.0")
    a = inner(grad(v), grad(u)) * dx
    L = v * f * dx
    for j, rj in enumerate(g):
        L += inner(rj, v) * ds(j + 1)

    # setup Dirichlet BC at left boundary
    c0 = Constant(0.0)
    bc = DirichletBC(V, c0, left)

    # Compute solution
    u_h = Function(V)
    solve(a == L, u_h, bc)

    # Define cell and facet residuals
    R_T = -(f + div(grad(u_h)))
    n = FacetNormal(mesh)
    R_dT = dot(grad(u_h), n)

    # Will use space of constants to localize indicator form
    DG = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(DG)
    h = CellSize(mesh)

    # Define form for assembling error indicators
    eta = (h ** 2 * R_T ** 2 * w * dx + avg(h) * avg(inner(R_dT, R_dT)) * 2 * avg(w) * dS)
    # add Neumann terms
    for j, rj in enumerate(g):
        R_NT = rj - inner(grad(u_h), n)
        eta += h * w * inner(R_NT, R_NT) * ds(j)

    # Assemble error indicators
    eta_indicator = assemble(eta)

    # Calculate error
    error_estimate = sqrt(sum(i for i in eta_indicator.array()))
    print "error_estimate = ", error_estimate

    # map DG dofs to cell indices
    dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
    eta_indicator = eta_indicator[dofs]
    # take sqrt of indicators
    indicators = array([sqrt(i) for i in eta_indicator])

    # Stop if error estimates is less than tolerance
    if error_estimate < tolerance:
        print "\nEstimated error < tolerance: %s < %s" % (error_estimate,
                                                        tolerance)
        break

    # Mark cells for refinement based on maximal marking strategy
    largest_error = max(indicators)
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
        cell_markers[c] = indicators[c.index()] > (fraction * largest_error)

    # Refine mesh
    mesh = refine(mesh, cell_markers)

# Plot final solution
plot(u_h)
plot(mesh)

# Hold plots
interactive()
