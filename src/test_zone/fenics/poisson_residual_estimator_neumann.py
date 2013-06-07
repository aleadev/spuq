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
N = 10
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

# define Neumann fluxes
g = (Constant(0.0), Expression("A*sin(3.*pi*x[1])", A=0), Constant(0.0))

# adaptation loop
for i in range(max_iterations):
    # set all but left boundary as Neumann boundary
    Neumann_parts = FacetFunction("size_t", mesh, 0)
    top.mark(Neumann_parts, 1)
    right.mark(Neumann_parts, 2)
    bottom.mark(Neumann_parts, 3)
    # define boundary measures
    ds = Measure("ds")[Neumann_parts]

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant("1.0")
    a = inner(grad(v), grad(u)) * dx
    L = v * f * dx
    for j, gj in enumerate(g):
        L += inner(gj, v) * ds(j + 1)

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
    eta1 = h ** 2 * R_T ** 2 * w * dx
    eta2 = 0 * w * dx + avg(h) * avg(inner(R_dT, R_dT)) * 2 * avg(w) * dS
    eta3 = 0 * w * dx
    eta = (h ** 2 * R_T ** 2 * w * dx + avg(h) * avg(inner(R_dT, R_dT)) * 2 * avg(w) * dS)
    # add Neumann terms
    for j, gj in enumerate(g):
        R_NT = gj - inner(grad(u_h), n)
        eta += h * w * inner(R_NT, R_NT) * ds(j + 1)
        eta3 += h * w * inner(R_NT, R_NT) * ds(j + 1)

    # Assemble error indicators
    eta_indicator = assemble(eta)
    eta_indicator1 = assemble(eta1)
    eta_indicator2 = assemble(eta2)
    eta_indicator3 = assemble(eta3)
#    print "eta1", eta_indicator1.array()
#    print "eta2", eta_indicator2.array()
#    print "eta3", eta_indicator3.array()
    e1 = Function(DG, eta_indicator1)
    e2 = Function(DG, eta_indicator2)
    e3 = Function(DG, eta_indicator3)
    V1 = FunctionSpace(refine(refine(mesh)), 'CG', 1)
    f1 = interpolate(e1, V1)
    f2 = interpolate(e2, V1)
    f3 = interpolate(e3, V1)
    plot(f1, title='volume residual')
    plot(f2, title='edge residual')
    plot(f3, title='Neumann residual')

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
plot(u_h, title='solution')
plot(mesh, title='mesh')

# Hold plots
interactive()
