"""
Demo for adaptive Poisson using a residual-based energy-norm error
estimator

  eta_h**2 = sum_T eta_T**2

with

  eta_T**2 = h_T**2 ||R_T||_T**2 + h_T ||R_dT||_dT**2

where

  R_T = - (f + div grad u_h)
  R_dT = jump(grad u_h * n)

and a 'maximal' marking strategy (refining those cells for
which the error indicator is greater than a certain fraction of the
largest error indicator)
"""

from dolfin import *
from numpy import array, sqrt

# Error tolerance
tolerance = 0.1
max_iterations = 100

# Refinement fraction
fraction = 0.5

# Create initial mesh
#mesh = UnitSquare(4, 4)
mesh = Mesh("lshape.xml")

for i in range(max_iterations):

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
#    f = Expression("10*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)",
#                   degree=3)
    f = Constant("1.0")
    a = inner(grad(v), grad(u)) * dx
    L = v * f * dx

    # Define boundary condition
#    bc = DirichletBC(V, 0.0, "near(x[0], 0.0) || near(x[0], 1.0)")
    def u0_boundary(x, on_boundary):
        return on_boundary
    #    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, u0_boundary)

    # Compute solution
    u_h = Function(V)
    solve(a == L, u_h, bc)

    # Define cell and facet residuals
    R_T = -(f + div(grad(u_h)))
    n = FacetNormal(mesh)
    R_dT = dot(grad(u_h), n)

    # Will use space of constants to localize indicator form
    Constants = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(Constants)
    h = CellSize(mesh)

    # Define form for assembling error indicators
    form = (h ** 2 * R_T ** 2 * w * dx + avg(h) * avg(R_dT) ** 2 * 2 * avg(w) * dS)
#            + h * R_dT ** 2 * w * ds)

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
        cell_markers[c] = indicators[c.index()] > (fraction * largest_error)

    # Refine mesh
    mesh = refine(mesh, cell_markers)

# Plot final solution
plot(u_h)
plot(mesh)

# Hold plots
interactive()
