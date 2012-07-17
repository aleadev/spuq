# source: https://answers.launchpad.net/dolfin/+question/179257
# http://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_equation
from dolfin import *

xlength = 100.0 #[mm]
ylength = 2.0 #[mm]
xelements = 500
yelements = 10
mesh = Rectangle(0, 0, xlength, ylength, xelements, yelements)

llc, lrc, top = compile_subdomains(['near(x[0], 0.0) && near(x[1], 0.0)', \
                                    'near(x[0], length) && near(x[1], 0.0)',
                                    'near(x[1], length) && on_boundary'])

lrc.length = xlength
top.length = ylength

facet_domains = FacetFunction("uint", mesh, 0)
top.mark(facet_domains, 3)

tr = Expression((('X1', 'X2')), X1=0.0, X2=0.0) # [N/mm]
tr.X1 = 0.0 # [N/mm]
tr.X2 = -0.2 # [N/mm]

V = VectorFunctionSpace(mesh, 'CG', 1)

# Note use of Vector DirichletBC and "pointwise"
bc_right = DirichletBC(V, Constant((0.0, 0.0)), lrc, "pointwise")
bc_left_1 = DirichletBC(V.sub(1), Constant(0.0), llc, "pointwise")
bc = [bc_left_1, bc_right]

print "Marked edges:", sum(facet_domains.array() == 3)
print "Fixed dofs:", bc_right.get_boundary_values()
print "Fixed dofs:", bc_left_1.get_boundary_values()

#definition for the variational formulation
u = TrialFunction(V)
w = TestFunction(V)

#material coeff.s of a stainless steal
nu = 0.3 # POISSON ratio
E = 210000.0 #MPa Young modulus
G = E / (2.0 * (1.0 + nu)) # shear modulus
l = 2.0 * G * nu / (1.0 - 2.0 * nu) # lambda (a Lame constant)

#building the variational form F=0
F = w[i].dx(i) * 2.0 * G * nu / (1.0 - 2.0 * nu) * u[k].dx(k) * dx
F += w[i].dx(j) * G * u[i].dx(j) * dx
F += w[i].dx(j) * G * u[j].dx(i) * dx
F += -w[i] * tr[i] * ds(3)
a = lhs(F)
L = rhs(F)
A = assemble(a, exterior_facet_domains=facet_domains, mesh=mesh)
b = assemble(L, exterior_facet_domains=facet_domains, mesh=mesh)

for condition in bc :
    condition.apply(A, b)

u = Function(V)
solve(A, u.vector(), b)

# write out
file_u = File('elastostatic_deformations.pvd')
file_u << u

# Plot
plot(u, mode='displacement', interactive=True)
