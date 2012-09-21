from __future__ import division
import logging
import os, sys
import functools
import numpy as np
from math import sqrt

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.coefficient_field import ParametricCoefficientField
from spuq.application.egsz.adaptive_solver import prepare_rhs, prepare_rhs_copy, pcg_solve
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.plot.plotter import Plotter
from spuq.stochastics.random_variable import UniformRV
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitInterval, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active, DirichletBC, Expression)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# setup logging
# log level and format configuration
LOG_LEVEL = logging.INFO
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=__file__[:-2] + 'log', level=LOG_LEVEL,
                    format=log_format)

# FEniCS logging
from dolfin import (set_log_level, set_log_active, INFO, DEBUG, WARNING)
set_log_active(True)
set_log_level(WARNING)
fenics_logger = logging.getLogger("FFC")
fenics_logger.setLevel(logging.WARNING)
fenics_logger = logging.getLogger("UFL")
fenics_logger.setLevel(logging.WARNING)

# module logger
logger = logging.getLogger(__name__)
logging.getLogger("spuq.application.egsz.multi_operator").disabled = True
#logging.getLogger("spuq.application.egsz.marking").setLevel(logging.INFO)
# add console logging output
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
ch.setFormatter(logging.Formatter(log_format))
logger.addHandler(ch)
logging.getLogger("spuq").addHandler(ch)

# determine path of this module
path = os.path.dirname(__file__)
lshape_xml = os.path.join(path, 'lshape.xml')

# ------------------------------------------------------------

# utility functions 

# setup initial multivector
def setup_vec(mesh):
    fs = FunctionSpace(mesh, "CG", 1)
    vec = FEniCSVector(Function(fs))
    return vec


# ============================================================
# PART A: Simulation Options
# ============================================================

# domain
domain = 'square'

# initial mesh elements
initial_mesh_N = 5*8

# decay exponent
decay_exp = 2


# ============================================================
# PART B: Problem Setup
# ============================================================


# define initial multiindices
mis = list(Multiindex.createCompleteOrderSet(1, 1))
#mis = list(Multiindex.createCompleteOrderSet(0, 1))
#print mis
#os.sys.exit()

# setup meshes
mesh0 = UnitInterval(initial_mesh_N)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)

# define coefficient field
rvs = lambda i: UniformRV(a= -1, b=1)
a0 = Constant(1.0)
a = lambda i: Expression("B", B=1.0/(4.0+i*i))
coeff_field = ParametricCoefficientField(a0, a, rvs)




# define Dirichlet boundary
def u0_boundary(x, on_boundary):
    return on_boundary

Dirichlet_boundary = (u0_boundary,)
#uD = (Constant(-2.0), )
uD = (Constant(-2.0), )

Neumann_boundary = None
g = None


# define source term
f = Constant(1.0)

pde = FEMPoisson(dirichlet_boundary=Dirichlet_boundary, uD=uD, 
                 neumann_boundary=Neumann_boundary, g=g,
                 f=f)


# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs)

# pcg solver
pcg_eps = 1e-6
pcg_maxiter = 100

np.set_printoptions(linewidth=1000, precision=3, suppress=True)

# get boundary dofs
dofs = []
bcs = pde.create_dirichlet_bcs(w[Multiindex()].basis, None, None)
for bc in bcs:
    dofs += bc.get_boundary_values().keys()
print dofs


if True:
    b = 0 * w
    zero  = Multiindex()
    b[zero].coeffs = pde.assemble_rhs(coeff_field.mean_func, basis=b[zero].basis)
    for m in range(w.max_order):
        eps_m = zero.inc(m)
        am_f, am_rv = coeff_field[m]
        beta = am_rv.orth_polys.get_beta(1)

        g0 = pde.assemble_rhs(am_f, basis=b[eps_m].basis, f=Constant(0.0))
        g0[dofs]=0
        b[eps_m].coeffs += beta[1] * g0

        g0 = pde.assemble_rhs(am_f, basis=b[zero].basis, f=Constant(0.0))
        g0[dofs]=0
        b[zero].coeffs += beta[0] * g0
        #b[eps_m].coeffs[dofs] += beta[1] * pde.assemble_rhs(am_f, basis=b[eps_m].basis, f=Constant(0.0))[dofs]
        #b[zero].coeffs[dofs] += beta[0] * pde.assemble_rhs(am_f, basis=b[zero].basis, f=Constant(0.0))[dofs]
    b0 = 1 * b

if True:
    b = 0 * w
    w0 = 1 * w
    b = 1*b0
    zero  = Multiindex()
    b[zero].coeffs = pde.assemble_rhs(coeff_field.mean_func, basis=b[zero].basis)
    pde.set_dirichlet_bc_entries(w0[mu], homogeneous=False)
    for mu in w0.active_indices():
        pde.set_dirichlet_bc_entries(w0[mu], homogeneous=bool(mu.order!=0))

    d = A * w0
    pde.copy_dirichlet_bc(d, b)
    #b[zero].coeffs = pde.assemble_rhs(coeff_field.mean_func, basis=b[zero].basis)
    b1 = b

b2 = prepare_rhs(A, w, coeff_field, pde)
b3 = prepare_rhs_copy(A, w, coeff_field, pde)


bl = 0 * b
for mu in w.active_indices():
    bl[mu].coeffs[dofs]=1


for mu in w.active_indices():
    print
    print "="*80
    print mu
    print np.array([b0[mu].coeffs.array(),b1[mu].coeffs.array(),b2[mu].coeffs.array(),b3[mu].coeffs.array(),bl[mu].coeffs.array()]).T


b = b2
B = []
for mu in b.active_indices():
    B += [b[mu].array()]
print np.array(B)
print pde.assemble_operator(coeff_field.mean_func, w[Multiindex()].basis)._matrix.array()
print pde.assemble_operator_inner_dofs(coeff_field[0][0], w[Multiindex()].basis)._matrix.array()
#sys.exit(0)


P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)

#for mu in w.active_indices():
#    pde.set_dirichlet_bc_entries(w[mu], homogeneous=bool(mu.order!=0))
#w, zeta = pcg_solve(A, w, coeff_field, pde, {}, pcg_eps, pcg_maxiter)
#numit = -1

logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

if True:
    for mu in w.active_indices():
        plot(w[mu]._fefunc, title="Parametric solution for mu=%s"%mu)
    interactive()
