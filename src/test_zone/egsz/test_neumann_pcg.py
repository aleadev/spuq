from __future__ import division
import logging
import os
import functools
from math import sqrt

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
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
initial_mesh_N = 40

# decay exponent
decay_exp = 2


# ============================================================
# PART B: Problem Setup
# ============================================================


# define initial multiindices
mis = list(Multiindex.createCompleteOrderSet(2, 1))
#mis = list(Multiindex.createCompleteOrderSet(0, 1))
#print mis
#os.sys.exit()

# setup meshes
mesh0, boundaries = SampleDomain.setupDomain(domain, initial_mesh_N=initial_mesh_N)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)


# debug---
#from dolfin import refine
#meshes[1] = refine(meshes[1])
# ---debug

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)
logger.info("active indices of w after initialisation: %s", w.active_indices())

# ---debug
#from spuq.application.egsz.multi_vector import MultiVectorWithProjection
#if SAVE_SOLUTION != "":
#    w.pickle(SAVE_SOLUTION)
#u = MultiVectorWithProjection.from_pickle(SAVE_SOLUTION, FEniCSVector)
#import sys
#sys.exit()
# ---debug

# define coefficient field
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials")
gamma = 0.9
coeff_field = SampleProblem.setupCF(coeff_types[1], decayexp=decay_exp, 
                                    gamma=gamma, freqscale=1, freqskip=0, 
                                    rvtype="uniform")
# define Dirichlet boundary
Dirichlet_boundary = (boundaries['left'], boundaries['right'])
uD = (Constant(-1.0), Constant(3.0))

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


print w
print w[Multiindex()]


if True:
    b = 0 * w
    mu  = Multiindex()
    b[mu].coeffs = pde.assemble_rhs(coeff_field.mean_func, basis=b[mu].basis)
    for m in range(w.max_order):
        eps_m = mu.inc(m)
        am_f, am_rv = coeff_field[m]
        beta = am_rv.orth_polys.get_beta(1)
        b[eps_m].coeffs += beta[1] * pde.assemble_rhs(am_f, basis=b[eps_m].basis, f=Constant(1.0))
        b[mu].coeffs += beta[0] * pde.assemble_rhs(am_f, basis=b[mu].basis, f=Constant(0.0))

if True:
    pde.set_dirichlet_bc_entries(w[mu], homogeneous=False)
    for m in range(w.max_order):
        eps_m = mu.inc(m)
        pde.set_dirichlet_bc_entries(w[eps_m], homogeneous=True)

    d = A * w
    pde.copy_dirichlet_bc(d, b)


P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)
logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

plot(w[Multiindex()]._fefunc)
plot(w[Multiindex([1])]._fefunc)
plot(w[Multiindex([0,1])]._fefunc)
interactive()
