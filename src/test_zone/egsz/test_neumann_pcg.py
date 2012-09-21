from __future__ import division
import logging
import os
import functools
import numpy as np
from math import sqrt

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.adaptive_solver import prepare_rhs, prepare_rhs_copy, pcg_solve
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
initial_mesh_N = 3
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
coeff_field = SampleProblem.setupCF(coeff_types[0], decayexp=decay_exp, 
                                    gamma=gamma, freqscale=1, freqskip=20,
                                    rvtype="uniform")
# define Dirichlet boundary
Dirichlet_boundary = (boundaries['left'], boundaries['right'])
uD = (Constant(-2.0), Constant(3.0))

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

#np.set_printoptions(

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
        b[eps_m].coeffs += beta[1] * pde.assemble_rhs(am_f, basis=b[eps_m].basis, f=Constant(0.0))
        b[zero].coeffs += beta[0] * pde.assemble_rhs(am_f, basis=b[zero].basis, f=Constant(0.0))
        b[eps_m].coeffs[dofs]=0
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


np.set_printoptions(linewidth=1000)
for mu in w.active_indices():
    print
    print "="*80
    print mu
    print np.array([b0[mu].coeffs.array(),b1[mu].coeffs.array(),b2[mu].coeffs.array(),b3[mu].coeffs.array(),bl[mu].coeffs.array()]).T


b = b1

#P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
#w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)
w, zeta = pcg_solve(A, w, coeff_field, pde, {}, pcg_eps, pcg_maxiter)
numit = -1

logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

if True:
    for mu in w.active_indices():
        plot(w[mu]._fefunc, title="Parametric solution for mu=%s"%mu)
    interactive()
