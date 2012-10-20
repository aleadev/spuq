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
from spuq.application.egsz.adaptive_solver import prepare_rhs, pcg_solve
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.plot.plotter import Plotter
from spuq.application.egsz.egsz_utils import setup_logging
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

LOG_LEVEL = logging.INFO
logger = setup_logging(LOG_LEVEL)

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
#initial_mesh_N = 3
initial_mesh_N = 10

# decay exponent
decay_exp = 2


# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = list(Multiindex.createCompleteOrderSet(4, 1))
#mis = list(Multiindex.createCompleteOrderSet(0, 1))
#mis = [mis[0]]

# setup meshes
mesh0, boundaries, dim = SampleDomain.setupDomain(domain, initial_mesh_N=initial_mesh_N)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=5, randref=(0.6, 0.5))

# debug---
#from dolfin import refine
#meshes[1] = refine(meshes[1])
# ---debug

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)
logger.info("active indices of w after initialisation: %s", w.active_indices())

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




import math
from spuq.linalg.vector import inner
def norm(v):
    return math.sqrt(inner(v, v))

def traceit(frame, event, arg):
    filename = frame.f_code.co_filename
    funcname = frame.f_code.co_name
    lineno = frame.f_lineno

    if event == "line" and lineno == 33 and funcname == "pcg":
        locals = frame.f_locals
        locals["norm"] = norm
        globals = frame.f_globals
        print "Iter %s -> %s, %s, %s" % eval(
            "(i-1, norm(rho[i-1]), norm(s[i-1]), norm(v[i-1]))",
            globals, locals)

    return traceit

import sys
print sys.settrace(traceit)



# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs)

# pcg solver
pcg_eps = 1e-6
pcg_maxiter = 100

b = prepare_rhs(A, w, coeff_field, pde)
P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)

logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

if True:
    for mu in w.active_indices():
        plot(w[mu]._fefunc, title="solution %s" % str(mu))
    interactive()
