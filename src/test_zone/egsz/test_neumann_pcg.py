from __future__ import division
import logging
import os
import functools
import numpy as np
from math import sqrt

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.adaptive_solver import setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator, ASSEMBLY_TYPE
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.adaptive_solver import prepare_rhs, pcg_solve
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.plot.plotter import Plotter
from spuq.application.egsz.experiment_starter import ExperimentStarter
from spuq.application.egsz.egsz_utils import setup_logging
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)


# ------------------------------------------------------------

configfile = "test_neumann_pcg.conf"
config = ExperimentStarter._parse_config(configfile=configfile)

# propagate config values
for sec in config.keys():
    if sec == "LOGGING":
        continue
    secconf = config[sec]
    for key, val in secconf.iteritems():
        print "CONF_" + key + "= secconf['" + key + "'] =", secconf[key]
        exec "CONF_" + key + "= secconf['" + key + "']"

# setup logging
print "LOG_LEVEL = logging." + config["LOGGING"]["level"]
exec "LOG_LEVEL = logging." + config["LOGGING"]["level"]
logger = setup_logging(LOG_LEVEL)

# save current settings
ExperimentStarter._extract_config(globals(), savefile="test_neumann_pcg-save.conf")


# ============================================================
# PART A: Simulation Options
# ============================================================

# initial mesh elements
initial_mesh_N = CONF_initial_mesh_N

# plotting flag
PLOT_SOLUTION = True


# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = list(Multiindex.createCompleteOrderSet(CONF_initial_Lambda, 1))
#mis = list(Multiindex.createCompleteOrderSet(0, 1))
#mis = [mis[0], mis[2]]
#mis = [mis[0]]
print "MIS", mis
    
# setup domain and meshes
mesh0, boundaries, dim = SampleDomain.setupDomain(CONF_domain, initial_mesh_N=initial_mesh_N)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# define coefficient field
# NOTE: for proper treatment of corner points, see elasticity_residual_estimator
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                    freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale)
pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, CONF_domain, CONF_problem_type, boundaries, coeff_field)

# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs, assembly_type=eval("ASSEMBLY_TYPE." + CONF_assembly_type))

# setup initial solution multivector
w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=CONF_FEM_degree))
logger.info("active indices of w after initialisation: %s", w.active_indices())



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
sys.settrace(traceit)



# pcg solver
b = prepare_rhs(A, w, coeff_field, pde)
P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
w, zeta, numit = pcg(A, b, P, w0=w, eps=CONF_pcg_eps, maxiter=CONF_pcg_maxiter)
logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

if True:
    for mu in w.active_indices():
        plot(w[mu]._fefunc, title="solution %s" % str(mu))
    interactive()
