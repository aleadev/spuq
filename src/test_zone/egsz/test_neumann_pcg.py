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
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.application.egsz.adaptive_solver import prepare_rhs, pcg_solve
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.plot.plotter import Plotter
from spuq.application.egsz.experiment_starter import ExperimentStarter
from spuq.application.egsz.egsz_utils import setup_logging
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active, VectorFunctionSpace, project, grad)
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)


# ------------------------------------------------------------

# use alternate mayavi plotting
MAYAVI_PLOTTING = True

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
#sys.settrace(traceit)


# pcg solver
b = prepare_rhs(A, w, coeff_field, pde)
P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
w, zeta, numit = pcg(A, b, P, w0=w, eps=CONF_pcg_eps, maxiter=CONF_pcg_maxiter)
logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

#if True:
#    for mu in w.active_indices():
#        plot(w[mu]._fefunc, title="solution %s" % str(mu))
#    interactive()


if PLOT_SOLUTION:
    # get random field sample and evaluate solution (direct and parametric)
    RV_samples = coeff_field.sample_rvs()
    if False:
        foo = []
        for i, sam in enumerate(RV_samples):
            foo.append(sam)
            if i==100:
                break
            print "RV_samples:", foo
        RV_samples = foo
    if not True:
        RV_samples =  [0.4665877729078365, 0.793184920279451, 0.5180825582224986, 
                       0.04011850006307882, -0.2758205582187734, -0.15819618000193025, 
                       0.35323233387386366, -0.4420968599815964, -0.8260491796406144, 
                       -0.9089562185225297, 0.5751333963473928, -0.03307779164657276, 
                       0.15427690586776888]
    ref_maxm = w.max_order
    mu0 = Multiindex()
    sub_spaces = w[mu0].basis.num_sub_spaces
    degree = w[mu0].basis.degree
    maxh = w[mu0].basis.minh
    projection_basis = get_projection_basis(mesh0, maxh=maxh, degree=degree, sub_spaces=sub_spaces)
    sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
    sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis)
    sol_variance = compute_solution_variance(coeff_field, w, projection_basis)
#        # debug---
#        if not True:        
#            for mu in w.active_indices():
#                for i, wi in enumerate(w_history):
#                    if i == len(w_history) - 1 or True:
#                        plot(wi[mu]._fefunc, title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
##                        plot(wi[mu]._fefunc.function_space().mesh(), title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
#                interactive()
#        # ---debug

    mesh_param = sample_sol_param._fefunc.function_space().mesh()
    mesh_direct = sample_sol_direct._fefunc.function_space().mesh()

    V=sample_sol_param._fefunc.function_space()
    Vd = VectorFunctionSpace(V.mesh(), "DG", 0)
    gradsol=project(grad(sample_sol_param._fefunc),Vd)
    gradsolx, gradsoly=gradsol.split(deepcopy=True)
    #grad_y = project(dolfin.inner(grad(sample_sol_param._fefunc),Constant((1.0,0.0))), V)


    MAYAVI_PLOTTING = False
    if not MAYAVI_PLOTTING:
        wireframe = not True
        viz_p = plot(sample_sol_param._fefunc, title="parametric solution", mesh=mesh_param, wireframe=wireframe)#, rescale=False)
        viz_d = plot(sample_sol_direct._fefunc, title="direct solution", mesh=mesh_direct, wireframe=wireframe)#, rescale=False)
        viz_x = plot(gradsolx, title="x-gradient solution", wireframe=wireframe)#, rescale=False)
        viz_y = plot(gradsoly, title="y-gradient solution", wireframe=wireframe)#, rescale=False)
    else:
        Plotter.plotMesh(sample_sol_param._fefunc)
        Plotter.plotMesh(sample_sol_direct._fefunc)

        Plotter.plotMesh(gradsolx)
        Plotter.plotMesh(gradsoly)

    if not MAYAVI_PLOTTING:
        interactive()
    else:
        Plotter.show()
