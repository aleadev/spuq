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
from spuq.application.egsz.mc_error_sampling import sample_error_mc
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
                        plot, interactive, set_log_level, set_log_active)
    from spuq.application.egsz.fem_discretisation import FEMNavierLame
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# randomly refine initial meshes
RANDOM_MESHES = False

# use alternate mayavi plotting
MAYAVI_PLOTTING = True

# determine path of this module
path = os.path.dirname(__file__)

configfile = "test_elasticity_pcg.conf"
config = ExperimentStarter._parse_config(configfile=os.path.join(path, configfile))

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
ExperimentStarter._extract_config(globals(), savefile=os.path.join(path, "test_elasticity_pcg-save.conf"))

# ------------------------------------------------------------

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
if RANDOM_MESHES:
    meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.5, 0.4))
else:
    meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# define coefficient field
# NOTE: for proper treatment of corner points, see elasticity_residual_estimator
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
from itertools import count
muparam = (CONF_mu, (0 for _ in count())) 
coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                    freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale, secondparam=muparam)

# setup boundary conditions
try:
    mu = CONF_mu
except:
    mu = None
pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, CONF_domain, CONF_problem_type, boundaries, coeff_field)

# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs, assembly_type=eval("ASSEMBLY_TYPE." + CONF_assembly_type))

# setup initial solution multivector
w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=CONF_FEM_degree))
logger.info("active indices of w after initialisation: %s", w.active_indices())


# ============================================================
# PART C: Assemble and Solve
# ============================================================

try:
    # pcg solver
    b = prepare_rhs(A, w, coeff_field, pde)
    P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
    w, zeta, numit = pcg(A, b, P, w0=w, eps=CONF_pcg_eps, maxiter=CONF_pcg_maxiter)
    logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

#    raise Exception("TESTING")
except Exception as e:
    print e
    def get_exception_frame_data():
        import sys
        import traceback
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
        return (frame, frame.f_code.co_name, frame.f_locals)

    def do_debug(**kwargs):
        from spuq.utils.debug import Mdb
        Mdb().set_trace()

    def test_symmetry(A=None, w0=None, **kwargs):
        from spuq.linalg.operator import evaluate_operator_matrix
        v = w0
        P = v.to_euclidian_operator
        Q = v.from_euclidian_operator
        print "Computing the matrix. This may take a while..."
        A_mat = evaluate_operator_matrix(P * A * Q)

        import scipy.linalg as la
        print "norm(A-A^T) = %s" % la.norm(A_mat - A_mat.T)
        print "norm(A) = %s" % la.norm(A_mat)
        B = 0.5 * (A_mat + A_mat.T)
        lam = la.eig(B)[0]
        print "l_min = %s, l_max = %s" % (min(lam), max(lam))

    def test_symmetry_mod(A=None, **kwargs):
        test_symmetry(A=A, w0=w, **kwargs)

    debug_map = {"pcg": test_symmetry,
                 "<module>": test_symmetry_mod}

    (frame, name, locals) = get_exception_frame_data()
    if name in debug_map:
        debug_map[name](**locals)
        do_debug(**locals)
    else:
        raise

    #sys.exit()




#if True:
#    for i, mesh in enumerate(meshes):
#        plot(mesh, title="mesh %i" % i, mode="displacement")
#    interactive()

# plot sample solution
if PLOT_SOLUTION:
    # get random field sample and evaluate solution (direct and parametric)
    RV_samples = coeff_field.sample_rvs()
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
    wireframe = True
    if not MAYAVI_PLOTTING:
        viz_p = plot(sample_sol_param._fefunc, title="parametric solution", mode="displacement", mesh=mesh_param, wireframe=wireframe)#, rescale=False)
        viz_d = plot(sample_sol_direct._fefunc, title="direct solution", mode="displacement", mesh=mesh_direct, wireframe=wireframe)#, rescale=False)
    else:
        PLOTSCALE = 100 if CONF_boundary_type == 3 else 1
        Plotter.plotMesh(sample_sol_param._fefunc, displacement=True, scale=PLOTSCALE)
        Plotter.plotMesh(sample_sol_direct._fefunc, displacement=True, scale=PLOTSCALE)
    
#    for mu in w.active_indices():
#        if not MAYAVI_PLOTTING:
#            viz_p = plot(w[mu]._fefunc, title="parametric solution: " + str(mu), mode="displacement", mesh=mesh_param, wireframe=wireframe)
#        else:
#            Plotter.plotMesh(w[mu]._fefunc, displacement=True, scale=PLOTSCALE)
    if not MAYAVI_PLOTTING:
        interactive()
    else:
        Plotter.show()
