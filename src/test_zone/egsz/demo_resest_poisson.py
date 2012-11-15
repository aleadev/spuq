from __future__ import division
import logging
import os
import functools
from math import sqrt

from spuq.application.egsz.experiment_starter import ExperimentStarter
from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator, ASSEMBLY_TYPE
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.plot.plotter import Plotter
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
ExperimentStarter._extract_config(globals(), savefile="demo_resest_poisson-save.conf")


# ============================================================
# PART A: Simulation Options
# ============================================================

# flags for residual, projection, new mi refinement 
REFINEMENT = {"RES":CONF_refine_residual, "PROJ":CONF_refine_projection, "MI":CONF_refine_Lambda}

# initial mesh elements
initial_mesh_N = CONF_initial_mesh_N

# plotting flag
PLOT_SOLUTION = True
MAYAVI_PLOTTING = True

# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(CONF_initial_Lambda, 1)]
#mis = [mis[0], mis[2]]
#mis = [mis[0]]

# setup domain and meshes
mesh0, boundaries, dim = SampleDomain.setupDomain(CONF_domain, initial_mesh_N=initial_mesh_N)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.4, 0.3))
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# define coefficient field
# NOTE: for proper treatment of corner points, see elasticity_residual_estimator
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                    freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale)

# setup boundary conditions and pde
pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, CONF_domain, CONF_problem_type, boundaries, coeff_field)

# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs, assembly_type=eval("ASSEMBLY_TYPE." + CONF_assembly_type))

# setup initial solution multivector
w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=CONF_FEM_degree))
logger.info("active indices of w after initialisation: %s", w.active_indices())


# ============================================================
# PART C: Adaptive Algorithm
# ============================================================

# refinement loop
# ===============
w_history = []
w0 = w
w, sim_stats = AdaptiveSolver(A, coeff_field, pde, mis, w0, mesh0, CONF_FEM_degree, gamma=CONF_gamma, cQ=CONF_cQ, ceta=CONF_ceta,
                    # marking parameters
                    theta_eta=CONF_theta_eta, theta_zeta=CONF_theta_zeta, min_zeta=CONF_min_zeta,
                    maxh=CONF_maxh, newmi_add_maxm=CONF_newmi_add_maxm, theta_delta=CONF_theta_delta,
                    max_Lambda_frac=CONF_max_Lambda_frac,
                    # residual error evaluation
                    quadrature_degree=CONF_quadrature_degree,
                    # projection error evaluation
                    projection_degree_increase=CONF_projection_degree_increase, refine_projection_mesh=CONF_refine_projection_mesh,
                    # pcg solver
                    pcg_eps=CONF_pcg_eps, pcg_maxiter=CONF_pcg_maxiter,
                    # adaptive algorithm threshold
                    error_eps=CONF_error_eps,
                    # refinements
                    max_refinements=CONF_iterations, do_refinement=REFINEMENT, do_uniform_refinement=CONF_uniform_refinement,
                    w_history=w_history)

from operator import itemgetter
active_mi = [(mu, w[mu]._fefunc.function_space().mesh().num_cells()) for mu in w.active_indices()]
active_mi = sorted(active_mi, key=itemgetter(1), reverse=True)
logger.info("==== FINAL MESHES ====")
for mu in active_mi:
    logger.info("--- %s has %s cells", mu[0], mu[1])
print "ACTIVE MI:", active_mi
print

# memory usage info
import resource
logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")


if len(sim_stats) > 1:
    errest = sqrt(sim_stats[-1]["EST"])
    reserr = sim_stats[-1]["RES"]
    projerr = sim_stats[-1]["PROJ"]

    alpha = CONF_coeff_scale
    beta = 1
    factor = beta / sqrt(alpha)
    print "Scaling behaviour of the error:"
    print "  alpha, beta, factor:", alpha, beta, factor
    print "  errest: ", errest, errest/factor
    print "  reserr: ", reserr, reserr/factor
    print "  projerr:", projerr, projerr/factor


if len(sim_stats) > 1:
    try:
        from matplotlib.pyplot import figure, show, legend
        x = [s["DOFS"] for s in sim_stats]
        L2 = [s["L2"] for s in sim_stats]
        H1 = [s["H1"] for s in sim_stats]
        errest = [sqrt(s["EST"]) for s in sim_stats]
        res_part = [s["RES-PART"] for s in sim_stats]
        proj_part = [s["PROJ-PART"] for s in sim_stats]
        pcg_part = [s["PCG-PART"] for s in sim_stats]
        mi = [s["MI"] for s in sim_stats]
        num_mi = [len(m) for m in mi]
        print "errest", errest
        print "reserr", reserr
        print "projerr", projerr

        # figure 1
        # --------
        fig1 = figure()
        fig1.suptitle("residual estimator components")
        ax = fig1.add_subplot(111)
        if REFINEMENT["MI"]:
            ax.loglog(x, num_mi, '--y+', label='active mi')
        ax.loglog(x, errest, '-g<', label='error estimator')
        ax.loglog(x, res_part, '-.cx', label='residual part')
        ax.loglog(x[1:], proj_part[1:], '-.m>', label='projection part')
        ax.loglog(x, pcg_part, '-.b^', label='pcg part')
        legend(loc='upper right')

        # figure 2
        # --------
        fig3 = figure()
        ax = fig3.add_subplot(111)
        ax.loglog(x, errest, '-g<', label='error estimator')
        legend(loc='upper right')
        show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        logger.info("skipped plotting since matplotlib is not available...")


# plot sample solution
if PLOT_SOLUTION:
    # get random field sample and evaluate solution (direct and parametric)
    RV_samples = coeff_field.sample_rvs()
    ref_maxm = w_history[-1].max_order
    sub_spaces = w[Multiindex()].basis.num_sub_spaces
    degree = w[Multiindex()].basis.degree
    maxh = w[Multiindex()].basis.minh
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
        viz_p = plot(sample_sol_param._fefunc, title="parametric solution", wireframe=wireframe)#, rescale=False)
        viz_d = plot(sample_sol_direct._fefunc, title="direct solution", wireframe=wireframe)#, rescale=False)
    else:
        Plotter.plotMesh(sample_sol_param._fefunc)
        Plotter.plotMesh(sample_sol_direct._fefunc)

    for mu in w.active_indices():
        if not MAYAVI_PLOTTING:
            viz_p = plot(w[mu]._fefunc, title="parametric solution: " + str(mu), mode="displacement", mesh=mesh_param, wireframe=wireframe)
        else:
            Plotter.plotMesh(w[mu]._fefunc)
            
    if not MAYAVI_PLOTTING:
        interactive()
    else:
        Plotter.show()
