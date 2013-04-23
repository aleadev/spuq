from __future__ import division
import os
import functools
import logging
from math import sqrt
from collections import defaultdict

from spuq.application.egsz.adaptive_solver2 import AdaptiveSolver
from spuq.application.egsz.multi_operator2 import MultiOperator
from spuq.application.egsz.sample_problems2 import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.application.egsz.multi_vector import MultiVectorSharedBasis
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.application.egsz.egsz_utils import setup_logging, stats_plotter
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------
logger = logging.getLogger(__name__)

def run_SFEM(opts, conf):
    # propagate config values
    for sec in conf.keys():
        if sec == "LOGGING":
            continue
        secconf = conf[sec]
        for key, val in secconf.iteritems():
            print "CONF_" + key + "= secconf['" + key + "'] =", secconf[key]
            exec "CONF_" + key + "= secconf['" + key + "']"

    # setup logging
    print "LOG_LEVEL = logging." + conf["LOGGING"]["level"]
    exec "LOG_LEVEL = logging." + conf["LOGGING"]["level"]
    setup_logging(LOG_LEVEL, logfile=CONF_experiment_name + "_SFEM")
    
    # determine path of this module
    path = os.path.dirname(__file__)
    
    # ============================================================
    # PART A: Simulation Options
    # ============================================================
    
    # flags for residual and tail refinement 
    REFINEMENT = {"RES":CONF_refine_residual, "TAIL":CONF_refine_tail, "OSC":CONF_refine_osc}

    
    # ============================================================
    # PART B: Problem Setup
    # ============================================================
    
    # define initial multiindices
    mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(CONF_initial_Lambda, 1)]
    
    # setup domain and meshes
    mesh0, boundaries, dim = SampleDomain.setupDomain(CONF_domain, initial_mesh_N=CONF_initial_mesh_N)
    #meshes = SampleProblem.setupMesh(mesh0, num_refine=10, randref=(0.4, 0.3))
    mesh0 = SampleProblem.setupMesh(mesh0, num_refine=0)
    
    # define coefficient field
    # NOTE: for proper treatment of corner points, see elasticity_residual_estimator
    coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
    from itertools import count
    if CONF_mu is not None:
        muparam = (CONF_mu, (0 for _ in count()))
    else:
        muparam = None 
    coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                    freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale, secondparam=muparam)

    # setup boundary conditions and pde
    pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, CONF_domain, CONF_problem_type, boundaries, coeff_field)

    # define multioperator
    A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs)

    # setup initial solution multivector
    w = SampleProblem.setupMultiVector(mis, pde, mesh0, CONF_FEM_degree)
    logger.info("active indices of w after initialisation: %s", w.active_indices())

    sim_stats = None
    w_history = []
    if opts.continueSFEM:
        try:
            logger.info("CONTINUIING EXPERIMENT: loading previous data of %s...", CONF_experiment_name)
            import pickle
            LOAD_SOLUTION = os.path.join(opts.basedir, CONF_experiment_name)
            logger.info("loading solutions from %s" % os.path.join(LOAD_SOLUTION, 'SFEM2-SOLUTIONS.pkl'))
            # load solutions
            with open(os.path.join(LOAD_SOLUTION, 'SFEM2-SOLUTIONS.pkl'), 'rb') as fin:
                w_history = pickle.load(fin)
            # convert to MultiVectorWithProjection
            for i, mv in enumerate(w_history):
                w_history[i] = MultiVectorSharedBasis(multivector=w_history[i])
            # load simulation data
            logger.info("loading statistics from %s" % os.path.join(LOAD_SOLUTION, 'SIM2-STATS.pkl'))
            with open(os.path.join(LOAD_SOLUTION, 'SIM2-STATS.pkl'), 'rb') as fin:
                sim_stats = pickle.load(fin)
            logger.info("active indices of w after initialisation: %s", w_history[-1].active_indices())
            w0 = w_history[-1]
        except:
            logger.warn("FAILED LOADING EXPERIMENT %s --- STARTING NEW DATA", CONF_experiment_name)
            w0 = w    
    else:
        w0 = w

    
    # ============================================================
    # PART C: Adaptive Algorithm
    # ============================================================
    
    # refinement loop
    # ===============
    w, sim_stats = AdaptiveSolver(A, coeff_field, pde, mis, w0, mesh0, CONF_FEM_degree,
                        # marking parameters
                        rho=CONF_rho, # tail factor
                        sigma=CONF_sigma, # residual factor
                        theta_x=CONF_theta_x, # residual marking bulk parameter
                        theta_y=CONF_theta_y, # tail bound marking bulk paramter
                        maxh=CONF_maxh, # maximal mesh width for coefficient maximum norm evaluation
                        add_maxm=CONF_add_maxm, # maximal search length for new new
                        # residual error evaluation
                        quadrature_degree=CONF_quadrature_degree,
                        # pcg solver
                        pcg_eps=CONF_pcg_eps, pcg_maxiter=CONF_pcg_maxiter,
                        # adaptive algorithm threshold
                        error_eps=CONF_error_eps,
                        # refinements
                        max_refinements=CONF_iterations, do_refinement=REFINEMENT, do_uniform_refinement=CONF_uniform_refinement,
                        w_history=w_history,
                        sim_stats=sim_stats)
    
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
    
    
    # ============================================================
    # PART D: Export of Solutions and Simulation Data
    # ============================================================
        
    # flag for final solution export
    if opts.saveData:
        import pickle
        SAVE_SOLUTION = os.path.join(opts.basedir, CONF_experiment_name)
        try:
            os.makedirs(SAVE_SOLUTION)
        except:
            pass
        logger.info("saving solutions into %s" % os.path.join(SAVE_SOLUTION, 'SFEM2-SOLUTIONS.pkl'))
        # save solutions
        with open(os.path.join(SAVE_SOLUTION, 'SFEM2-SOLUTIONS.pkl'), 'wb') as fout:
            pickle.dump(w_history, fout)
        # save simulation data
        sim_stats[0]["OPTS"] = opts
        sim_stats[0]["CONF"] = conf
        logger.info("saving statistics into %s" % os.path.join(SAVE_SOLUTION, 'SIM2-STATS.pkl'))
        with open(os.path.join(SAVE_SOLUTION, 'SIM2-STATS.pkl'), 'wb') as fout:
            pickle.dump(sim_stats, fout)


    # ============================================================
    # PART E: Plotting
    # ============================================================
    
    # plot residuals
    if opts.plotEstimator and len(sim_stats) > 1:
        try:
            from matplotlib.pyplot import figure, show, legend
            X = [s["DOFS"] for s in sim_stats]
            print "DOFS", X
            L2 = [s["ERROR-L2"] for s in sim_stats]
            H1A = [s["ERROR-H1A"] for s in sim_stats]
            err_est = [s["ERROR-EST"] for s in sim_stats]
            err_res = [s["ERROR-RES"] for s in sim_stats]
            err_tail = [s["ERROR-TAIL"] for s in sim_stats]
            mi = [s["MI"] for s in sim_stats]
            num_mi = [len(m) for m in mi]
#            reserrmu = defaultdict(list)
#            for rem in _reserrmu:
#                for mu, v in rem:
#                    reserrmu[mu].append(v)
#            projerrmu = defaultdict(list)
#            for pem in _projerrmu:
#                for mu, v in pem:
#                    projerrmu[mu].append(v)
#            print "errest", errest
    
#            # --------
#            # figure 2
#            # --------
#            fig2 = figure()
#            fig2.suptitle("error estimator")
#            ax = fig2.add_subplot(111)
#            ax.loglog(X, errest, '-g<', label='error estimator')
#            legend(loc='upper right')
            
            # --------
            # figure 1
            # --------
            fig1 = figure()
            fig1.suptitle("residual estimator")
            ax = fig1.add_subplot(111)
            if REFINEMENT["TAIL"]:
                ax.loglog(X, num_mi, '--y+', label='active mi')
            ax.loglog(X, err_est, '-g<', label='error estimator')
            ax.loglog(X, err_res, '-.cx', label='residual')
            ax.loglog(X, err_tail, '-.m>', label='tail')
#            ax.loglog(X, pcg_part, '-.b>', label='pcg part')
            legend(loc='upper right')
            
            print "EST", err_est
            print "RES", err_res
            print "TAIL", err_tail
            
            show()  # this invalidates the figure instances...
        except:
            import traceback
            print traceback.format_exc()
            logger.info("skipped plotting since matplotlib is not available...")
