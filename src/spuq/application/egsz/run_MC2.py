from __future__ import division
import os
import logging
from math import sqrt
from collections import defaultdict

from spuq.application.egsz.multi_operator2 import MultiOperator
from spuq.application.egsz.sample_problems2 import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active, refine)
    from spuq.application.egsz.egsz_utils import setup_logging
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------
logger = logging.getLogger(__name__)

def run_MC(opts, conf):
    # propagate config values
    _G = globals()
    for sec in conf.keys():
        if sec == "LOGGING":
            continue
        secconf = conf[sec]
        for key, val in secconf.iteritems():
            print "CONF_" + key + "= secconf['" + key + "'] =", secconf[key]
            _G["CONF_" + key] = secconf[key]

    # setup logging
    _G["LOG_LEVEL"] = eval("logging." + conf["LOGGING"]["level"])
    print "LOG_LEVEL = logging." + conf["LOGGING"]["level"]    
    setup_logging(LOG_LEVEL, logfile=CONF_experiment_name + "_MC-P{0}".format(CONF_FEM_degree))
    
    # determine path of this module
    path = os.path.dirname(__file__)


    # ============================================================
    # PART A: Setup Problem
    # ============================================================
    
    # get boundaries
    mesh0, boundaries, dim = SampleDomain.setupDomain(CONF_domain, initial_mesh_N=CONF_initial_mesh_N)

    # define coefficient field
    coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
    from itertools import count
    if CONF_mu is not None:
        muparam = (CONF_mu, (0 for _ in count()))
    else:
        muparam = None 
    coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                    freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale, secondparam=muparam)
    
    # setup boundary conditions and pde
#    initial_mesh_N = CONF_initial_mesh_N
    pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, CONF_domain, CONF_problem_type, boundaries, coeff_field)
    
    # define multioperator
    A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs)

    
    # ============================================================
    # PART B: Import Solution
    # ============================================================
    import pickle
    PATH_SOLUTION = os.path.join(opts.basedir, CONF_experiment_name)
    FILE_SOLUTION = 'SFEM2-SOLUTIONS-P{0}.pkl'.format(CONF_FEM_degree)
    FILE_STATS = 'SIM2-STATS-P{0}.pkl'.format(CONF_FEM_degree)

    print "LOADING solutions from %s" % os.path.join(PATH_SOLUTION, FILE_SOLUTION)
    logger.info("LOADING solutions from %s" % os.path.join(PATH_SOLUTION, FILE_SOLUTION))
    # load solutions
    with open(os.path.join(PATH_SOLUTION, FILE_SOLUTION), 'rb') as fin:
        w_history = pickle.load(fin)
    # load simulation data
    logger.info("LOADING statistics from %s" % os.path.join(PATH_SOLUTION, FILE_STATS))
    with open(os.path.join(PATH_SOLUTION, FILE_STATS), 'rb') as fin:
        sim_stats = pickle.load(fin)

    logger.info("active indices of w after initialisation: %s", w_history[-1].active_indices())

    
    # ============================================================
    # PART C: MC Error Sampling
    # ============================================================
    
    MC_N = CONF_N
    MC_HMAX = CONF_maxh
    if CONF_runs > 0:
        # determine reference mesh
        w = w_history[-1]
        ref_mesh = w.basis.basis.mesh        
        for _ in range(CONF_ref_mesh_refine):
            ref_mesh = refine(ref_mesh)
        ref_maxm = CONF_sampling_order if CONF_sampling_order > 0 else w.max_order + CONF_sampling_order_increase
        for i, w in enumerate(w_history):
#            if i == 0:
#                continue
            logger.info("MC error sampling for w[%i] (of %i)", i, len(w_history))
            
            # memory usage info
            import resource
            logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")
            
            MC_start = 0
            old_stats = sim_stats[i]
            if opts.continueMC:
                try:
                    MC_start = sim_stats[i]["MC-N"]
                    logger.info("CONTINUING MC of %s for solution (iteration) %s of %s", PATH_SOLUTION, i, len(w_history))
                except:
                    logger.info("STARTING MC of %s for solution (iteration) %s of %s", PATH_SOLUTION, i, len(w_history))
            if MC_start <= 0:
                    sim_stats[i]["MC-N"] = 0
                    sim_stats[i]["MC-ERROR-L2"] = 0
                    sim_stats[i]["MC-ERROR-H1A"] = 0
#                     sim_stats[i]["MC-ERROR-L2_a0"] = 0
#                     sim_stats[i]["MC-ERROR-H1_a0"] = 0
            
            MC_RUNS = max(CONF_runs - MC_start, 0)
            if MC_RUNS > 0:
                logger.info("STARTING %s MC RUNS", MC_RUNS)
#                L2err, H1err, L2err_a0, H1err_a0, N = sample_error_mc(w, pde, A, coeff_field, mesh0, ref_maxm, MC_RUNS, MC_N, MC_HMAX)
                L2err, H1err, L2err_a0, H1err_a0, N = sample_error_mc(w, pde, A, coeff_field, ref_mesh, ref_maxm, MC_RUNS, MC_N, MC_HMAX)
                # combine current and previous results
                sim_stats[i]["MC-N"] = N + old_stats["MC-N"]
                sim_stats[i]["MC-ERROR-L2"] = (L2err * N + old_stats["MC-ERROR-L2"]) / sim_stats[i]["MC-N"]
                sim_stats[i]["MC-ERROR-H1A"] = (H1err * N + old_stats["MC-ERROR-H1A"]) / sim_stats[i]["MC-N"]
#                 sim_stats[i]["MC-ERROR-L2_a0"] = (L2err_a0 * N + old_stats["MC-ERRORL2_a0"]) / sim_stats[i]["MC-N"]
#                 sim_stats[i]["MC-ERROR-H1A_a0"] = (H1err_a0 * N + old_stats["MC-ERROR-H1A_a0"]) / sim_stats[i]["MC-N"]
                print "MC-ERROR-H1A (N:%i) = %f" % (sim_stats[i]["MC-N"], sim_stats[i]["MC-ERROR-H1A"])
            else:
                logger.info("SKIPPING MC RUN since sufficiently many samples are available")
    
    # ============================================================
    # PART D: Export Updated Data and Plotting
    # ============================================================
    # save updated data
    if opts.saveData:
        # save updated statistics
        print "SAVING statistics into %s" % os.path.join(PATH_SOLUTION, FILE_STATS)
        print sim_stats[-1].keys()
        logger.info("SAVING statistics into %s" % os.path.join(PATH_SOLUTION, FILE_STATS))
        with open(os.path.join(PATH_SOLUTION, FILE_STATS), 'wb') as fout:
            pickle.dump(sim_stats, fout)
    
    # plot residuals
    if opts.plotEstimator and len(sim_stats) > 1:
        try:
            from matplotlib.pyplot import figure, show, legend
            
            X = [s["DOFS"] for s in sim_stats]
            err_L2 = [s["MC-ERROR-L2"] for s in sim_stats]
            err_H1A = [s["MC-ERROR-H1A"] for s in sim_stats]
            err_est = [s["ERROR-EST"] for s in sim_stats]
            err_res = [s["ERROR-RES"] for s in sim_stats]
            err_tail = [s["ERROR-TAIL"] for s in sim_stats]
            mi = [s["MI"] for s in sim_stats]
            num_mi = [len(m) for m in mi]
            eff_H1A = [est / err for est, err in zip(err_est, err_H1A)]
            
            # --------
            # figure 1
            # --------
            fig1 = figure()
            fig1.suptitle("residual estimator")
            ax = fig1.add_subplot(111)
            if REFINEMENT["TAIL"]:
                ax.loglog(X, num_mi, '--y+', label='active mi')
            ax.loglog(X, eff_H1A, '--yo', label='efficiency')
            ax.loglog(X, err_L2, '-.b>', label='L2 error')
            ax.loglog(X, err_H1A, '-.r>', label='H1A error')
            ax.loglog(X, err_est, '-g<', label='error estimator')
            ax.loglog(X, err_res, '-.cx', label='residual')
            ax.loglog(X, err_tail, '-.m>', label='tail')
            legend(loc='upper right')

            print "error L2", err_L2
            print "error H1A", err_H1A
            print "EST", err_est
            print "RES", err_res
            print "TAIL", err_tail
            
            show()  # this invalidates the figure instances...
        except:
            import traceback
            print traceback.format_exc()
            logger.info("skipped plotting since matplotlib is not available...")
