from __future__ import division
import logging
import os
import functools
from math import sqrt
from collections import defaultdict

from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, ASSEMBLY_TYPE
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.application.egsz.fem_discretisation import FEMNavierLame
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------


def setup_logging(level):
    # log level and format configuration
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
    return logger


def run_MC(opts, conf):
    # propagate config values
    for sec in conf.keys():
        if sec == "LOGGING":
            continue
        secconf = conf[sec]
        for key, val in secconf.iteritems():
            print "CONF_" + key + "= secconf['" + key + "']"
            exec "CONF_" + key + "= secconf['" + key + "']"

    # setup logging
    exec "LOG_LEVEL = logging." + conf["LOGGING"]["level"]
    logger = setup_logging(LOG_LEVEL)
    
    # determine path of this module
    path = os.path.dirname(__file__)

    # flag for final solution export
    if not opts.saveData:
        SAVE_SOLUTION = ''
    else:
        SAVE_SOLUTION = os.path.join(opts.basedir, "MC-results")

    
    # ============================================================
    # PART A: Import Solution
    # ============================================================
    # NOTE: save at this point since MC tends to run out of memory
    if SAVE_SOLUTION != "":
        # save solution (also creates directory if not existing)
        w.pickle(SAVE_SOLUTION)
        # save simulation data
        import pickle
        with open(os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'), 'wb') as fout:
            pickle.dump(sim_stats, fout)
    
    
    # ============================================================
    # PART B: MC Error Sampling
    # ============================================================
    if CONF_runs > 0:
        ref_maxm = w_history[-1].max_order
        for i, w in enumerate(w_history):
            if i == 0:
                continue
            logger.info("MC error sampling for w[%i] (of %i)", i, len(w_history))
            # memory usage info
            logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")
            L2err, H1err, L2err_a0, H1err_a0 = sample_error_mc(w, pde, A, coeff_field, mesh0, ref_maxm, MC_RUNS, MC_N, MC_HMAX)
            sim_stats[i - 1]["MC-L2ERR"] = L2err
            sim_stats[i - 1]["MC-H1ERR"] = H1err
            sim_stats[i - 1]["MC-L2ERR_a0"] = L2err_a0
            sim_stats[i - 1]["MC-H1ERR_a0"] = H1err_a0
    
    
    # ============================================================
    # PART C: Export Updated Data and Plotting
    # ============================================================
    # save updated data
    if SAVE_SOLUTION != "":
        # save updated statistics
        import pickle
        with open(os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'), 'wb') as fout:
            pickle.dump(sim_stats, fout)
    
    # plot residuals
    if opts.plotError and len(sim_stats) > 1:
        try:
            from matplotlib.pyplot import figure, show, legend
            x = [s["DOFS"] for s in sim_stats]
            L2 = [s["L2"] for s in sim_stats]
            H1 = [s["H1"] for s in sim_stats]
            errest = [sqrt(s["EST"]) for s in sim_stats]
            reserr = [s["RES"] for s in sim_stats]
            projerr = [s["PROJ"] for s in sim_stats]
            _reserrmu = [s["RES-mu"] for s in sim_stats]
            _projerrmu = [s["PROJ-mu"] for s in sim_stats]
            if CONF_runs > 0:
                mcL2 = [s["MC-L2ERR"] for s in sim_stats]
                mcH1 = [s["MC-H1ERR"] for s in sim_stats]
                mcL2_a0 = [s["MC-L2ERR_a0"] for s in sim_stats]
                mcH1_a0 = [s["MC-H1ERR_a0"] for s in sim_stats]
                effest = [est / err for est, err in zip(errest, mcH1)]
            mi = [s["MI"] for s in sim_stats]
            num_mi = [len(m) for m in mi]
            reserrmu = defaultdict(list)
            for rem in _reserrmu:
                for mu, v in rem:
                    reserrmu[mu].append(v)
            print "errest", errest
            if CONF_runs > 0:
                print "mcH1", mcH1
                print "efficiency", [est / err for est, err in zip(errest, mcH1)]
                
            # figure 1
            # --------
    #        fig = figure()
    #        fig.suptitle("error")
    #        ax = fig.add_subplot(111)
    #        ax.loglog(x, H1, '-g<', label='H1 residual')
    #        ax.loglog(x, L2, '-c+', label='L2 residual')
    #        ax.loglog(x, mcH1, '-b^', label='MC H1 error')
    #        ax.loglog(x, mcL2, '-ro', label='MC L2 error')
    #        ax.loglog(x, mcH1_a0, '-.b^', label='MC H1 error a0')
    #        ax.loglog(x, mcL2_a0, '-.ro', label='MC L2 error a0')
    #        legend(loc='upper right')
    #        if SAVE_SOLUTION != "":
    #            fig.savefig(os.path.join(SAVE_SOLUTION, 'RES.png'))
    
            # --------
            # figure 2
            # --------
            fig2 = figure()
            fig2.suptitle("residual estimator")
            ax = fig2.add_subplot(111)
            if REFINEMENT["MI"]:
                ax.loglog(x, num_mi, '--y+', label='active mi')
            ax.loglog(x, errest, '-g<', label='error estimator')
            ax.loglog(x, reserr, '-.cx', label='residual part')
            ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
            if MC_RUNS > 0:
                ax.loglog(x, mcH1, '-b^', label='MC H1 error')
                ax.loglog(x, mcL2, '-ro', label='MC L2 error')
    #        ax.loglog(x, H1, '-b^', label='H1 residual')
    #        ax.loglog(x, L2, '-ro', label='L2 residual')
            legend(loc='upper right')
            if SAVE_SOLUTION != "":
                fig2.savefig(os.path.join(SAVE_SOLUTION, 'EST.png'))
                fig2.savefig(os.path.join(SAVE_SOLUTION, 'EST.eps'))
    
            # --------
            # figure 3
            # --------
            fig3 = figure()
            fig3.suptitle("efficiency residual estimator")
            ax = fig3.add_subplot(111)
            ax.loglog(x, errest, '-g<', label='error estimator')
            if MC_RUNS > 0:
                ax.loglog(x, mcH1, '-b^', label='MC H1 error')
                ax.loglog(x, effest, '-ro', label='efficiency')        
            legend(loc='upper right')
            if SAVE_SOLUTION != "":
                fig3.savefig(os.path.join(SAVE_SOLUTION, 'ESTEFF.png'))
                fig3.savefig(os.path.join(SAVE_SOLUTION, 'ESTEFF.eps'))
    
            # --------
            # figure 4
            # --------
            fig4 = figure()
            fig4.suptitle("residual contributions")
            ax = fig4.add_subplot(111)
            for mu, v in reserrmu.iteritems():
                ms = str(mu)
                ms = ms[ms.find('=') + 1:-1]
                ax.loglog(x[-len(v):], v, '-g<', label=ms)
            legend(loc='upper right')
    #        if SAVE_SOLUTION != "":
    #            fig4.savefig(os.path.join(SAVE_SOLUTION, 'RESCONTRIB.png'))
    #            fig4.savefig(os.path.join(SAVE_SOLUTION, 'RESCONTRIB.eps'))
            
            show()  # this invalidates the figure instances...
        except:
            import traceback
            print traceback.format_exc()
            logger.info("skipped plotting since matplotlib is not available...")
    
