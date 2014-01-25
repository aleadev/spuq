from __future__ import division
from functools import partial
from collections import defaultdict
from math import sqrt
import logging
import os

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator2 import MultiOperator, PreconditioningOperator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.fem_discretisation import FEMDiscretisation, zero_function
from spuq.application.egsz.multi_vector import MultiVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything
from spuq.utils.timing import timing

try:
    from dolfin import (Function, FunctionSpace, cells, Constant, refine)
    from spuq.application.egsz.marking2 import Marking
    from spuq.application.egsz.residual_estimator2 import ResidualEstimator
    from spuq.application.equilibration.equilibration_estimator import GlobalEquilibrationEstimator, LocalEquilibrationEstimator
    from spuq.fem.fenics.fenics_utils import error_norm
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# retrieve logger
logger = logging.getLogger(__name__)


# ============================================================
# PART A: PCG Solver
# ============================================================

def prepare_rhs(A, w, coeff_field, pde):
    b = 0 * w
    assert b.active_indices() == w.active_indices()
    zero = Multiindex()
    b[zero].coeffs = pde.assemble_rhs(basis=b[zero].basis, coeff=coeff_field.mean_func,
                                      withNeumannBC=True)
    
    f = pde.f
    if f.value_rank() == 0:
        zero_func = Constant(0.0)
    else:
        zero_func = Constant((0.0,) * f.value_size())
    zero_func = zero_function(b[zero].basis._fefs) 

    for m in range(w.max_order):
        eps_m = zero.inc(m)
        am_f, am_rv = coeff_field[m]
        beta = am_rv.orth_polys.get_beta(0)

        if eps_m in b.active_indices():
            g0 = b[eps_m].copy()
            g0.coeffs = pde.assemble_rhs(basis=b[eps_m].basis, coeff=am_f, withNeumannBC=False, f=zero_func)  # this equates to homogeneous Neumann bc
            pde.set_dirichlet_bc_entries(g0, homogeneous=True)
            b[eps_m] += beta[1] * g0

        g0 = b[zero].copy()
        g0.coeffs = pde.assemble_rhs(basis=b[zero].basis, coeff=am_f, f=zero_func)
        pde.set_dirichlet_bc_entries(g0, homogeneous=True)
        b[zero] += beta[0] * g0
    return b


def pcg_solve(A, w, coeff_field, pde, stats, pcg_eps, pcg_maxiter):
    b = prepare_rhs(A, w, coeff_field, pde)
    P = PreconditioningOperator(coeff_field.mean_func,
                                pde.assemble_solve_operator)

    w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)
    logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

    b2 = A * w
    stats["RESIDUAL-L2"] = error_norm(b, b2, "L2")
    stats["RESIDUAL-H1A"] = error_norm(b, b2, pde.energy_norm)
    stats["DOFS"] = sum([b[mu]._fefunc.function_space().dim() for mu in b.keys()])
    stats["CELLS"] = sum([b[mu]._fefunc.function_space().mesh().num_cells() for mu in b.keys()])
    logger.info("[pcg] Residual = [%s (L2)] [%s (H1A)] with [%s dofs] and [%s cells]", stats["RESIDUAL-L2"], stats["RESIDUAL-H1A"], stats["DOFS"], stats["CELLS"])
    return w, zeta


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

@takes(MultiOperator, CoefficientField, FEMDiscretisation, list, MultiVector, anything, int)
def AdaptiveSolver(A, coeff_field, pde,
                    mis, w0, mesh0, degree,
                    # marking parameters
                    rho=1.0, # tail factor
                    theta_x=0.4, # residual marking bulk parameter
                    theta_y=0.4, # tail bound marking bulk paramter
                    maxh=0.1, # maximal mesh width for coefficient maximum norm evaluation
                    add_maxm=100, # maximal search length for new new multiindices (to be added to max order of solution w)
                    # estimator
                    estimator_type = "RESIDUAL",
                    quadrature_degree= -1,
                    # pcg solver
                    pcg_eps=1e-6,
                    pcg_maxiter=100,
                    # adaptive algorithm threshold
                    error_eps=1e-2,
                    # refinements
                    max_refinements=5,
                    max_dof=1e10,
                    do_refinement={"RES":True, "TAIL":True, "OSC":True},
                    do_uniform_refinement=False,
                    refine_osc_factor=1.0,
                    w_history=None,
                    sim_stats=None):
    
    # define store function for timings
    def _store_stats(val, key, stats):
        stats[key] = val
    
    # get rhs
    f = pde.f

    # setup w and statistics
    w = w0
    if sim_stats is None:
        assert w_history is None or len(w_history) == 0
        sim_stats = []

    try:
        start_iteration = max(len(sim_stats) - 1, 0)
    except:
        start_iteration = 0
    logger.info("START/CONTINUE EXPERIMENT at iteration %i", start_iteration)

    # data collection
    import resource
    refinement = None
    for refinement in range(start_iteration, max_refinements + 1):
        logger.info("************* REFINEMENT LOOP iteration {0} (of {1} or max dofs {2}) *************".format(refinement, max_refinements, max_dof))
        # memory usage info
        logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")

        # ---------
        # pcg solve
        # ---------
        
        stats = {}
        with timing(msg="pcg_solve", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-PCG", stats=stats)):
            w, zeta = pcg_solve(A, w, coeff_field, pde, stats, pcg_eps, pcg_maxiter)

        logger.info("DIM of w = %s", w.dim)
        if w_history is not None and (refinement == 0 or start_iteration < refinement):
            w_history.append(w)

        # -------------------
        # evaluate estimators
        # -------------------
        
        # evaluate estimate_y
        logger.debug("evaluating upper tail bound")
        with timing(msg="ResidualEstimator.evaluateUpperTailBound", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-TAIL", stats=stats)):
            global_zeta, zeta, zeta_bar, eval_zeta_m = ResidualEstimator.evaluateUpperTailBound(w, coeff_field, pde, maxh, add_maxm)

        # evaluate estimate_x
        if estimator_type.upper() == "RESIDUAL":
            # evaluate estimate_x
            logger.debug("evaluating residual bound (residual)")
            with timing(msg="ResidualEstimator.evaluateResidualEstimator", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-RES", stats=stats)):
                global_eta, eta, eta_local = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, pde, f, quadrature_degree)
        elif estimator_type.upper() == "EQUILIBRATION_GLOBAL":
            logger.debug("evaluating residual bound (global equilibration)")
            with timing(msg="GlobalEquilibrationEstimator.evaluateEstimator", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-RES", stats=stats)):
                global_eta, eta, eta_local = GlobalEquilibrationEstimator.evaluateEstimator(w, coeff_field, pde, f, quadrature_degree)
        elif estimator_type.upper() == "EQUILIBRATION_GLOBAL":
            logger.debug("evaluating residual bound (global equilibration)")
            with timing(msg="GlobalEquilibrationEstimator.evaluateEstimator", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-RES", stats=stats)):
                global_eta, eta, eta_local = LocalEquilibrationEstimator.evaluateEstimator(w, coeff_field, pde, f, quadrature_degree)
        else:
            raise TypeError("invalid estimator type %s" %estimator_type.upper())

        # set overall error
        xi = sqrt(global_eta ** 2 + global_zeta ** 2)
        logger.info("Overall Estimator Error xi = %s while spatial error is %s and tail error is %s", xi, global_eta, global_zeta)

        # store simulation data
        stats["ERROR-EST"] = xi
        stats["ERROR-RES"] = global_eta
        stats["ERROR-TAIL"] = global_zeta
        stats["MARKING-RES"] = 0
        stats["MARKING-MI"] = 0
#        stats["MARKING-OSC"] = 0
        stats["CADELTA"] = 0
        stats["TIME-MARK-RES"] = 0
        stats["TIME-REFINE-RES"] = 0
        stats["TIME-MARK-TAIL"] = 0
        stats["TIME-REFINE-TAIL"] = 0
        stats["TIME-REFINE-OSC"] = 0
        stats["MI"] = [mu for mu in w.active_indices()]
        stats["DIM"] = w.dim
        if refinement == 0 or start_iteration < refinement:
            sim_stats.append(stats)
            print "SIM_STATS:", sim_stats[refinement]
            
        # exit when either error threshold or max_refinements or max_dof is reached
        if refinement > max_refinements:
            logger.info("SKIPPING REFINEMENT after FINAL SOLUTION in ITERATION %i", refinement)
            break
        if sim_stats[refinement]["DOFS"] >= max_dof:
            logger.info("REACHED %i DOFS, EXITING refinement loop", sim_stats[refinement]["DOFS"])
            break
        if xi <= error_eps:
            logger.info("SKIPPING REFINEMENT since ERROR REACHED requested ACCURACY, xi=%f", xi)
            break

        # -----------------------------------
        # mark and refine and activate new mi
        # -----------------------------------

        if refinement < max_refinements:
            logger.debug("START marking === %s", str(do_refinement))
            # === mark x ===
            res_marked = False
            if do_refinement["RES"]:
                cell_ids = []
                if not do_uniform_refinement:        
                    if global_eta > rho * global_zeta or not do_refinement["TAIL"]:
                        logger.info("REFINE RES")
                        with timing(msg="Marking.mark_x", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-MARK-RES", stats=stats)):
                            cell_ids = Marking.mark_x(global_eta, eta_local, theta_x)
                        res_marked = True
                    else:
                        logger.info("SKIP REFINE RES -> mark stochastic modes instead")
                else:
                    # uniformly refine mesh
                    logger.info("UNIFORM refinement RES")
                    cell_ids = [c.index() for c in cells(w.basis._fefs.mesh())]
                    res_marked = True
            else:
                logger.info("SKIP residual refinement")
            # refine mesh
            if res_marked:
                logger.debug("w.dim BEFORE refinement: %s", w.dim)
                with timing(msg="Marking.refine_x", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-REFINE-RES", stats=stats)):
                    w = Marking.refine_x(w, cell_ids)
                logger.debug("w.dim AFTER refinement: %s", w.dim)
                            
            # === mark y ===
            if do_refinement["TAIL"] and not res_marked:
                logger.info("REFINE TAIL")
                with timing(msg="Marking.mark_y", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-MARK-TAIL", stats=stats)):
                    new_mi = Marking.mark_y(w.active_indices(), zeta, eval_zeta_m, theta_y, add_maxm)
                # add new multiindices
                with timing(msg="Marking.refine_y", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-REFINE-TAIL", stats=stats)):
                    Marking.refine_y(w, new_mi)
            else:
                new_mi = []
                logger.info("SKIP tail refinement")

            # === uniformly refine for coefficient function oscillations ===
            if do_refinement["OSC"]:
                logger.info("REFINE OSC")
                with timing(msg="Marking.refine_osc", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-REFINE-OSC", stats=stats)):
                    w, maxh, Cadelta = Marking.refine_osc(w, coeff_field, refine_osc_factor)
                    logger.info("coefficient oscillations require maxh %f with current mesh maxh %f and Cadelta %f" % (maxh, w.basis.basis.mesh.hmax(), Cadelta))
                    stats["CADELTA"] = Cadelta
            else:
                logger.info("SKIP oscillation refinement")
            
            logger.info("MARKING was carried out with %s (res) cells and %s (mi) new multiindices", len(cell_ids), len(new_mi))
            stats["MARKING-RES"] = len(cell_ids)
            stats["MARKING-MI"] = len(new_mi)
    
    if refinement:
        logger.info("ENDED refinement loop after %i of (max) %i refinements with %i dofs and %i active multiindices",
                    refinement, max_refinements, sim_stats[refinement]["DOFS"], len(sim_stats[refinement]["MI"]))

    return w, sim_stats
