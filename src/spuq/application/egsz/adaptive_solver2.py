from __future__ import division
from functools import partial
from collections import defaultdict
from math import sqrt
from collections import namedtuple
import logging
import os

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.fem_discretisation import FEMDiscretisation
from spuq.application.egsz.multi_vector import MultiVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything
from spuq.utils.timing import timing

try:
    from dolfin import (Function, FunctionSpace, cells, Constant, refine)
    from spuq.application.egsz.marking import Marking
    from spuq.application.egsz.residual_estimator import ResidualEstimator
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_utils import error_norm
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# retrieve logger
logger = logging.getLogger(__name__)


# setup initial multivector
def setup_vector(pde, basis):
    mesh = basis._fefs.mesh()
    degree = basis._fefs.ufl_element().degree()
    fs = pde.function_space(mesh, degree=degree)
    vec = FEniCSVector(Function(fs))
    return vec


# ============================================================
# PART A: PCG Solver
# ============================================================

def prepare_rhs(A, w, coeff_field, pde):
    b = 0 * w
    zero = Multiindex()
    b[zero].coeffs = pde.assemble_rhs(coeff_field.mean_func, basis=b[zero].basis, withNeumannBC=True)
    
    f = pde._f
    if f.value_rank() == 0:
        zero_func = Constant(0.0)
    else:
        zero_func = Constant((0.0,) * f.value_size())

    for m in range(w.max_order):
        eps_m = zero.inc(m)
        am_f, am_rv = coeff_field[m]
        beta = am_rv.orth_polys.get_beta(0)

        if eps_m in b.active_indices():
            g0 = b[eps_m].copy()
            g0.coeffs = pde.assemble_rhs(am_f, basis=b[eps_m].basis, withNeumannBC=False, f=zero_func)  # this equates to homogeneous Neumann bc
            pde.set_dirichlet_bc_entries(g0, homogeneous=True)
            b[eps_m] += beta[1] * g0

        g0 = b[zero].copy()
        g0.coeffs = pde.assemble_rhs(am_f, basis=b[zero].basis, f=zero_func)
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
    stats["L2"] = error_norm(b, b2, "L2")
    stats["H1"] = error_norm(b, b2, pde.norm)
#    stats["H1"] = error_norm(b, b2, "H1")
    stats["DOFS"] = sum([b[mu]._fefunc.function_space().dim() for mu in b.keys()])
    stats["CELLS"] = sum([b[mu]._fefunc.function_space().mesh().num_cells() for mu in b.keys()])
    logger.info("Residual = [%s (L2)] [%s (H1)] with [%s dofs] and [%s cells]", stats["L2"], stats["H1"], stats["DOFS"], stats["CELLS"])
    return w, zeta


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

@takes(MultiOperator, CoefficientField, FEMDiscretisation, list, MultiVector, anything, int)
def AdaptiveSolver(A, coeff_field, pde,
                    mis, w0, mesh0, degree,
                    # marking parameters
                    rho=1.0, # tail factor
                    sigma=1.0, # residual factor
                    theta_x=0.4, # residual marking bulk parameter
                    theta_y=0.4, # tail bound marking bulk paramter
                    add_maxm=20, # maximal search length for new new multiindices (to be added to max order of solution w)
                    # residual error
                    quadrature_degree= -1,
                    # pcg solver
                    pcg_eps=1e-6,
                    pcg_maxiter=100,
                    # adaptive algorithm threshold
                    error_eps=1e-2,
                    # refinements
                    max_refinements=5,
                    max_inner_refinements=20, # max iterations for inner residual refinement loop
                    do_refinement={"RES":True, "TAIL":True},
                    do_uniform_refinement=False,
                    w_history=None,
                    sim_stats=None):
    
    # define store function for timings
    from functools import partial
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
        logger.info("************* REFINEMENT LOOP [OUTER] iteration %i (of %i) *************", refinement, max_refinements)
        # memory usage info
        logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")

        # pcg solve
        # ---------
        stats = {}
        with timing(msg="pcg_solve", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-PCG", stats=stats)):
            w, zeta = pcg_solve(A, w, coeff_field, pde, stats, pcg_eps, pcg_maxiter)

        logger.info("DIM of w = %s", w.dim)
        if w_history is not None and (refinement == 0 or start_iteration < refinement):
            w_history.append(w)

        # inner refinement loop (residual estimator)
        # ------------------------------------------
        for inner_refinement in range(max_inner_refinements):
            # evaluate estimate_y
            logger.debug("evaluating upper tail bound")
            with timing(msg="ResidualEstimator.evaluateUpperTailBound", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-TAIL", stats=stats)):
                global_eta, eta = ResidualEstimator.evaluateUpperTailBound(w, coeff_field, pde, maxh, add_maxm)
            # evaluate estimate_x
            with timing(msg="ResidualEstimator.evaluateResidualEstimator", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-RES", stats=stats)):
                global_eta, eta = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, pde, f, zeta, quadrature_degree)




            
            reserrmu = [(mu, sqrt(sum(resind[mu].coeffs ** 2))) for mu in resind.keys()]
            projerrmu = [(mu, sqrt(sum(projind[mu].coeffs ** 2))) for mu in projind.keys()]
            res_part, proj_part, pcg_part = estparts[0], estparts[1], estparts[2]
            err_res, err_proj, err_pcg = errors[0], errors[1], errors[2]
            logger.info("Overall Estimator Error xi = %s while residual error is %s, projection error is %s, pcg error is %s", xi, res_part, proj_part, pcg_part)
            
            stats.update(timing_stats)
            stats["EST"] = xi
            stats["RES-PART"] = res_part
            stats["PCG-PART"] = pcg_part
            stats["ERR-RES"] = err_res
            stats["ERR-PCG"] = err_pcg
            stats["ETA-ERR"] = errors[0]
            stats["DELTA-ERR"] = errors[1]
            stats["ZETA-ERR"] = errors[2]
            stats["RES-mu"] = reserrmu
            stats["MARKING-RES"] = 0
            stats["MARKING-PROJ"] = 0
            stats["MARKING-MI"] = 0
            stats["TIME-MARKING"] = 0
            stats["MI"] = [(mu, vec.basis.dim) for mu, vec in w.iteritems()]
            if refinement == 0 or start_iteration < refinement:
                sim_stats.append(stats)            
    #            print "SIM_STATS:", sim_stats[refinement]




        
        logger.debug("squared error components: eta=%s  delta=%s  zeta=%", errors[0], errors[1], errors[2])

        # exit when either error threshold or max_refinements is reached
        if refinement > max_refinements:
            logger.info("skipping refinement after final solution in iteration %i", refinement)
            break
        if xi <= error_eps:
            logger.info("error reached requested accuracy, xi=%f", xi)
            break





    return w, sim_stats
