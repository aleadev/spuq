from __future__ import division
from functools import partial
from collections import defaultdict
from math import sqrt
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
def setup_vector(mesh, pde, degree=1, maxh=None):
#    fs = FunctionSpace(mesh, "CG", degree)
    if maxh is not None:
        old_mesh = mesh
        while mesh.hmax() > maxh:
            mesh = refine(old_mesh)
            old_mesh = mesh
    fs = pde.function_space(mesh, degree=degree)
    vec = FEniCSVector(Function(fs))
    return vec


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

# refinement loop
# ===============
# error constants
@takes(MultiOperator, CoefficientField, FEMDiscretisation, list, MultiVector, anything, int)
def AdaptiveSolver(A, coeff_field, pde,
                    mis, w0, mesh0, degree,
                    gamma=0.9,
                    cQ=1.0,
                    ceta=1.0,
                    # marking parameters
                    theta_eta=0.4, # residual marking bulk parameter
                    theta_zeta=0.3, # projection marking threshold factor
                    min_zeta=1e-8, # minimal projection error to be considered 
                    maxh=0.1, # maximal mesh width for projection maximum norm evaluation
                    newmi_add_maxm=10, # maximal search length for new new multiindices (to be added to max order of solution w)
                    theta_delta=0.8, # number new multiindex activation bound
                    max_Lambda_frac=1 / 10, # max fraction of |Lambda| for new multiindices
                    # residual error
                    quadrature_degree= -1,
                    # projection error
                    projection_degree_increase=1,
                    refine_projection_mesh=1,
                    # pcg solver
                    pcg_eps=1e-6,
                    pcg_maxiter=100,
                    # adaptive algorithm threshold
                    error_eps=1e-2,
                    # refinements
                    max_refinements=7,
                    do_refinement={"RES":True, "PROJ":True, "MI":False},
                    do_uniform_refinement=False,
                    w_history=None,
                    sim_stats=None):
    f = pde.f

    w = w0
    if sim_stats is None:
        assert w_history is None or len(w_history) == 1
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
        logger.info("************* REFINEMENT LOOP iteration %i (of %i) *************", refinement, max_refinements)
        # memory usage info
        logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")

        # pcg solve
        # ---------
        stats = {}
        with timing(msg="pcg_solve", logfunc=logger.info):
            w, zeta = pcg_solve(A, w, coeff_field, pde, stats, pcg_eps, pcg_maxiter)

        logger.info("DIM of w = %s", w.dim)
        if w_history is not None and start_iteration < refinement:
            w_history.append(w)

        # error evaluation
        # ----------------
        # residual and projection errors
        logger.debug("evaluating ResidualEstimator.evaluateError")
        with timing(msg="ResidualEstimator.evaluateError", logfunc=logger.info):
            xi, resind, projind, estparts, errors = ResidualEstimator.evaluateError(w, coeff_field, pde, f, zeta, gamma, ceta, cQ,
                                                                                    maxh, quadrature_degree, projection_degree_increase,
                                                                                    refine_projection_mesh)
        reserrmu = [(mu, sqrt(sum(resind[mu].coeffs ** 2))) for mu in resind.keys()]
        projerrmu = [(mu, sqrt(sum(projind[mu].coeffs ** 2))) for mu in projind.keys()]
        res_part = estparts[0]
        proj_part = estparts[1]
        pcg_part = estparts[2]
        logger.info("Overall Estimator Error xi = %s while residual error is %s, projection error is %s, pcg error is %s", xi, res_part, proj_part, pcg_part)
        stats["EST"] = xi
        stats["RES-PART"] = res_part
        stats["PROJ-PART"] = proj_part
        stats["PCG-PART"] = pcg_part
        stats["ETA-ERR"] = errors[0]
        stats["DELTA-ERR"] = errors[1]
        stats["ZETA-ERR"] = errors[2]
        stats["RES-mu"] = reserrmu
        stats["PROJ-mu"] = projerrmu
        stats["PROJ-MAX-ZETA"] = 0
        stats["PROJ-MAX-INACTIVE-ZETA"] = 0
        stats["MI"] = [(mu, vec.basis.dim) for mu, vec in w.iteritems()]
        if (start_iteration == 0 or start_iteration < refinement):
            sim_stats.append(stats)
        print sim_stats[refinement]
        logger.debug("squared error components: eta=%s  delta=%s  zeta=%", errors[0], errors[1], errors[2])
        # inactive mi projection error
        logger.debug("evaluating ResidualEstimator.evaluateInactiveProjectionError")
        with timing(msg="ResidualEstimator.evaluateInactiveMIProjectionError", logfunc=logger.info):
            mierr = ResidualEstimator.evaluateInactiveMIProjectionError(w, coeff_field, maxh, newmi_add_maxm) 

        # exit when either error threshold or max_refinements is reached
        if refinement > max_refinements:
            logger.info("skipping refinement after final solution in iteration %i", refinement)
            break
        if xi <= error_eps:
            logger.info("error reached requested accuracy, xi=%f", xi)
            break

        # marking
        # -------
        if refinement < max_refinements:
            if not do_uniform_refinement:        
                logger.debug("starting Marking.mark")
                mesh_markers_R, mesh_markers_P, new_multiindices, proj_zeta = Marking.mark(resind, projind, mierr, w.max_order,
                                                                                theta_eta, theta_zeta, theta_delta,
                                                                                min_zeta, maxh, max_Lambda_frac)
                sim_stats[-1]["PROJ-MAX-ZETA"] = proj_zeta[0]
                sim_stats[-1]["PROJ-MAX-INACTIVE-ZETA"] = proj_zeta[1]
                logger.info("PROJECTION error values: max_zeta = %s  and  max_inactive_zeta = %s  with threshold factor theta_zeta = %s  (=%s)",
                            proj_zeta[0], proj_zeta[1], theta_zeta, theta_zeta * proj_zeta[0])
                logger.info("MARKING will be carried out with %s (res) + %s (proj) cells and %s new multiindices",
                            sum([len(cell_ids) for cell_ids in mesh_markers_R.itervalues()]),
                            sum([len(cell_ids) for cell_ids in mesh_markers_P.itervalues()]), len(new_multiindices))
                if do_refinement["RES"]:
                    mesh_markers = mesh_markers_R.copy()
                else:
                    mesh_markers = defaultdict(set)
                    logger.info("SKIP residual refinement")
    
                if do_refinement["PROJ"]:
                    for mu, cells in mesh_markers_P.iteritems():
                        if len(cells) > 0:
                            mesh_markers[mu] = mesh_markers[mu].union(cells)
                else:
                    logger.info("SKIP projection refinement")
    
                if not do_refinement["MI"] or refinement == max_refinements:
                    new_multiindices = {}
                    logger.info("SKIP new multiindex refinement")
            else:
                logger.info("UNIFORM REFINEMENT active")
                mesh_markers = {}
                for mu, vec in w.iteritems():
                    from dolfin import cells
                    mesh_markers[mu] = list([c.index() for c in cells(vec._fefunc.function_space().mesh())])
                new_multiindices = {}
            
            # carry out refinement of meshes
            with timing(msg="Marking.refine", logfunc=logger.info):
                Marking.refine(w, mesh_markers, new_multiindices.keys(), partial(setup_vector, pde=pde, mesh=mesh0, degree=degree))
    
    if refinement:
        logger.info("ENDED refinement loop after %i of %i refinements with %i dofs and %i active multiindices",
                    refinement, max_refinements, sim_stats[refinement]["DOFS"], len(sim_stats[refinement]["MI"]))

#    except Exception as ex:
#        import pickle
#        logger.error("EXCEPTION during AdaptiveSolver: %s", str(ex))
#        print "DIM of w:", w.dim
#        if not w_history is None:
#            w_history.append(w)
#        wname = "W-PCG-FAILED.pkl"
#        try:
#            with open(wname, 'wb') as fout:
#                pickle.dump(w, fout)
#        except Exception as ex:
#            logger.error("NEXT EXCEPTION %s", str(ex))
#        logger.info("exported last multivector w to %s", wname)
#    finally:
    return w, sim_stats
