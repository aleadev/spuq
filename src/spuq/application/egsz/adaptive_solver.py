from __future__ import division
from functools import partial
from math import sqrt
import logging
import os

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import PreconditioningOperator
from spuq.math_utils.multiindex import Multiindex
try:
    from dolfin import (Function, FunctionSpace, cells)
    from spuq.application.egsz.marking import Marking
    from spuq.application.egsz.residual_estimator import ResidualEstimator
    from spuq.application.egsz.fem_discretisation import FEMPoisson
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
def setup_vec(mesh):
    fs = FunctionSpace(mesh, "CG", 1)
    vec = FEniCSVector(Function(fs))
    return vec


def pcg_solve(A, w, coeff_field, f, stats, pcg_eps, pcg_maxiter):
    b = 0 * w
    b0 = b[Multiindex()]
    b0.coeffs = FEMPoisson.assemble_rhs(f, b0.basis)
    P = PreconditioningOperator(coeff_field.mean_func, FEMPoisson.assemble_solve_operator)
    w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)
    logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)
    b2 = A * w
    stats["L2"] = error_norm(b, b2, "L2")
    stats["H1"] = error_norm(b, b2, "H1")
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
def AdaptiveSolver(A, coeff_field, f,
                    mis, w0, mesh0,
                    gamma=0.9,
                    cQ=1.0,
                    ceta=1.0,
                    # marking parameters
                    theta_eta=0.6, # residual marking bulk parameter
                    theta_zeta=0.5, # projection marking threshold factor
                    min_zeta=1e-15, # minimal projection error considered
                    maxh=0.1, # maximal mesh width for projection maximum norm evaluation
                    maxm=10, # maximal search length for new new multiindices
                    theta_delta=0.1, # number new multiindex activation bound
                    # pcg solver
                    pcg_eps=1e-6,
                    pcg_maxiter=100,
                    error_eps=1e-2,
                    # refinements
                    max_refinements=7,
                    do_refinement={"RES":True, "PROJ":True, "MI":False},
                    do_uniform_refinement=False):
    w = w0
    
    # data collection
    sim_info = []
    R = []              # residual, estimator and dof progress
    # refinement loop
    for refinement in range(max_refinements + 1):
        logger.info("************* REFINEMENT LOOP iteration %i *************", refinement)

        # pcg solve
        # ---------
        w, zeta = pcg_solve(A, w, coeff_field, f, R, pcg_eps, pcg_maxiter)


        #sim_info[0] = ([(mu, vec.basis.dim) for mu, vec in w.iteritems()], R[-1])
        #logger.info("Residuals: %s", R)
        #logger.info("Simulation run data: %s", sim_info)


        # error evaluation
        # ----------------
        xi, resind, projind = ResidualEstimator.evaluateError(w, coeff_field, f, zeta, gamma, ceta, cQ, maxh)
        reserr = sqrt(sum([sum(resind[mu].coeffs ** 2) for mu in resind.keys()]))
        projerr = sqrt(sum([sum(projind[mu].coeffs ** 2) for mu in projind.keys()]))
        logger.info("Estimator Error = %s while residual error is %s and projection error is %s", xi, reserr, projerr)

        sim_info.append(([(mu, vec.basis.dim) for mu, vec in w.iteritems()],
                        sum(vec.basis.dim for mu, vec in w.iteritems())))
        R.append( {"EST": xi, "RES": reserr, "PROJ": projerr, "MI": len(w.active_indices())})

        # exit, when either error threshold or max_refinements is reached
        if xi <= error_eps:
            logger.info("error reached requested accuracy, xi=%f", xi)
            break

        if refinement >= max_refinements:
            break

        # marking
        # -------
        if not do_uniform_refinement:
            mesh_markers_R, mesh_markers_P, new_multiindices = Marking.mark(resind, projind, w, coeff_field,
                                                                            theta_eta, theta_zeta, theta_delta,
                                                                            min_zeta, maxh, maxm)
            logger.info("MARKING will be carried out with %s cells and %s new multiindices",
                        sum([len(cell_ids) for cell_ids in mesh_markers_R.itervalues()])
                        + sum([len(cell_ids) for cell_ids in mesh_markers_P.itervalues()]), len(new_multiindices))
            if do_refinement["RES"]:
                mesh_markers = mesh_markers_R.copy()
            else:
                logger.info("UNIFORM REFINEMENT active")
                mesh_markers = {}
                logger.info("SKIP residual refinement")

            if do_refinement["PROJ"]:
                for mu, mucells in mesh_markers_P.iteritems():
                    if len(mucells) > 0:
                        mesh_markers[mu] = mesh_markers[mu].union(mucells)
            else:
                logger.info("SKIP projection refinement")

            if not do_refinement["MI"]:
                new_multiindices = {}
                logger.info("SKIP new multiindex refinement")
        else:
            logger.info("UNIFORM REFINEMENT active")
            mesh_markers = {}
            for mu, vec in w.iteritems():
                mesh_markers[mu] = list([c.index() for c in cells(vec._fefunc.function_space().mesh())])
            new_multiindices = {}
        Marking.refine(w, mesh_markers, new_multiindices.keys(), partial(setup_vec, mesh=mesh0))

    logger.info("ENDED refinement loop at refinement %i with %i dofs and %i active multiindices",
                refinement, sim_info[-1][1], len(w.active_indices()))

    return w, {'res': R, 'sim_info':sim_info}
