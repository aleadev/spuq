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


def pcg_solve(A, w, coeff_field, f, R, pcg_eps, pcg_maxiter):
    b = 0 * w
    b0 = b[Multiindex()]
    b0.coeffs = FEMPoisson.assemble_rhs(f, b0.basis)
    P = PreconditioningOperator(coeff_field.mean_func, FEMPoisson.assemble_solve_operator)
    w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)
    logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)
    b2 = A * w
    L2error = error_norm(b, b2, "L2")
    H1error = error_norm(b, b2, "H1")
    dofs = sum([b[mu]._fefunc.function_space().dim() for mu in b.keys()])
    elems = sum([b[mu]._fefunc.function_space().mesh().num_cells() for mu in b.keys()])
    R.append({"L2":L2error, "H1":H1error, "DOFS":dofs, "CELLS":elems})
    logger.info("Residual = [%s (L2)] [%s (H1)] with [%s dofs] and [%s cells]", L2error, H1error, dofs, elems)
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
                    maxh=1 / 10, # maximal mesh width for projection maximum norm evaluation
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
    sim_info = {}
    R = list()              # residual, estimator and dof progress
    if max_refinements > 0:
        # refinement loop
        for refinement in range(max_refinements):
            logger.info("************* REFINEMENT LOOP iteration %i *************", refinement + 1)
    
            # pcg solve
            # ---------
            w, zeta = pcg_solve(A, w, coeff_field, f, R, pcg_eps, pcg_maxiter)

            # error evaluation
            # ----------------
            xi, resind, projind = ResidualEstimator.evaluateError(w, coeff_field, f, zeta, gamma, ceta, cQ, 1 / 10)
            reserr = sqrt(sum([sum(resind[mu].coeffs ** 2) for mu in resind.keys()]))
            projerr = sqrt(sum([sum(projind[mu].coeffs ** 2) for mu in projind.keys()]))
            logger.info("Estimator Error = %s while residual error is %s and projection error is %s", xi, reserr, projerr)
            sim_info[refinement] = ([(mu, vec.basis.dim) for mu, vec in w.iteritems()], R[-1])
            R[-1]["EST"] = xi
            R[-1]["RES"] = reserr
            R[-1]["PROJ"] = projerr
            R[-1]["MI"] = len(sim_info[refinement][0])
            if xi <= error_eps:
                logger.info("error reached requested accuracy, xi=%f", xi)
                break
    
            #    # debug---
            #    projglobal, _ = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, local=False)
            #    for mu, val in projglobal.iteritems():
            #        logger.info("GLOBAL Projection Error for %s = %f", mu, val)
            #    logger.info("GLOBAL Projection Error = %f", sqrt(sum([v ** 2 for v in projglobal.itervalues()])))
            #    # ---debug
    
            # marking
            # -------
            if refinement < max_refinements - 1:
                if not do_uniform_refinement:
                    mesh_markers_R, mesh_markers_P, new_multiindices = \
                    Marking.mark(resind, projind, w, coeff_field, theta_eta, theta_zeta, theta_delta, min_zeta, maxh, maxm)
                    logger.info("MARKING will be carried out with %s cells and %s new multiindices", sum([len(cell_ids) for cell_ids in mesh_markers_R.itervalues()])
                    + sum([len(cell_ids) for cell_ids in mesh_markers_P.itervalues()]), len(new_multiindices))
                    if do_refinement["RES"]:
                        mesh_markers = mesh_markers_R.copy()
    
                    #                # debug---
                    #                # fully refine deterministic mesh
                    #                mm = w[Multiindex()]._fefunc.function_space().mesh()
                    #                mesh_markers[Multiindex()] = list(range(mm.num_cells()))
                    #                # ---debug
                    else:
                        mesh_markers = {}
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
                    #            # debug---
                    #            mu = Multiindex()
                    #            mesh_markers[mu] = list([c.index() for c in cells(w[mu]._fefunc.function_space().mesh())])
                    #            # ---debug
                    new_multiindices = {}
                Marking.refine(w, mesh_markers, new_multiindices.keys(), partial(setup_vec, mesh=mesh0))
        logger.info("ENDED refinement loop at refinement %i with %i dofs and %i active multiindices",
                    refinement, sim_info[refinement][1]["DOFS"], len(sim_info[refinement][0]))
    else:
        w, _ = pcg_solve(A, w, coeff_field, f, R, pcg_eps, pcg_maxiter)
        sim_info[0] = ([(mu, vec.basis.dim) for mu, vec in w.iteritems()], R[-1])
        logger.info("Residuals: %s", R)
        logger.info("Simulation run data: %s", sim_info)
    return w, {'res': R, 'sim_info':sim_info}
