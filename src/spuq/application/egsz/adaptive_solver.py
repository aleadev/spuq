from __future__ import division
from functools import partial
from collections import defaultdict
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
def setup_vector(mesh, pde, degree=1):
#    fs = FunctionSpace(mesh, "CG", degree)
    fs = pde.function_space(mesh, degree=degree)
    vec = FEniCSVector(Function(fs))
    return vec


def pcg_solve(A, w, coeff_field, pde, rhs, stats, pcg_eps, pcg_maxiter):
    b = 0 * w
    b0 = b[Multiindex()]
    b0.coeffs = rhs(basis=b0.basis)
    P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
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
                    w_history=None):
    f = pde.f
    rhs = pde.assemble_rhs

    w = w0
    if not w_history is None:
        w_history.append(w)

    # data collection
    sim_stats = []                  # mis, residual, estimator and dof progress
    for refinement in range(max_refinements + 1):
        logger.info("************* REFINEMENT LOOP iteration %i *************", refinement)
        # memory usage info
        import resource
        logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")

        # pcg solve
        # ---------
        stats = {}
        w, zeta = pcg_solve(A, w, coeff_field, pde, rhs, stats, pcg_eps, pcg_maxiter)
        if not w_history is None:
            w_history.append(w)

#        print "===== SOLUTION w"
#        for mu in w.active_indices():
#            print "for mu:", w[mu]
#            print w[mu].array()

        # error evaluation
        # ----------------
        # residual and projection errors
        xi, resind, projind = ResidualEstimator.evaluateError(w, coeff_field, pde, f, zeta, gamma, ceta, cQ, maxh, quadrature_degree, projection_degree_increase, refine_projection_mesh)
        reserr = sqrt(sum([sum(resind[mu].coeffs ** 2) for mu in resind.keys()]))
        projerr = sqrt(sum([sum(projind[mu].coeffs ** 2) for mu in projind.keys()]))
        logger.info("Overall Estimator Error xi = %s while residual error is %s and projection error is %s", xi, reserr, projerr)
        stats["EST"] = xi
        stats["RES"] = reserr
        stats["PROJ"] = projerr
        stats["MI"] = [(mu, vec.basis.dim) for mu, vec in w.iteritems()]
        sim_stats.append(stats)
        # inactice mi projection error
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
        if not do_uniform_refinement:
            mesh_markers_R, mesh_markers_P, new_multiindices = Marking.mark(resind, projind, mierr, w.max_order,
                                                                            theta_eta, theta_zeta, theta_delta,
                                                                            min_zeta, maxh, max_Lambda_frac)
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
        Marking.refine(w, mesh_markers, new_multiindices.keys(), partial(setup_vector, pde=pde, mesh=mesh0, degree=degree))

    logger.info("ENDED refinement loop after %i of %i refinements with %i dofs and %i active multiindices",
                refinement, max_refinements, sim_stats[refinement]["DOFS"], len(sim_stats[refinement]["MI"]))

    return w, sim_stats
