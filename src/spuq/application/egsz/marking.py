"""Implementation of the EGSZ marking algorithm for the residual a posteriori error estimator.

The marking is carried out with respect to the
    [a] spatial residual
    [b] projection error between meshes
    [c] projection error of inactive multiindices. 
"""

from __future__ import division
from math import ceil
from collections import defaultdict
from operator import itemgetter

from spuq.application.egsz.residual_estimator import ResidualEstimator
from spuq.application.egsz.multi_vector import MultiVector
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.fem.fenics.fenics_utils import weighted_H1_norm
from spuq.utils.type_check import takes, anything, optional

import logging

logger = logging.getLogger(__name__)

class Marking(object):
    """EGSZ marking strategy for residual estimator."""

    @classmethod
    @takes(anything, MultiVector, dict, list, callable)
    def refine(cls, w, mesh_markers, new_multiindices, eval_vec):
        """ """
        # create new refined (and enlarged) multi vector
        for mu, cell_ids in mesh_markers.iteritems():
            logger.info("REFINE: refining %s of %s cells for mesh of mu %s", len(cell_ids), w[mu]._fefunc.function_space().mesh().num_cells(), mu)
            w[mu] = w[mu].refine(cell_ids, with_prolongation=True)

        # determine current mesh sizes
        minh, maxh = 1e6, 0
        for mu in w.active_indices():
            mesh = w[mu].mesh
            minH, maxH = mesh.hmin(), mesh.hmax()
            if minH < minh:
                minh = minH 
            if maxH > maxh:
                maxh = maxH 
        logger.info("REFINE: current meshes minh = %s and maxh = %s", minh, maxh)

        # add new multiindices to solution vector
        for mu in new_multiindices:
            logger.info("REFINE: adding new multiindex %s", mu)
            w[mu] = eval_vec()
            logger.info("REFINE: new mesh maxh = %s", w[mu].mesh.hmax())


    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, float, float, float, float, optional(float), optional(int),
           optional(int), optional(bool))
    def estimate_mark(cls, w, coeff_field, pde, f, theta_eta, theta_zeta, theta_delta, min_zeta, maxh=1 / 10, maxm=10,
                       quadrature_degree= -1, projection_degree_increase=1, refine_projection_mesh=1):
        """Convenience method which evaluates the residual and the projection indicators and then calls the marking algorithm."""
        #        # testing -->
        #        if logger.isEnabledFor(logging.DEBUG):
        #            projglobal, _ = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, local=False)
        #            for mu, val in projglobal.iteritems():
        #                logger.debug("GLOBAL Projection Error for %s = %f", mu, val)
        #        # <-- testing

        # evaluate residual estimator
        resind, _ = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, pde, f, quadrature_degree)
        # evaluate projection errors
        projind, _ = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, projection_degree_increase, refine_projection_mesh)
        # evaluate inanctive mi projection error
        mierr = ResidualEstimator.evaluateInactiveMIProjectionError(w, coeff_field, maxh, maxm)
        # mark
        return cls.mark(resind, projind, mierr, w.max_order, theta_eta, theta_zeta, theta_delta, min_zeta, maxh, maxm)


    @classmethod
    @takes(anything, MultiVector, MultiVector, list, int, float, float, float, float, optional(float))
    def mark(cls, resind, projind, mierr, maxorder_Lambda, theta_eta, theta_zeta, theta_delta, min_zeta, maxh=1 / 10, max_Lambda_frac=1 / 10):
        """Evaluate residual and projection errors, mark elements with bulk criterion and identify multiindices to activate."""
        mesh_markers_R = cls.mark_residual(resind, theta_eta)
        mesh_markers_P, max_zeta = cls.mark_projection(projind, theta_zeta, min_zeta, maxh)
        max_inactive_mi_zeta = 0
        if max_zeta >= min_zeta:
            new_mi, max_inactive_mi_zeta = cls.mark_inactive_multiindices(mierr, theta_delta, max_zeta, maxorder_Lambda, max_Lambda_frac)
        else:
            new_mi = {}
            logger.info("SKIPPING search for new multiindices due to very small max_zeta = %s", max_zeta)
        return mesh_markers_R, mesh_markers_P, new_mi, (max_zeta, max_inactive_mi_zeta)


    @classmethod
    @takes(anything, MultiVector, float)
    def mark_residual(cls, resind, theta_eta):
        """Evaluate residual estimator and carry out Doerfler marking (bulk criterion) for elements with parameter theta."""
        # residual marking
        # ================

        #        if logger.isEnabledFor(logging.DEBUG):
        #            for mu, cellres in resind.iteritems():
        #                logger.debug("resind[%s] = %s", mu, cellres)

        allresind = list()
        for mu, resmu in resind.iteritems():
            allresind = allresind + [(resmu.coeffs[i], i, mu) for i in range(len(resmu.coeffs))]
        allresind = sorted(allresind, key=itemgetter(0), reverse=True)
        global_res = sum([res[0] for res in allresind])
        logger.info("(mark_residual) global residual is %f, want to mark for %f", global_res, theta_eta * global_res)
        # TODO: check that indexing and cell ids are consistent (it would be safer to always work with cell indices) 
        # setup marking sets
        mesh_markers = defaultdict(set)
        marked_res = 0.0
        for res in allresind:
            if marked_res >= theta_eta * global_res:
                break
            mesh_markers[res[2]].add(res[1])
            marked_res += res[0]
        logger.info("(mark_residual) MARKED elements: %s",
            [(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()])
        return mesh_markers


    @classmethod
    @takes(anything, MultiVector, float, optional(float), optional(float))
    def mark_projection(cls, projind, theta_zeta, min_zeta=1e-10, maxh=1 / 10):
        """Evaluate projection error for active multiindices and determine multiindices to be refined."""
        # projection marking
        # ==================
        # setup marking sets
        mesh_markers = defaultdict(set)
        max_zeta = max([max(projind[mu].coeffs) for mu in projind.active_indices()])
        logger.info("max_zeta = %f", max_zeta)

        #        if logger.isEnabledFor(logging.DEBUG):
        #            for mu, cellproj in projind.iteritems():
        #                logger.debug("projind[%s] = %s", mu, cellproj)

        if max_zeta >= min_zeta:
            for mu, vec in projind.iteritems():
                indmu = [i for i, p in enumerate(vec.coeffs) if p >= theta_zeta * max_zeta]
                mesh_markers[mu] = mesh_markers[mu].union(set(indmu))
                logger.debug("PROJ MARKING %i elements in %s", len(indmu), mu)

            logger.info("FINAL MARKED elements: %s",
                str([(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()]))
        else:
            logger.info("NO PROJECTION MARKING due to very small projection error")
        return mesh_markers, max_zeta


    @classmethod
    @takes(anything, list, float, float, int, optional(float))
    def mark_inactive_multiindices(cls, Lambda_candidates, theta_delta, max_zeta, maxorder_Lambda, max_Lambda_frac=1 / 10):
        """Determine multiindices to be activated."""
        # new multiindex activation
        # =========================
        zeta_threshold = theta_delta * max_zeta
        lambdaN = int(ceil(max_Lambda_frac * maxorder_Lambda))                    # max number new multiindices
        # select indices with largest projection error
        Lambda_selection = sorted(Lambda_candidates, key=itemgetter(1), reverse=True)[:min(len(Lambda_candidates), lambdaN)]
        try:
            lambda_max = Lambda_selection[0][1]
        except:
            lambda_max = -1
        # apply threshold criterion
        Lambda_selection = [l for l in Lambda_selection if l[1] >= zeta_threshold]
        if len(Lambda_selection) > 0:
            logger.info("SELECTED NEW MULTIINDICES (zeta_thresh = %s, lambda_max = %s) %s", zeta_threshold, lambda_max, Lambda_selection)
        else:
            logger.info("NO NEW MULTIINDICES SELECTED")
        return dict(Lambda_selection), lambda_max
