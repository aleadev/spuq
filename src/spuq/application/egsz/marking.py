"""Implementation of the EGSZ marking algorithm for the residual a posteriori error estimator.

The marking is carried out with respect to the
    [a] spatial residual
    [b] projection error between meshes
    [c] projection error of inactive multiindices. 
"""

from __future__ import division
from math import ceil
from operator import itemgetter
from collections import defaultdict
from itertools import count

from dolfin import Expression, Function, norm

from spuq.application.egsz.residual_estimator import ResidualEstimator
from spuq.application.egsz.multi_vector import MultiVector
from spuq.application.egsz.coefficient_field import CoefficientField
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
            vec = w[mu].refine(cell_ids, with_prolongation=False)
            w[mu] = eval_vec(vec)

        # add new multiindices to solution vector
        for mu in new_multiindices:
            w[mu] = eval_vec()
    

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, float, float, float, float)
    def mark(cls, w, coeff_field, f, theta_eta, theta_zeta, theta_delta, min_zeta, maxh):
        """Evaluate residual and projection errors, mark elements with bulk criterion and identify multiindices to activate."""
        mesh_markers_R = cls.mark_residual(w, coeff_field, f, theta_eta)
        mesh_markers_P, max_zeta = cls.mark_projection(w, coeff_field, theta_zeta, min_zeta, maxh)
        new_mi = cls.mark_inactive_multiindices(w, coeff_field, theta_delta, max_zeta)
        return mesh_markers_R, mesh_markers_P, new_mi
    

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, float)
    def mark_residual(cls, w, coeff_field, f, theta_eta):
        """Evaluate residual estimator and carry out Doerfler marking (bulk criterion) for elements with parameter theta."""
        # evaluate residual estimator
        resind, reserr = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, f)
        # residual marking
        # ================
        global_res = sum([res[1] for res in reserr.items()])
        allresind = list()
        for mu, resmu in resind.iteritems():
            allresind = allresind + [(resmu.coeffs[i], i, mu) for i in range(len(resmu.coeffs))]
        allresind = sorted(allresind, key=itemgetter(1))
        # TODO: check that indexing and cell ids are consistent (it would be safer to always work with cell indices) 
        # setup marking sets
        mesh_markers = defaultdict(set)
        marked_res = 0
        for res in allresind:
            if marked_res >= theta_eta * global_res:
                break
            mesh_markers[res[2]].add(res[1])
            marked_res += res[0]
        print "RES MARKED elements:\n", [(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()]
        return mesh_markers
    
    
    @classmethod
    @takes(anything, MultiVector, CoefficientField, float, optional(float), optional(float))
    def mark_projection(cls, w, coeff_field, theta_zeta, min_zeta=1e-10, maxh=1 / 10):
        """Evaluate projection error for active multiindices and determine multiindices to be refined."""
        # evaluate projection errors
        projind = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh)
        
        # testing -->
        if logger.isEnabledFor(logging.DEBUG):
            projglobal = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, local=False)
            for mu, val in projglobal.iteritems():
                logger.debug("GLOBAL Projection Error for %s = %f", mu, val)
        # <-- testing

        # projection marking
        # ==================
        # setup marking sets
        mesh_markers = defaultdict(set)
        max_zeta = max([max(projind[mu].coeffs) for mu in projind.active_indices()])
        logger.info("max_zeta = %f", max_zeta)
        if max_zeta >= min_zeta:
            for mu, vec in projind.iteritems():
                indmu = [i for i, p in enumerate(vec.coeffs) if p >= theta_zeta * max_zeta]
                mesh_markers[mu] = mesh_markers[mu].union(set(indmu)) 
                logger.debug("PROJ MARKING %i elements in %s", len(indmu), str(mu))
        
            logger.info("FINAL MARKED elements: %s", str([(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()]))
        else:
            logger.info("NO PROJECTION MARKING due to very small projection error")
        return mesh_markers, max_zeta
    
    
    @classmethod
    @takes(anything, MultiVector, CoefficientField, float, float, float, optional(int))
    def mark_inactive_multiindices(cls, w, coeff_field, theta_delta, max_zeta, maxm=10):
        """Estimate projection error for inactive indices and determine multiindices to be activated."""
        # new multiindex activation
        # =========================
        # determine possible new indices
        a0_f, _ = coeff_field[0]
        Ldelta = {}
        Delta = w.active_indices()     
        deltaN = int(ceil(0.1 * len(Delta)))               # max number new multiindices
        for mu in Delta:
            norm_w = norm(w[mu].coeffs, 'L2')
            for m in count(1):
                mu1 = mu.inc(m)
                if mu1 not in Delta:
                    if m > maxm or m >= len(coeff_field):  # or len(Ldelta) >= deltaN
                        break 
                    am_f, am_rv = coeff_field[m]
                    beta = am_rv.orth_polys.get_beta(1)
                    # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
                    f = Function(w[mu]._fefunc.function_space())
                    f.interpolate(a0_f)
                    min_a0 = min(f.vector().array())
                    f.interpolate(am_f)
                    max_am = max(f.vector().array())
                    ainfty = max_am / min_a0
                    assert isinstance(ainfty, float)
                    
                    logger.debug("A*** %f -- %f -- %f", beta[1], ainfty, norm_w)
                    logger.debug("B*** %f", beta[1] * ainfty * norm_w)
                    logger.debug("C*** %f", theta_delta, max_zeta)
                    logger.debug("D*** %f", theta_delta * max_zeta)
                    logger.debug("E*** %s", bool(beta[1] * ainfty * norm_w >= theta_delta * max_zeta))
                    
                    if beta[1] * ainfty * norm_w >= theta_delta * max_zeta:
                        val1 = beta[1] * ainfty * norm_w
                        if mu1 not in Ldelta.keys() or (mu1 in Ldelta.keys() and Ldelta[mu1] < val1):
                            Ldelta[mu1] = val1
                    
        logger.debug("POSSIBLE NEW MULTIINDICES %s", sorted(Ldelta.iteritems(), key=itemgetter(1), reverse=True))
        Ldelta = sorted(Ldelta.iteritems(), key=itemgetter(1), reverse=True)[:min(len(Ldelta), deltaN)]
        logger.info("SELECTED NEW MULTIINDICES %s", Ldelta)
        return dict(Ldelta)
