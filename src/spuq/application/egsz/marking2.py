"""Implementation of the EGSZ2 marking algorithm for the residual a posteriori error estimator.

The marking is carried out with respect to the
    [a] spatial residual
    [b] upper tail bound for inactive multiindices. 
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
    """EGSZ2 marking strategy for residual estimator."""

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, float, optional(int))
    def mark_x(cls, w, coeff_field, pde, f, theta_x, quadrature_degree= -1):
        """Carry out Doerfler marking (bulk criterion) for elements with parameter theta."""
        # evaluate residual indicators
        resind, _ = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, pde, f, quadrature_degree)


        #        if logger.isEnabledFor(logging.DEBUG):
        #            for mu, cellres in resind.iteritems():
        #                logger.debug("resind[%s] = %s", mu, cellres)

        allresind = list()
        for mu, resmu in resind.iteritems():
            allresind = allresind + [(resmu.coeffs[i], i, mu) for i in range(len(resmu.coeffs))]
        allresind = sorted(allresind, key=itemgetter(0), reverse=True)
        global_res = sum([res[0] for res in allresind])
        logger.info("(mark_residual) global residual is %f, want to mark for %f", global_res, theta_eta * global_res)
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
    @takes(anything, list, float, float, int, optional(float), optional(str))
    def mark_y(cls, w, coeff_field, pde, theta_delta, maxh=1 / 10, add_maxm=10):
        """Carry out Doerfler marking by activation of new indices."""
        # evaluate upper tail bound
        z, zeta, zeta_bar, eval_zeta_m = ResidualEstimator.evaluateUpperTailBound(cls, w, coeff_field, pde, maxh, add_maxm)




        
        zeta_threshold = theta_delta * max_zeta
        lambdaN = int(ceil(max_Lambda_frac * maxorder_Lambda))                    # max number new multiindices
        # select indices with largest projection error
        Lambda_selection_all = sorted(Lambda_candidates, key=itemgetter(1), reverse=True)
        Lambda_selection = Lambda_selection_all[:min(len(Lambda_candidates), lambdaN)]
        try:
            lambda_max = Lambda_selection[0][1]
#            assert lambda_max == max([v for v in Lambda_selection.values()])
        except:
            lambda_max = -1
        # apply threshold criterion
        Lambda_selection = [l for l in Lambda_selection if l[1] >= zeta_threshold]
        if len(Lambda_selection) > 0:
            logger.info("SELECTED NEW MULTIINDICES (zeta_thresh = %s, lambda_max = %s) %s", zeta_threshold, lambda_max, Lambda_selection)
        else:
            logger.info("NO NEW MULTIINDICES SELECTED")
        return dict(Lambda_selection), lambda_max, dict(Lambda_selection_all)
