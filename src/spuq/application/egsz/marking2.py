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
    @takes(anything, float, anything, float)
    def mark_x(cls, eta, eta_local, theta_x):
        """Carry out Doerfler marking (bulk criterion) for elements with parameter theta."""
        eta_local = sorted(eta_local, key=itemgetter(0), reverse=True)
        logger.info("(mark_x) global residual is %f, want to mark for %f", global_eta, theta_x * global_eta)
        # setup marking sets
        mesh_markers = defaultdict(set)
        marked_eta = 0.0
        for eta_cell in eta_local:
            if theta_x * global_eta <= marked_eta:
                break
            mesh_markers.add(res[1])
            marked_eta += res[0]
        logger.info("(mark_x) MARKED elements: %s", len(mesh_markers))
        return mesh_markers

    @classmethod
    @takes(anything, MultiVector, (list, tuple))
    def refine_x(cls, w, cell_ids):
        w.refine(cell_ids)

    @classmethod
    @takes(anything, float, dict, dict, anything, float, int)
    def mark_y(cls, Lambda, global_zeta, zeta, zeta_bar, eval_zeta_m, theta_y, max_new_mi=100):
        """Carry out Doerfler marking by activation of new indices."""
        def supp(Lambda):
            s = [set(mu.supp) for mu in Lambda]
            return set.union(*s)
        suppLambda = supp(Lambda)
        maxm = max(suppLambda)        
        new_mi = []
        marked_zeta = 0.0
        while True:
            zeta = sorted(zeta, key=itemgetter(1))
            mu = zeta[-1][0]
            new_mi.append(mu)
            marked_zeta += zeta[-1][1]
            # extend set if necessary (see section 5.7)
            try:
                mu2 = mu.dec(maxm)
                mu = Lambda[Lambda.index(mu2)]
                logger.debug("extending multiindex canidates since %s is at the boundary of Lambda (reachable from %s)", mu, mu2)
                minm = min(set(range(1, maxm + 2)).difference(set(suppLambda)))
                new_mu = mu.inc(minm)
                new_zeta = eval_zeta_m(mu, minm)
                zeta.append((new_mu, new_zeta))
            finally:
                logger.debug("no further extension of multiindex candidates required")
            
            # break if sufficiently many new mi are selected
            if theta_y * zeta <= marked_zeta or len(new_mi) >= max_new_mi or len(zeta) == 0:
                break 

        if len(zeta) == 0:
            logger.warning("list of mi candidates is empty")

        if len(new_mi) > 0:
            logger.info("SELECTED NEW MULTIINDICES %s", new_mi)
        else:
            logger.info("NO NEW MULTIINDICES SELECTED")
        return new_mi

    @classmethod
    @takes(anything, MultiVector, (list, tuple), callable)
    def refine_y(cls, w, new_mi, setup_vector):
        for mu in new_mi:
            w[mu] = setup_vector()

    @classmethod
    def refine_osc(cls, w, coeff, M):
        osc_refinements = 0
        # TODO
        return osc_refinements
    
