"""Implementation of the EGSZ2 marking algorithm for the residual a posteriori error estimator.

The marking is carried out with respect to the
    [a] spatial residual
    [b] upper tail bound for inactive multiindices. 
"""

from __future__ import division
from math import ceil, fabs
from collections import defaultdict
from operator import itemgetter

from spuq.application.egsz.residual_estimator import ResidualEstimator
from spuq.application.egsz.multi_vector import MultiVector, supp
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.fem.fenics.fenics_utils import weighted_H1_norm
from spuq.utils.type_check import takes, anything, optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

class Marking(object):
    """EGSZ2 marking strategy for residual estimator."""

    @classmethod
    @takes(anything, float, MultiVector, float)
    def mark_x(cls, global_eta, eta_local, theta_x):
        """Carry out Doerfler marking (bulk criterion) for elements with parameter theta."""
        # add indicators of all mu and sort
        eta_list = [err for err in eta_local.values()]
        eta_local = eta_list.pop().as_array() ** 2
        while len(eta_list) > 0:
            eta_local += eta_list.pop().as_array() ** 2
        eta_local = np.sqrt(eta_local)
        eta_local_ind = [(x, i) for i, x in enumerate(eta_local)]
        eta_local_ind = sorted(eta_local_ind, key=itemgetter(0), reverse=True)
#        print "XXXXX", eta_local_ind
        logger.info("(mark_x) global residual is %f, want to mark for %f", global_eta, theta_x * global_eta)
        # verify global eta by summing up
        assert fabs(global_eta - np.sqrt(sum([x ** 2 for x in eta_local]))) < 1e-10
        
        # setup marking sets
        mesh_markers = set()
        marked_eta = 0.0
        for eta_cell in eta_local_ind:
            # break if sufficiently many cells are selected
            if theta_x * global_eta <= np.sqrt(marked_eta):
                break
            print "ETA_CELL", eta_cell, eta_cell[0] ** 2, marked_eta
            mesh_markers.add(eta_cell[1])
            marked_eta += eta_cell[0] ** 2
        logger.info("(mark_x) MARKED elements: %s (of %s)", len(mesh_markers), len(eta_local))
        return mesh_markers

    @classmethod
    @takes(anything, MultiVector, (list, tuple, set))
    def refine_x(cls, w, cell_ids):
        return w.refine(cell_ids)

    @classmethod
    @takes(anything, (list, tuple), dict, callable, float, int)
    def mark_y(cls, Lambda, zeta_, eval_zeta_m, theta_y, max_new_mi=100):
        """Carry out Doerfler marking by activation of new indices."""
        zeta = zeta_
        global_zeta = np.sqrt(sum([z ** 2 for z in zeta_.values()]))
        suppLambda = supp(Lambda)
        maxm = max(suppLambda)
        logger.debug("---- SUPPORT Lambda %s   maxm %s   Lambda %s ", suppLambda, maxm, Lambda)
        new_mi = []
        marked_zeta = 0.0
        while True:
            # break if sufficiently many new mi are selected
            if theta_y * global_zeta <= marked_zeta or len(new_mi) >= max_new_mi or len(zeta) == 0:
                break
            sorted_zeta = sorted(zeta.items(), key=itemgetter(1))
            logger.debug("SORTED ZETA %s", sorted_zeta)
            new_zeta = sorted_zeta[-1]
            mu = new_zeta[0]
            zeta.pop(mu)
            logger.debug("ADDING %s to new_mi %s", mu, new_mi)
            assert mu not in Lambda
            new_mi.append(mu)
            marked_zeta = np.sqrt(marked_zeta ** 2 + new_zeta[1] ** 2)
            # extend set of inactive potential indices if necessary (see section 5.7)
            mu2 = mu.dec(maxm)
            # NOTE: the following is a slight extension of the algorithm in the paper since it executed the extension on all active multiindices (and not only with the latest activated)
#            if mu2 in Lambda:
            for mu2 in Lambda:
                minm = min(set(range(1, maxm + 2)).difference(set(suppLambda))) # find min(N\setminus supp\Lambda)
                new_mu = mu2.inc(minm)
#                assert new_mu not in Lambda
                if new_mu not in Lambda and new_mu not in zeta.keys():
                    logger.debug("extending multiindex candidates by %s since %s is at the boundary of Lambda (reachable from %s), minm: %s", new_mu, mu, mu2, minm)
                    zeta[new_mu] = eval_zeta_m(mu2, minm)
                    global_zeta = np.sqrt(global_zeta ** 2 + zeta[new_mu] ** 2)
                    logger.debug("new global_zeta is %f", global_zeta)
            else:
                logger.debug("no further extension of multiindex candidates required")

        if len(zeta) == 0:
            if theta_y * global_zeta > marked_zeta:
                logger.warning("list of mi candidates is empty and reduction goal not reached, %f > %f!", theta_y * global_zeta, marked_zeta)

        if len(new_mi) > 0:
            logger.info("SELECTED NEW MULTIINDICES %s", new_mi)
        else:
            logger.info("NO NEW MULTIINDICES SELECTED")
        return new_mi

    @classmethod
    @takes(anything, MultiVector, (list, tuple), callable)
    def refine_y(cls, w, new_mi):
        V = w.basis
        for mu in new_mi:
            w[mu] = V.basis.new_vector()

    @classmethod
    def refine_osc(cls, w, coeff, M):
        osc_refinements = 0
        # TODO
        return osc_refinements
    
