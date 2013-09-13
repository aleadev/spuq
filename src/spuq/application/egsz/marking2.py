"""Implementation of the EGSZ2 marking algorithm for the residual a posteriori error estimator.

The marking is carried out with respect to the
    [a] spatial residual
    [b] upper tail bound for inactive multiindices
    [c] some oscillation condition of the coefficient. 
"""

from __future__ import division
from math import fabs
from operator import itemgetter

from spuq.application.egsz.multi_vector import MultiVector, supp
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
#            print "ETA_CELL", eta_cell, eta_cell[0] ** 2, marked_eta
            mesh_markers.add(eta_cell[1])
            marked_eta += eta_cell[0] ** 2
        
#        # DEBUG ---
#        if len(eta_local) / len(mesh_markers) > 10 or len(mesh_markers) < 10:
#            print "X"*20, eta_local_ind[:20]
#            print "Y"*20, sorted([x for x in eta_local], reverse=True)[:20]
#            print "Z"*20, sorted([x ** 2 for x in eta_local], reverse=True)[:20]
#            print "A"*20, sum([x ** 2 for x in eta_local])
#        # --- DEBUG
        
        logger.info("(mark_x) MARKED elements: %s (of %s)", len(mesh_markers), len(eta_local))
        return mesh_markers

    @classmethod
    @takes(anything, MultiVector, (list, tuple, set))
    def refine_x(cls, w, cell_ids):
        return w.refine(cell_ids)

    @classmethod
    @takes(anything, (list, tuple), dict, callable, float, int)
    def mark_y(cls, Lambda, zeta_, eval_zeta_m, theta_y, max_new_mi=100, type=0):
        """Carry out Doerfler marking by activation of new indices."""
        zeta = zeta_
        global_zeta = np.sqrt(sum([z ** 2 for z in zeta_.values()]))
        suppLambda = supp(Lambda)
        maxm = max(suppLambda)
        logger.debug("---- SUPPORT Lambda %s   maxm %s   Lambda %s ", suppLambda, maxm, Lambda)
        # A modified paper marking
        # ========================
        if type == 0:
            new_mi = []
            marked_zeta = 0.0
            while True:
                # break if sufficiently many new mi are selected
                if theta_y * global_zeta <= marked_zeta or len(new_mi) >= max_new_mi or len(zeta) == 0:
                    if len(new_mi) >= max_new_mi:
                        logger.warn("max new_mi reached (%i) WITHOUT sufficient share of global zeta!" % len(new_mi))
                    if len(zeta) == 0:
                        logger.warn("NO MORE MI TO MARK!")
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
#                mu2 = mu.dec(maxm)
                # NOTE: the following is a slight extension of the algorithm in the paper since it executed the extension on all active multiindices (and not only with the latest activated)
    #            if mu2 in Lambda:
                minm = min(set(range(1, maxm + 2)).difference(set(suppLambda))) # find min(N\setminus supp\Lambda)
                for mu2 in Lambda:
                    new_mu = mu2.inc(minm)
    #                assert new_mu not in Lambda
#                    if new_mu not in Lambda and new_mu not in zeta.keys():
                    if new_mu not in zeta.keys():
#                        logger.debug("extending multiindex candidates by %s since %s is at the boundary of Lambda (reachable from %s), minm: %s", new_mu, mu, mu2, minm)
                        logger.debug("extending multiindex candidates by %s since it is at the boundary of Lambda (reachable from %s), minm: %s", new_mu, mu2, minm)
                        zeta[new_mu] = eval_zeta_m(mu2, minm)
                        # update global zeta
                        global_zeta = np.sqrt(global_zeta ** 2 + zeta[new_mu] ** 2)
                        logger.debug("new global_zeta is %f", global_zeta)
                else:
                    logger.debug("no further extension of multiindex candidates required")
                    if len(new_mi) >= max_new_mi:
                        logger.debug("maximal number new mi reached!")
                    elif len(zeta) == 0:
                        logger.debug("no more new indices available!")
                        
        # B minimal y-dimension marking
        # =============================
        else:
            assert type == 1
            target_zeta = theta_y * global_zeta
             
            # === EVALUATE EXTENSION ===
            # ==========================
            
            # determine possible new mi
            new_y = {}
            minm = min(set(range(1, maxm + 2)).difference(set(suppLambda))) # find min(N\setminus supp\Lambda)
            for mu2 in Lambda:
                new_mu = mu2.inc(minm)
#                assert new_mu not in Lambda
#                if new_mu not in Lambda and new_mu not in zeta.keys() and new_mu not in new_y.keys():
                if new_mu not in zeta.keys() and new_mu not in new_y.keys():
                    logger.debug("extending multiindex candidates by %s since it is at the boundary of Lambda (reachable from %s), minm: %s", new_mu, mu2, minm)
                    new_val = eval_zeta_m(mu2, minm)
                    # update global zeta
                    global_zeta = np.sqrt(global_zeta ** 2 + new_val ** 2)
                    logger.debug("new global_zeta is %f", global_zeta)
                    # test for new y dimension
                    if len(set(supp([new_mu])).difference(set(suppLambda))) > 0:
                        new_y[new_mu] = new_val
                    else: 
                        zeta[new_mu] = new_val                         
                else:
                    logger.debug("no further extension of multiindex candidates required")
            
            # === DETERMINE NEW Y DIMENSIONS ===
            # ==================================
            
            # determine how many new y dimensions are needed
            new_mi = []
            sorted_new_y = sorted(new_y.items(), key=itemgetter(1))
            sum_zeta_val = np.sqrt(sum([z ** 2 for z in zeta.values()]))
            # add new dimension y while sum_zeta_val is smaller than required marking value
            while sum_zeta_val < target_zeta and len(sorted_new_y) > 0:
                # add new largest y
                new_zeta = sorted_new_y[-1]
                mu = new_zeta[0]
                sorted_new_y.pop(mu)
                logger.debug("ADDING NEW Y %s to new_mi %s while target_zeta is %s", mu, new_mi, target_zeta)
                assert mu not in Lambda
                new_mi.append(mu)
                target_zeta = np.sqrt(target_zeta ** 2 - new_zeta[1] ** 2)
            
            if len(sorted_new_y) == 0 and zeta_val < target_zeta:
                logger.warn("UNABLE to mark sufficiently many NEW MI!") 

            # === DETERMINE HIGHER ORDER ACTIVE MI EXTENSION ===
            # ==================================================

            # add mi corresponding to already active y dimensions
            sorted_zeta = sorted(zeta.items(), key=itemgetter(1))
            logger.debug("SORTED ZETA %s", sorted_zeta)
            marked_zeta = 0
            while marked_zeta < target_zeta and len(sorted_zeta) > 0:
                new_zeta = sorted_zeta[-1]
                mu = new_zeta[0]
                zeta.pop(mu)
                logger.debug("ADDING EXTENSION OF EXISTING MI %s to new_mi %s while marked_zeta is %s", mu, new_mi, marked_zeta)
                assert mu not in Lambda
                new_mi.append(mu)
                marked_zeta = np.sqrt(marked_zeta ** 2 + new_zeta[1] ** 2)
            zeta = sorted_zeta

        if len(zeta) == 0:
            if target_zeta > marked_zeta:
                logger.warning("list of mi candidates is empty and reduction goal NOT REACHED, %f > %f!", theta_y * global_zeta, marked_zeta)

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
    def refine_osc(cls, w, coeff_field, a=1):
        Cadelta = 1.0
        mesh_maxh = w.basis.basis.mesh.hmax()
        coeff_min_val, coeff_max_grad = 1e10, 0.0
        suppLambda = supp(w.active_indices())
        if len(suppLambda) > 0:
            try:
#                 a0_f = coeff_field.mean_func
#                print "suppLambda", suppLambda
                for m in suppLambda:
                    coeff, _ = coeff_field[m]
                    min_val, max_grad = abs(coeff.min_val), abs(coeff.max_grad)
                    coeff_min_val, coeff_max_grad = min(coeff_min_val, min_val), max(coeff_max_grad, max_grad)
                    logger.debug("\tm:", m, min_val, max_grad, coeff_min_val, coeff_max_grad)
                # determine (4.14) c_{a,\delta}
                Cadelta = mesh_maxh * coeff_max_grad / coeff_min_val
            except:
                logger.error("coefficient does not provide max_val and max_grad. OSC refinement not supported for this case...")
        
            # determine maximal mesh size to resolve coefficient oscillations
            logger.debug("OSC marking maxh {0} and Cadelta {1} with scaling factor {2}".format(mesh_maxh, Cadelta, a))
            print "OSC marking maxh {0} and Cadelta {1} with scaling factor {2}".format(mesh_maxh, Cadelta, a)
            maxh = a * mesh_maxh / Cadelta
            
            # create appropriate mesh by refinement and project current solution
            new_w = w.refine_maxh(maxh)
            return new_w, maxh, Cadelta
        else:
            logger.info("SKIP OSC refinement since only active mi is deterministic.")
            return w, 1.0, Cadelta
