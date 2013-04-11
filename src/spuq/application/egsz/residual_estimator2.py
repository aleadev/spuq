"""EGSZ2 a posteriori residual estimator (FEniCS centric implementation)

The residual estimator consists of a volume term :math:`\eta_{\mu,T}`, an edge term
:math:`\eta_{\mu,S}` and the projection error :math:`\delta_\mu`. The former two terms
are based on the flux of the discrete solution while the latter term measures the
projection error between different FE meshes.

In an extended form for the more generic orthonormal polynomials in spuq, the three
terms are defined for some discrete :math:`w_N\in\mathcal{V}_N` by


.. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
          \delta_\mu(w_N) &:= \sum_{m=1}^\infty || a_m/\overline{a} ||_{L^\infty(D)
                          ||| \alpha_{\mu+1}^m \nabla(\Pi_{\mu+e_m}^\mu (\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}) ) - w_{N,\mu+e_m} |||
                          + ||| \alpha_{\mu-1}^m \nabla(\Pi_{\mu-e_m}^\mu (\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m}) ) - w_{N,\mu-e_m} |||


The coefficients :math:`\alpha_j` follow from the recurrence coefficients
:math:`a_n,b_m,c_m` of the orthonormal polynomials by

.. math::
        \alpha_{n-1} &:= c_n/b_n \\
        \alpha_n &:= a_n/b_n \\
        \alpha_{n+1} &:= 1/b_n
"""

from __future__ import division
import numpy as np
from operator import itemgetter

from dolfin import (assemble, dot, nabla_grad, dx, avg, dS, sqrt, norm, VectorFunctionSpace, cells,
                    Constant, FunctionSpace, TestFunction, CellSize, FacetNormal, parameters)

from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.fem.fenics.fenics_utils import weighted_H1_norm
from spuq.linalg.vector import FlatVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything, list_of, optional
from spuq.utils.timing import timing

import logging
logger = logging.getLogger(__name__)

class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.

    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
    fenics/dolfin implementation is based on
    https://answers.launchpad.net/dolfin/+question/177108
    """

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, float, float, float, float, int, optional(float), optional(int))
    def evaluateError(cls, w, coeff_field, pde, f, zeta, gamma, ceta, cQ, newmi_add_maxm, maxh=0.1, quadrature_degree= -1):
        """Evaluate EGSZ Error (7.5)."""
        logger.debug("starting evaluateResidualEstimator")

        # define store function for timings
        from functools import partial
        def _store_stats(val, key, stats):
            stats[key] = val

        timing_stats = {}
        with timing(msg="ResidualEstimator.evaluateResidualEstimator", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-RESIDUAL", stats=timing_stats)):
            resind, reserror = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, pde, f, quadrature_degree)

        logger.debug("starting evaluateInactiveProjectionError")
        with timing(msg="ResidualEstimator.evaluateInactiveMIProjectionError", logfunc=logger.info, store_func=partial(_store_stats, key="TIME-INACTIVE-MI", stats=timing_stats)):
            mierror = ResidualEstimator.evaluateInactiveMIProjectionError(w, coeff_field, pde, maxh, newmi_add_maxm) 

        eta = sum(reserror[mu] ** 2 for mu in reserror)
        delta_inactive_mi = sum(v[1] ** 2 for v in mierror)
        est1 = ceta / sqrt(1 - gamma) * sqrt(eta)
        est2 = cQ / sqrt(1 - gamma) * sqrt(delta_inactive_mi)
        est3 = cQ * sqrt(zeta / (1 - gamma))
        est4 = zeta / (1 - gamma)
#        xi = (ceta / sqrt(1 - gamma) * sqrt(eta) + cQ / sqrt(1 - gamma) * sqrt(delta)
#              + cQ * sqrt(zeta / (1 - gamma))) ** 2 + zeta / (1 - gamma)
        xi = (est1 + est2 + est3) ** 2 + est4
        logger.info("Total Residual ERROR Factors: A1=%s  A2=%s  A3=%s  A4=%s", ceta / sqrt(1 - gamma), cQ / sqrt(1 - gamma), cQ * sqrt(zeta / (1 - gamma)), zeta / (1 - gamma))
        return (xi, resind, mierror, (est1, est2, est3, est4), (eta, delta, zeta), timing_stats)


    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField, anything, anything, optional(int))
    def evaluateResidualEstimator(cls, w, coeff_field, pde, f, quadrature_degree= -1):
        """Evaluate residual estimator EGSZ (5.7) for all active mu of w."""
        # evaluate residual estimator for all multi indices
        eta = MultiVector()
        global_error = {}
        for mu in w.active_indices():
            eta[mu], global_error[mu] = cls._evaluateResidualEstimator(mu, w, coeff_field, pde, f, quadrature_degree)
        return (eta, global_error)


    @classmethod
    @takes(anything, Multiindex, MultiVectorWithProjection, CoefficientField, anything, anything, int)
    def _evaluateResidualEstimator(cls, mu, w, coeff_field, pde, f, quadrature_degree):
        """Evaluate the residual error according to EGSZ (5.7) which consists of volume terms (5.3) and jump terms (5.5).

            .. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
        """
        # set quadrature degree
        quadrature_degree_old = parameters["form_compiler"]["quadrature_degree"]
        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
        logger.debug("residual quadrature order = " + str(quadrature_degree))
    
        # get pde residual terms
        r_T = pde.r_T
        r_E = pde.r_E
        r_Nb = pde.r_Nb
        
        # get mean field of coefficient
        a0_f = coeff_field.mean_func

        # prepare some FEM variables
        V = w[mu]._fefunc.function_space()
        mesh = V.mesh()
        nu = FacetNormal(mesh)

        # initialise volume and edge residual with deterministic part
#        R_T = dot(nabla_grad(a0_f), nabla_grad(w[mu]._fefunc))
        R_T = r_T(a0_f, w[mu]._fefunc)
        if not mu:
            R_T = R_T + f
#        R_E = a0_f * dot(nabla_grad(w[mu]._fefunc), nu)
        R_E = r_E(a0_f, w[mu]._fefunc, nu)
        # get Neumann residual
        homogeneousNBC = False if mu.order == 0 else True
        R_Nb = r_Nb(a0_f, w[mu]._fefunc, nu, mesh, homogeneous=homogeneousNBC)

        # iterate m
        Lambda = w.active_indices()
        maxm = w.max_order
        if len(coeff_field) < maxm:
            logger.warning("insufficient length of coefficient field for MultiVector (%i < %i)", len(coeff_field), maxm)
            maxm = len(coeff_field)
            #        assert coeff_field.length >= maxm        # ensure coeff_field expansion is sufficiently long
        for m in range(maxm):
            am_f, am_rv = coeff_field[m]

            # prepare polynom coefficients
            beta = am_rv.orth_polys.get_beta(mu[m])

            # mu
            res = -beta[0] * w[mu]

            # mu+1
            mu1 = mu.inc(m)
            if mu1 in Lambda:
                w_mu1 = w.get_projection(mu1, mu)
                res += beta[1] * w_mu1

            # mu-1
            mu2 = mu.dec(m)
            if mu2 in Lambda:
                w_mu2 = w.get_projection(mu2, mu)
                res += beta[-1] * w_mu2

            # add volume contribution for m
#            r_t = dot(nabla_grad(am_f), nabla_grad(res._fefunc))
            R_T = R_T + r_T(am_f, res._fefunc)
            # add edge contribution for m
#            r_e = am_f * dot(nabla_grad(res._fefunc), nu)
            R_E = R_E + r_E(am_f, res._fefunc, nu)

        # prepare more FEM variables for residual assembly
        DG = FunctionSpace(mesh, "DG", 0)
        s = TestFunction(DG)
        h = CellSize(mesh)

        # scaling of residual terms and definition of residual form
        a0_s = a0_f[0] if isinstance(a0_f, tuple) else a0_f     # required for elasticity parameters
        res_form = (h ** 2 * (1 / a0_s) * dot(R_T, R_T) * s * dx
                    + avg(h) * dot(avg(R_E) / avg(a0_s), avg(R_E)) * 2 * avg(s) * dS)
        # add Neumann residuals
        if R_Nb is not None:
            for rj, dsj in R_Nb:
                res_form = res_form + (1 / a0_s) * h * s * rj ** 2 * dsj

        # FEM evaluate residual on mesh
        eta = assemble(res_form)
        eta_indicator = np.array([sqrt(e) for e in eta])
        # map DG dofs to cell indices
        dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
        eta_indicator = eta_indicator[dofs]
        global_error = sqrt(sum(e for e in eta))

#        # debug ---        
#        print "==========RESIDUAL ESTIMATOR============"
#        print "eta", eta
#        print "eta_indicator", eta_indicator
#        print "global =", global_error
#        # ---debug
        
        # restore quadrature degree
        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_old

        return (FlatVector(eta_indicator), global_error)


    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, optional(float), optional(int))
    def evaluateUpperTailBounds(cls, w, coeff_field, pde, maxh=1 / 10, add_maxm=10):
        """Estimate upper tail bounds according to Section 3.2."""
        def prepare_ainfty(Lambda, M):
            ainfty = []
            a0_f = coeff_field.mean_func
            if isinstance(a0_f, tuple):
                a0_f = a0_f[0]
            # retrieve (sufficiently fine) function space for maximum norm evaluation
            # NOTE: we use the deterministic mesh since it is assumed to be the finest
            V = w[Multiindex()].basis.refine_maxh(maxh)
            # determine min \overline{a} on D (approximately)
            f = FEniCSVector.from_basis(V, sub_spaces=0)
            f.interpolate(a0_f)
            min_a0 = f.min_val
            for m in range(M):
                am_f, am_rv = coeff_field[m]
                if isinstance(am_f, tuple):
                    am_f = am_f[0]
                # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
                f.interpolate(am_f)
                max_am = f.max_val
                ainftym = max_am / min_a0
                assert isinstance(ainftym, float)
                ainfty.append(ainftym)
            return ainfty
        
        def prepare_norm_w(self, energynorm, w):
            normw = {}
            for mu in w.active_indices():
                normw[mu] = energynorm(w[mu]._fefunc)        
            return normw

        # evaluate (3.15)
        def eval_zeta_bar(mu, coef_field, ainfty, normw):
            pass
        
        # evaluate (3.11)
        def eval_zeta(mu, coeff_field, ainfty, normw):
            pass
        
        # prepare some variables
        energynorm = pde.norm
        Lambda = w.active_indices()
        M = min(w.max_order + add_maxm, len(coeff_field))
        ainfty = prepare_ainfty(Lambda, M)
        normw = prepare_norm_w(energynorm, w)
        
        # evaluate estimator contributions of (3.16)
        from collections import defaultdict
        zeta = defaultdict(0)
        zeta_bar = defaultdict(0)
        for mu in Lambda:
            # iterate Lambda for 
            for mu in Lambda:
                zeta_bar[mu] = eval_zeta_bar(mu, coeff_field, ainfty, normw)
                
            # iterate multiindex extensions
            for m in range(M):
                mu1 = mu.inc(m)
                if mu1 in Lambda:
                    continue
#                _, am_rv = coeff_field[m]
#                beta = am_rv.orth_polys.get_beta(mu1[m])
#                val1 = beta[1] * ainfty[m] * norm_w
                zeta[mu1] += eval_zeta(mu, coeff_field, ainfty, normw) 
                
#         # set largest projection error for mu
#         if mu1 not in Lambda_candidates.keys() or (mu1 in Lambda_canssssdidates.keys() and Lambda_candidates[mu1] < val1):
#             Lambda_candidates[mu1] = val1
            
#         logger.debug("POSSIBLE NEW MULTIINDICES %s", sorted(Lambda_candidates.iteritems(), key=itemgetter(1), reverse=True))
#         Lambda_candidates = sorted(Lambda_candidates.iteritems(), key=itemgetter(1), reverse=True)
#         return Lambda_candidates
