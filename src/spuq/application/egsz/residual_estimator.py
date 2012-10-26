"""EGSZ a posteriori residual estimator (FEniCS centric implementation)

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

from dolfin import (assemble, dot, nabla_grad, dx, avg, dS, sqrt, norm, VectorFunctionSpace,
                    Constant, FunctionSpace, TestFunction, CellSize, FacetNormal, parameters)

from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.fem.fenics.fenics_utils import weighted_H1_norm
from spuq.linalg.vector import FlatVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything, list_of, optional

import logging

logger = logging.getLogger(__name__)

class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.

    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
    fenics/dolfin implementation is based on
    https://answers.launchpad.net/dolfin/+question/177108
    """

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, float, float, float, float, optional(float), optional(int), optional(int), optional(int))
    def evaluateError(cls, w, coeff_field, pde, f, zeta, gamma, ceta, cQ, maxh=0.1, quadrature_degree= -1, projection_degree_increase=1, refine_projection_mesh=1):
        """Evaluate EGSZ Error (7.5)."""
        logger.debug("starting evaluateResidualEstimator")
        resind, reserror = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, pde, f, quadrature_degree)
        logger.debug("starting evaluateProjectionEstimator")
        projind, projerror = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, True, projection_degree_increase, refine_projection_mesh)
        eta = sum([reserror[mu] ** 2 for mu in reserror.keys()])
        delta = sum([projerror[mu] ** 2 for mu in projerror.keys()])
        xi = (ceta / sqrt(1 - gamma) * sqrt(eta) + cQ / sqrt(1 - gamma) * sqrt(delta)
              + cQ * sqrt(zeta / (1 - gamma))) ** 2 + zeta / (1 - gamma)
        logger.info("Total Residual ERROR Factors: A1=%s  A2=%s  A3=%s  A4=%s", ceta / sqrt(1 - gamma), cQ / sqrt(1 - gamma), cQ * sqrt(zeta / (1 - gamma)), zeta / (1 - gamma))
        return (xi, resind, projind)


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
        R_Nb = r_Nb(a0_f, w[mu]._fefunc, nu, mesh)

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
        R_T = (1 / a0_s) * R_T
        R_E = (1 / a0_s) * R_E
        res_form = (h ** 2 * dot(R_T, R_T) * s * dx
                    + avg(h) * dot(avg(R_E), avg(R_E)) * 2 * avg(s) * dS)
        # add Neumann residuals
        if R_Nb is not None:
            for rj, dsj in R_Nb:
                res_form = res_form + (1 / a0_f) * h * s * rj * dsj

        # FEM evaluate residual on mesh
        eta = assemble(res_form)
        eta_indicator = np.array([sqrt(e) for e in eta])
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
    @takes(anything, MultiVectorWithProjection, CoefficientField, optional(float), optional(bool), optional(int))
    def evaluateProjectionError(cls, w, coeff_field, maxh=0.0, local=True, projection_degree_increase=1, refine_mesh=1):
        """Evaluate the projection error according to EGSZ (4.8).

        The global projection error
        ..math::
            \delta_\mu(w_N) := \sum_{m=1}^\infty ||a_m/\overline{a}||_{L^\infty(D)}
            \left\{ \int_D \overline{a}\alpha_{\mu_m+1}^\mu |\nabla(\Pi_{\mu+e_m}^\mu(\Pi_\mu^{\mu+e_m}w_{N,\mu+e_m}))|^2\;dx \right^{1/2}
            + \left\{ \int_D \overline{a}\alpha_{\mu_m-1}^\mu |\nabla(\Pi_{\mu-e_m}^\mu(\Pi_\mu^{\mu-e_m}w_{N,\mu-e_m}))|^2\;dx \right^{1/2}

        is localised by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx
        """
        
        if logger.isEnabledFor(logging.DEBUG):
            for mu in w.active_indices():
                logger.debug("[projection error] mesh for %s has %s cells", mu, w[mu]._fefunc.function_space().mesh().num_cells())
    
        global_error = {}
        if local:
            proj_error = MultiVector()
        else:
            proj_error = {}
        Lambda = w.active_indices()
        if len(Lambda) > 1:
            for mu in Lambda:
                maxm = w.max_order
                if len(coeff_field) < maxm:
                    logger.warning("insufficient length of coefficient field for MultiVector (%i < %i)",
                        len(coeff_field), maxm)
                    maxm = len(coeff_field)
                zeta_mu = [cls.evaluateLocalProjectionError(w, mu, m, coeff_field, Lambda, maxh, local, projection_degree_increase, refine_mesh)
                                for m in range(maxm)]
                dmu = sum(zeta_mu)
                if local:
                    proj_error[mu] = FlatVector(dmu)
#                    global_error[mu] = sqrt(sum([e ** 2 for e in dmu]))
                    global_error[mu] = sum([sqrt(sum([e ** 2 for e in perr])) for perr in zeta_mu])
                else:
                    proj_error[mu] = dmu
                    global_error = dmu
        else:
            mu = Lambda[0]
            if local:
                proj_error[mu] = FlatVector(np.zeros(w[mu].coeffs.size()))
            else:
                proj_error[mu] = 0
            global_error = {mu: 0}
        return proj_error, global_error


    @classmethod
    @takes(anything, MultiVectorWithProjection, Multiindex, int, CoefficientField, list_of(Multiindex), optional(float),
        optional(bool), optional(int))
    def evaluateLocalProjectionError(cls, w, mu, m, coeff_field, Lambda, maxh=0.0, local=True, projection_degree_increase=1, refine_mesh=1):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        The sum :math:`\zeta_{\mu,T,m}^{\mu+e_m} + \zeta_{\mu,T,m}^{\mu-e_m}` is returned.
        """

        # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
        a0_f = coeff_field.mean_func
        am_f, _ = coeff_field[m]
        if isinstance(a0_f, tuple):
            assert isinstance(am_f, tuple)
            a0_f = a0_f[0]
            am_f = am_f[0]
        # create discretisation space
        V_coeff = w[mu].basis.refine_maxh(maxh)
        # interpolate coefficient functions on mesh
        f_coeff = V_coeff.new_vector(sub_spaces=0)
#        print "evaluateLocalProjectionError"
#        print f_coeff.num_sub_spaces
#        print a0_f.value_shape()
        f_coeff.interpolate(a0_f)
        amin = f_coeff.min_val
        f_coeff.interpolate(am_f)
        ammax = f_coeff.max_val
        ainfty = ammax / amin
        assert isinstance(ainfty, float)
        logger.debug("==== local projection error for mu = %s ====", mu)
        logger.debug("amin = %f  amax = %f  ainfty = %f", amin, ammax, ainfty)

        # prepare polynom coefficients
        _, am_rv = coeff_field[m]
        beta = am_rv.orth_polys.get_beta(mu[m])

        # mu+1
        mu1 = mu.inc(m)
        if mu1 in Lambda:
            logger.debug("[LPE-A] local projection error for mu = %s with %s", mu, mu1)

            # debug---
#            if True:
#                from dolfin import Function, inner
#                V1 = w[mu]._fefunc.function_space();
#                ufl = V1.ufl_element();
#                V2 = FunctionSpace(V1.mesh(), ufl.family(), ufl.degree() + 1)
#                f1 = Function(V1)
#                f1.interpolate(w[mu1]._fefunc)
#                f12 = Function(V2)
#                f12.interpolate(f1)
#                f2 = Function(V2)
#                f2.interpolate(w[mu1]._fefunc)
#                err2 = Function(V2, f2.vector() - f12.vector())
#                aerr = a0_f * inner(nabla_grad(err2), nabla_grad(err2)) * dx
#                perr = sqrt(assemble(aerr))
#                logger.info("DEBUG A --- global projection error %s - %s: %s", mu1, mu, perr)
            # ---debug

            # evaluate H1 semi-norm of projection error
            error1, sum_up = w.get_projection_error_function(mu1, mu, 1 + projection_degree_increase, refine_mesh=refine_mesh)
            logger.debug("global projection error norms: L2 = %s and H1 = %s", norm(error1._fefunc, "L2"), norm(error1._fefunc, "H1"))
            pe = weighted_H1_norm(a0_f, error1, local)
            pe = sum_up(pe)     # summation for cells according to reference mesh refinement
            if local:
                logger.debug("summed local projection errors: %s", sqrt(sum([e ** 2 for e in pe])))
            else:
                logger.debug("global projection error: %s", pe)
            zeta1 = beta[1] * pe
        else:
            if local:
                zeta1 = np.zeros(w[mu].basis.mesh.num_cells())
            else:
                zeta1 = 0

        # mu -1
        mu2 = mu.dec(m)
        if mu2 in Lambda:
            logger.debug("[LPE-B] local projection error for mu = %s with %s", mu, mu2)

            # debug---
#            if True:
#                from dolfin import Function, inner
#                V1 = w[mu]._fefunc.function_space();
#                ufl = V1.ufl_element();
#                V2 = FunctionSpace(V1.mesh(), ufl.family(), ufl.degree() + 1)
#                f1 = Function(V1)
#                f1.interpolate(w[mu2]._fefunc)
#                f12 = Function(V2)
#                f12.interpolate(f1)
#                f2 = Function(V2)
#                f2.interpolate(w[mu2]._fefunc)
#                err2 = Function(V2, f2.vector() - f12.vector())
#                aerr = a0_f * inner(nabla_grad(err2), nabla_grad(err2)) * dx
#                perr = sqrt(assemble(aerr))
#                logger.info("DEBUG B --- global projection error %s - %s: %s", mu2, mu, perr)
            # ---debug
            
            # evaluate H1 semi-norm of projection error
            error2, sum_up = w.get_projection_error_function(mu2, mu, 1 + projection_degree_increase, refine_mesh=refine_mesh)
            logger.debug("global projection error norms: L2 = %s and H1 = %s", norm(error2._fefunc, "L2"), norm(error2._fefunc, "H1"))
            pe = weighted_H1_norm(a0_f, error2, local)
            pe = sum_up(pe)     # summation for cells according to reference mesh refinement
            if local:
                logger.debug("summed local projection errors: %s", sqrt(sum([e ** 2 for e in pe])))
            else:
                logger.debug("global projection error: %s", pe)
            zeta2 = beta[-1] * pe
        else:
            if local:
                zeta2 = np.zeros(w[mu].basis.mesh.num_cells())
            else:
                zeta2 = 0

        logger.debug("beta[-1] = %s  beta[1] = %s  ainfty = %s", beta[-1], beta[1], ainfty)
        zeta = ainfty * (zeta1 + zeta2)
        return zeta


    @classmethod
    @takes(anything, MultiVector, CoefficientField, optional(float), optional(int), optional(bool))
    def evaluateInactiveMIProjectionError(cls, w, coeff_field, maxh=1 / 10, add_maxm=10, accumulate=True):
        """Estimate projection error for inactive indices."""
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
        
        # determine possible new indices
        Lambda_candidates = {}
        Lambda = w.active_indices()
        M = min(w.max_order + add_maxm, len(coeff_field))
        ainfty = prepare_ainfty(Lambda, M)
        for mu in Lambda:
            # evaluate energy norm of w[mu]
            a0_f = coeff_field.mean_func
            if isinstance(a0_f, tuple):
                a0_f = a0_f[0]
            norm_w = weighted_H1_norm(a0_f, w[mu])
            logger.debug("NEW MI with mu = %s    norm(w) = %s", mu, norm_w)
            # iterate multiindex extensions
            for m in range(M):
                mu1 = mu.inc(m)
                if mu1 in Lambda:
                    continue
                _, am_rv = coeff_field[m]
                beta = am_rv.orth_polys.get_beta(mu1[m])

                #                    logger.debug("A*** %f -- %f -- %f", beta[1], ainfty[m], norm_w)
                #                    logger.debug("B*** %f", beta[1] * ainfty[m] * norm_w)
                #                    logger.debug("C*** %f -- %f", theta_delta, max_zeta)
                #                    logger.debug("D*** %f", theta_delta * max_zeta)
                #                    logger.debug("E*** %s", bool(beta[1] * ainfty[m] * norm_w >= theta_delta * max_zeta))
                logger.debug("\t beta[1] * ainfty * norm_w = %s", beta[1] * ainfty[m] * norm_w)
                
                val1 = beta[1] * ainfty[m] * norm_w
                if not accumulate:
                    # set largest projection error for mu
                    if mu1 not in Lambda_candidates.keys() or (mu1 in Lambda_candidates.keys() and Lambda_candidates[mu1] < val1):
                        Lambda_candidates[mu1] = val1
                else:
                    # accumulate projection errors for mu
                    if mu1 not in Lambda_candidates.keys():
                        Lambda_candidates[mu1] = val1
                    else:
                        Lambda_candidates[mu1] += val1

        logger.info("POSSIBLE NEW MULTIINDICES %s", sorted(Lambda_candidates.iteritems(), key=itemgetter(1), reverse=True))
        Lambda_candidates = sorted(Lambda_candidates.iteritems(), key=itemgetter(1), reverse=True)
        return Lambda_candidates
