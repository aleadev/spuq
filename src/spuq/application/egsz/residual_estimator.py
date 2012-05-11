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

from dolfin import (assemble, dot, nabla_grad, dx, avg, dS, sqrt, norm,
                    FunctionSpace, TestFunction, CellSize, FacetNormal)

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
    @takes(anything, MultiVector, CoefficientField, anything, float, float, float, float, optional(float), optional(int))
    def evaluateError(cls, w, coeff_field, f, zeta, gamma, ceta, cQ, maxh=0.1, projection_degree_increase=1):
        """Evaluate EGSZ Error (7.5)."""
        resind, reserror = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, f)
        projind, projerror = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, True, projection_degree_increase)
        eta = sum([reserror[mu] ** 2 for mu in reserror.keys()])
        delta = sum([projerror[mu] ** 2 for mu in projerror.keys()])
        xi = (ceta / sqrt(1 - gamma) * sqrt(eta) + cQ / sqrt(1 - gamma) * sqrt(delta)
              + cQ * sqrt(zeta / (1 - gamma))) ** 2 + zeta / (1 - gamma)
        return (xi, resind, projind)


    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField, anything)
    def evaluateResidualEstimator(cls, w, coeff_field, f):
        """Evaluate residual estimator EGSZ (5.7) for all active mu of w."""
        # evaluate residual estimator for all multi indices
        eta = MultiVector()
        global_error = {}
        for mu in w.active_indices():
            eta[mu], global_error[mu] = cls._evaluateResidualEstimator(mu, w, coeff_field, f)
        return (eta, global_error)


    @classmethod
    @takes(anything, Multiindex, MultiVectorWithProjection, CoefficientField, anything)
    def _evaluateResidualEstimator(cls, mu, w, coeff_field, f):
        """Evaluate the residual error according to EGSZ (5.7) which consists of volume terms (5.3) and jump terms (5.5).

            .. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
        """
        # get mean field of coefficient
        a0_f = coeff_field.mean_func

        # prepare some FEM variables
        V = w[mu]._fefunc.function_space()
        mesh = V.mesh()
        nu = FacetNormal(mesh)

        # initialise volume and edge residual with deterministic part
        R_T = dot(nabla_grad(a0_f), nabla_grad(w[mu]._fefunc))
        if not mu:
            R_T = R_T + f
        R_E = a0_f * dot(nabla_grad(w[mu]._fefunc), nu)

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
            r_t = dot(nabla_grad(am_f), nabla_grad(res._fefunc))
            R_T = R_T + r_t
            # add edge contribution for m
            r_e = am_f * dot(nabla_grad(res._fefunc), nu)
            R_E = R_E + r_e

        # prepare more FEM variables for residual assembly
        V = w[mu]._fefunc.function_space()
        DG = FunctionSpace(mesh, "DG", 0)
        s = TestFunction(DG)
        h = CellSize(mesh)

        # scaling of residual terms and definition of residual form
        R_T = (1 / a0_f) * R_T
        R_E = (1 / a0_f) * R_E
        res_form = (h ** 2 * R_T ** 2 * s * dx
                    + avg(h) * avg(R_E) ** 2 * 2 * avg(s) * dS)
        #                    + h * R_E * s * ds)    NOTE: this term is incorrect for Dirichlet BC, Neumann data is not supported yet!

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
        
        return (FlatVector(eta_indicator), global_error)


    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField, optional(float), optional(bool), optional(int))
    def evaluateProjectionError(cls, w, coeff_field, maxh=0.0, local=True, projection_degree_increase=1):
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
                zeta_mu = [cls.evaluateLocalProjectionError(w, mu, m, coeff_field, Lambda, maxh, local, projection_degree_increase)
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
    def evaluateLocalProjectionError(cls, w, mu, m, coeff_field, Lambda, maxh=0.0, local=True, projection_degree_increase=1):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        The sum :math:`\zeta_{\mu,T,m}^{\mu+e_m} + \zeta_{\mu,T,m}^{\mu-e_m}` is returned.
        """

        # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
        a0_f = coeff_field.mean_func
        am_f, _ = coeff_field[m]
        # create discretisation space
        V_coeff = w[mu].basis.refine_maxh(maxh)
        # interpolate coefficient functions on mesh
        f_coeff = V_coeff.new_vec()
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

            w_mu1_reference = w.get_projection(mu1, mu, 1 + projection_degree_increase)     # projection to higher order space as reference
            w_mu1 = w.get_projection(mu1, mu)
            w_mu1_V2 = w_mu1_reference.basis.project_onto(w_mu1)
            
            # evaluate H1 semi-norm of projection error
            error1 = w_mu1_V2 - w_mu1_reference
            logger.debug("global projection error norms: L2 = %s and H1 = %s", norm(error1._fefunc, "L2"), norm(error1._fefunc, "H1"))
            pe = weighted_H1_norm(a0_f, error1, local)
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

            w_mu2_reference = w.get_projection(mu2, mu, 1 + projection_degree_increase)     # projection to higher order space as reference
            w_mu2 = w.get_projection(mu2, mu)
            w_mu2_V2 = w_mu2_reference.basis.project_onto(w_mu2)
            
            # evaluate H1 semi-norm of projection error
            error2 = w_mu2_V2 - w_mu2_reference
            logger.debug("global projection error norms: L2 = %s and H1 = %s", norm(error2._fefunc, "L2"), norm(error2._fefunc, "H1"))
            pe = weighted_H1_norm(a0_f, error2, local)
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
