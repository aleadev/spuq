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

from dolfin import (assemble, inner, dot, nabla_grad, dx, avg, ds, dS, sqrt, refine, norm,
                    Function, FunctionSpace, TestFunction, CellSize, FacetNormal, Constant)

from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
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
    @takes(anything, MultiVector, CoefficientField, anything, float, float, float, float, optional(float))
    def evaluateError(cls, w, coeff_field, f, zeta, gamma, ceta, cQ, maxh=1 / 10):
        """Evaluate EGSZ Error (7.5)."""
        resind, reserror = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, f)
        projind, projerror = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh)
        eta = sum([reserror[mu] for mu in reserror.keys()])
        delta = sum([projerror[mu] for mu in projerror.keys()])
        xi = (ceta / sqrt(1 - gamma) * sqrt(eta) + cQ / sqrt(1 - gamma) * sqrt(delta) + cQ * sqrt(zeta / (1 - gamma))) ** 2 + zeta / (1 - gamma)
        return (xi, resind, projind)


    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField, anything)
    def evaluateResidualEstimator(cls, w, CF, f):
        """Evaluate residual estimator EGSZ (5.7) for all active mu of w."""
        # evaluate residual estimator for all multi indices
        eta = MultiVector()
        global_error = {}
        for mu in w.active_indices():
            eta[mu], global_error[mu] = cls._evaluateResidualEstimator(mu, w, CF, f)
        return (eta, global_error)


    @classmethod
    @takes(anything, Multiindex, MultiVectorWithProjection, CoefficientField, anything)
    def _evaluateResidualEstimator(cls, mu, w, CF, f):
        """Evaluate the residual error according to EGSZ (5.7) which consists of volume terms (5.3) and jump terms (5.5).

            .. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
        """
        # get mean field of coefficient
        a0_f, _ = CF[0]

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
        Delta = w.active_indices()
        maxm = max(len(mu) for mu in Delta) + 1
        if CF.length < maxm:
            logger.warning("insufficient length of coefficient field for MultiVector (%i < %i)", CF.length, maxm)
            maxm = CF.length  
#        assert CF.length >= maxm        # ensure CF expansion is sufficiently long
        for m in range(1, maxm):
            am_f, am_rv = CF[m]

            # prepare polynom coefficients
            beta = am_rv.orth_polys.get_beta(mu[m - 1])

            # mu
            res = -beta[0] * w[mu]

            # mu+1
            mu1 = mu.inc(m - 1)
            if mu1 in Delta:
                w_mu1 = w.get_projection(mu1, mu)
                res += beta[1] * w_mu1

            # mu-1
            mu2 = mu.dec(m - 1)
            if mu2 in Delta:
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
        R_T = 1 / a0_f * R_T ** 2
        R_E = 1 / a0_f * R_E ** 2
        res_form = (h ** 2 * R_T * s * dx
                    + avg(h) * avg(R_E) * 2 * avg(s) * dS)
#                    + h * R_E * s * ds)    NOTE: this term is incorrect for Dirichlet BC, Neumann data is not supported yet!

        # FEM evaluate residual on mesh
        eta = assemble(res_form)
        eta_indicator = np.array([sqrt(e) for e in eta])
        global_error = sqrt(sum(e ** 2 for e in eta))
        return (FlatVector(eta_indicator), global_error)


    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField, optional(float), optional(bool))
    def evaluateProjectionError(cls, w, CF, maxh=0.0, local=True):
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

        global_error = {}
        Delta = w.active_indices()
        if local:
            proj_error = MultiVector()
        else:
            proj_error = {}
        for mu in Delta:
            maxm = max(len(mu) for mu in Delta) + 1
            if CF.length < maxm:
                logger.warning("insufficient length of coefficient field for MultiVector (%i < %i)", CF.length, maxm)
                maxm = CF.length  
            dmu = sum(cls.evaluateLocalProjectionError(w, mu, m, CF, Delta, maxh, local)
                                                        for m in range(1, maxm))
            if local:
                proj_error[mu] = FlatVector(dmu)
                global_error[mu] = sum([e for e in dmu])
            else:
                proj_error[mu] = dmu
                global_error = dmu
        return proj_error, global_error


    @classmethod
    @takes(anything, MultiVectorWithProjection, Multiindex, int, CoefficientField, list_of(Multiindex), optional(float), optional(bool))
    def evaluateLocalProjectionError(cls, w, mu, m, CF, Delta, maxh=0.0, local=True):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        Both errors, :math:`\zeta_{\mu,T,m}^{\mu+e_m}` and :math:`\zeta_{\mu,T,m}^{\mu-e_m}` are returned.
        """

        # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
        a0_f, _ = CF[0]
        am_f, _ = CF[m]
        # create discretisation space
        V = w[mu]._fefunc.function_space()
        ufl = V.ufl_element()
        coeff_mesh = V.mesh()
        while maxh > 0 and coeff_mesh.hmax() > maxh:
            logger.debug("refining coefficient mesh for projection error evaluation")
            coeff_mesh = refine(coeff_mesh)
        # interpolate coefficient functions on mesh
        coeff_V = FunctionSpace(coeff_mesh, ufl.family(), ufl.degree())
        f = Function(coeff_V)
        f.interpolate(a0_f)
        amin = min(f.vector().array())
        f.interpolate(am_f)
        ammax = max(f.vector().array())
        ainfty = ammax / amin
        assert isinstance(ainfty, float)
        logger.debug("amin = %f  amax = %f  ainfty = %f", amin, ammax, ainfty)

        # prepare FEniCS discretisation variables
        projection_order_increase = 1
        mesh = V.mesh()
        V2 = FunctionSpace(mesh, ufl.family(), ufl.degree() + projection_order_increase)
        if local:
            DG = FunctionSpace(mesh, 'DG', 0)
            s = TestFunction(DG)
        else:
            s = Constant('1.0')

        # prepare polynom coefficients
        _, am_rv = CF[m]
        beta = am_rv.orth_polys.get_beta(mu[m - 1])

        # mu+1
        mu1 = mu.inc(m - 1)
        if mu1 in Delta:
            logger.debug("[LPE-A] local projection error for mu = %s with %s", mu, mu1)
            w_mu1 = w.get_projection(mu1, mu)
            w_mu1_V2 = Function(V2)
            w_mu1_V2.interpolate(w_mu1._fefunc)
            w_mu1_reference = Function(V2)                          # projection to higher order space as reference
            w_mu1_reference.interpolate(w[mu1]._fefunc)
            # evaluate H1 semi-norm of projection error
            error1 = Function(V2, w_mu1_V2.vector() - w_mu1_reference.vector())
            logger.debug("global error norms: L2 = %s and H1 = %s", norm(error1, "L2"), norm(error1, "H1"))
            a1 = a0_f * inner(nabla_grad(error1), nabla_grad(error1)) * s * dx
            pe = assemble(a1)
            if local:
                logger.debug("summed local errors: %s", sqrt(sum(pe)))
                zeta1 = beta[1] * np.array([sqrt(e) for e in pe])
            else:
                logger.debug("global error: %s", pe)
                zeta1 = beta[1] * sqrt(pe)
        else:
            if local:
                zeta1 = np.zeros(mesh.num_cells())
            else:
                zeta1 = 0

        # mu -1
        mu2 = mu.dec(m - 1)
        if mu2 in Delta:
            logger.debug("[LPE-B] local projection error for mu = %s with %s", mu, mu2)
            w_mu2 = w.get_projection(mu2, mu)
            w_mu2_V2 = Function(V2)
            w_mu2_V2.interpolate(w_mu2._fefunc)
            w_mu2_reference = Function(V2)                          # projection to higher order space as reference
            w_mu2_reference.interpolate(w[mu2]._fefunc)
            # evaluate H1 semi-norm of projection error
            error2 = Function(V2, w_mu2_V2.vector() - w_mu2_reference.vector())
            logger.debug("global error norms: L2 = %s and H1 = %s", norm(error2, "L2"), norm(error2, "H1"))
            a2 = a0_f * inner(nabla_grad(error2), nabla_grad(error2)) * s * dx
            pe = assemble(a2)
            if local:
                logger.debug("summed local errors: %s", sqrt(sum(pe)))
                zeta2 = beta[-1] * np.array([sqrt(e) for e in pe])
            else:
                logger.debug("global error: %s", pe)
                zeta2 = beta[-1] * pe
        else:
            if local:
                zeta2 = np.zeros(mesh.num_cells())
            else:
                zeta2 = 0

        zeta = ainfty * (zeta1 + zeta2)
        return zeta
