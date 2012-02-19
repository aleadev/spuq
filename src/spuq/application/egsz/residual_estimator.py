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

from dolfin import (assemble, inner, dot, nabla_grad, dx, avg, ds, dS, sqrt,
                    Function, FunctionSpace, TestFunction, CellSize, FacetNormal)

from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.linalg.vector import FlatVector 
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything, list_of


class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.

    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
    fenics/dolfin implementation is based on
    https://answers.launchpad.net/dolfin/+question/177108
    """

    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField, anything)
    def evaluateResidualEstimator(cls, w, CF, f):
        """Evaluate residual estimator EGSZ (5.7) for all active mu of wN."""

        # evaluate residual estimator for all multi indices
        eta = MultiVector()
        for mu in w.active_indices():
            a, b = cls._evaluateResidualEstimator(mu, w, CF, f)
            print type(a), type(b)
            eta[mu], errormu = cls._evaluateResidualEstimator(mu, w, CF, f)
            print "local error for mu ", mu, ": ", errormu
#            print eta[mu].array()
        return eta


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
        for m in range(1, len(CF)):
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
        R_T = 1 / sqrt(a0_f) * R_T ** 2
        R_E = 1 / sqrt(a0_f) * R_E ** 2
        res_form = (h ** 2 * R_T * s * dx
                    + avg(h) * avg(R_E) * 2 * avg(s) * dS
                    + h * R_E * s * ds)

        # FEM evaluate residual on mesh
        eta = assemble(res_form)
        eta_indicator = np.array([sqrt(e) for e in eta])
        error = sqrt(sum(i for i in eta.array()))
        return (FlatVector(eta_indicator), error)


    @classmethod
    @takes(anything, MultiVectorWithProjection, CoefficientField)
    def evaluateProjectionError(cls, w, CF):
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

        Delta = w.active_indices()
        print "\nDELTA: ", type(Delta)
        proj_error = MultiVector()
        for mu in Delta:
            dmu = sum(cls.evaluateLocalProjectionError(w, mu, m, CF, Delta)
                                                for m in range(1, len(CF)))
            proj_error[mu] = FlatVector(dmu)
        return proj_error


    @classmethod
    @takes(anything, MultiVectorWithProjection, Multiindex, int, CoefficientField, list_of(Multiindex))
    def evaluateLocalProjectionError(cls, w, mu, m, CF, Delta):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        Both errors, :math:`\zeta_{\mu,T,m}^{\mu+e_m}` and :math:`\zeta_{\mu,T,m}^{\mu-e_m}` are returned.
        """

        # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
        a0_f, _ = CF[0]
        am_f, _ = CF[m]
        f = Function(w[mu]._fefunc.function_space())
        f.interpolate(a0_f)
        amin = min(f.vector().array())
        f.interpolate(am_f)
        ammax = max(f.vector().array())
        ainfty = ammax / amin
        assert isinstance(ainfty, float)
#        print "\namin, amax, ainfty ", amin, ammax, ainfty

        # prepare FEniCS discretisation variables
        mesh = w[mu]._fefunc.function_space().mesh()
        DG = FunctionSpace(mesh, 'DG', 0)
        s = TestFunction(DG)

        # prepare polynom coefficients
        _, am_rv = CF[m]
        beta = am_rv.orth_polys.get_beta(mu[m - 1])

        # mu+1
        mu1 = mu.inc(m - 1)
        if mu1 in Delta:
            w_mu1_back = w.get_back_projection(mu1, mu)
            # evaluate H1 semi-norm of projection error
            error1 = w_mu1_back - w[mu]
            a1 = inner(nabla_grad(error1._fefunc), nabla_grad(error1._fefunc)) * s * dx
            zeta1 = beta[1] * assemble(a1)
            zeta1 = zeta1.array()
        else:
            zeta1 = np.zeros(mesh.num_cells())

        # mu -1
        mu2 = mu.dec(m - 1)
        if mu2 in Delta:
            w_mu2_back = w.get_back_projection(mu2, mu)
            # evaluate H1 semi-norm of projection error
            error2 = w_mu2_back - w[mu]
            a2 = inner(nabla_grad(error2._fefunc), nabla_grad(error2._fefunc)) * s * dx
            
            # TODO: beta[-1] sometimes returns numpy.float64 instead of float leading to wrong results in the following multiplication
            zeta2 = float(beta[-1]) * assemble(a2)
            print "\n-----ZETA2:", type(zeta2)
            print type(beta[-1])

#            zeta2 = beta[-1] * assemble(a2)
            zeta2 = zeta2.array()
        else:
            zeta2 = np.zeros(mesh.num_cells())

        zeta = ainfty * (zeta1 + zeta2)
        return zeta
