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

from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection
from spuq.fem.fenics.fenics_function import FEniCSGenericFunction
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything

from dolfin import (assemble, inner, dot, grad, dx, avg, ds, dS, sqrt,
                    FunctionSpace, TestFunction, CellSize, FacetNormal)


class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.

    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
    fenics/dolfin implementation is based on
    https://answers.launchpad.net/dolfin/+question/177108
    """

    @takes(anything, CoefficientField, FEniCSGenericFunction)
    def __init__(self, CF, f):
        self.CF = CF
        self.f = f

    @takes(anything, MultiVectorWithProjection)
    def evaluateResidualEstimator(self, w):
        """Evaluate residual estimator EGSZ (5.7) for all active mu of wN."""

        # evaluate residual estimator for all multi indices
        eta = MultiVector()
        for mu in w.active_set():
            eta[mu] = self._evaluateResidualEstimator(mu, w)
        return eta

    @takes(anything, Multiindex, MultiVectorWithProjection)
    def _evaluateResidualEstimator(self, mu, w):
        """Evaluate the residual error according to EGSZ (5.7) which consists of volume terms (5.3) and jump terms (5.5).

            .. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
        """

        # TODO: implement vector space operations on fenics vectors (and test them)

        # get mean field of coefficient
        a0_f, _ = self.CF[0]
        # TODO: implement sqrt(fenics_function) in a reasonable way
        a = sqrt(a0_f)

        # prepare FEM
        V = w[mu].functionspace
        mesh = V.mesh
        DG = FunctionSpace(mesh, "DG", 0)
        w = TestFunction(DG)
        h = CellSize(mesh)
        nu = FacetNormal(mesh)
        # initialise volume and edge residual with deterministic part
        # TODO: a0_f.f, or a0_f.df, or ???
        R_T = dot(grad(a0_f), grad(w[mu].f))
        if not mu:
            R_T = R_T + self.f
        else:
            R_T = 0 * self.f # TODO: ConstantExpression("0")?
        R_E = a0_f * grad(w[mu].f)

        # iterate m
        Delta = w.active_indices()
        for m in range(1, len(self.CF)):
            am_f, am_rv = self.CF[m]

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
            r_t = dot(grad(am_f), grad(res.f))
            R_T = R_T + r_t
            # add edge contribution for m
            r_e = a * dot(grad(res.f), grad(nu))
            R_E = R_E + r_e

        # scaling of residual terms and definition of residual form
        R_T = 1 / a * R_T ** 2
        R_E = 1 / a * R_E ** 2
        res_form = (h ** 2 * R_T * w * dx
                    + avg(h) * avg(R_E) * 2 * avg(w) * dS
                    + h * R_E * w * ds)

        # FEM evaluate residual on mesh
        eta = assemble(res_form)
        error = sqrt(sum(i for i in eta.array()))
        return (eta, error)


    @takes(anything, MultiVectorWithProjection)
    def evaluateProjectionError(self, w):
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

        delta = MultiVector()
        for mu in w.active_set():
            delta[mu] = sum(self.evaluateLocalProjectionError(w, mu, m)
                            for m in range(1, len(self.CF)))

        return delta


    @takes(anything, MultiVectorWithProjection, Multiindex, int)
    def evaluateLocalProjectionError(self, w, mu, m):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        Both errors, :math:`\zeta_{\mu,T,m}^{\mu+e_m}` and :math:`\zeta_{\mu,T,m}^{\mu-e_m}` are returned.
        """

        # determine ||a_m/\overline{a}||_{L\infty(D)}
        a0_f, _ = self._CF[0]
        am_f, _ = self._CF[m]
        mesh_points = w[mu].mesh.coordinates()
        amin = min(a0_f(mesh_points))
        ammax = max(am_f(mesh_points))
        ainfty = ammax / amin

        # prepare projections of wN
        # prepare polynom coefficients
        _, am_rv = self._CF[m]
        beta = am_rv.orth_polys.get_beta(mu[m - 1])

        # mu+1
        mu1 = mu.inc(m - 1)
        w_mu1_back = w.get_back_projection(mu1, mu)
        # evaluate H1 semi-norm of projection error
        error1 = w_mu1_back - w[mu]
        a1 = inner(grad(error1.f), grad(error1.f)) * dx
        zeta1 = beta[1] * assemble(a1)

        # mu -1
        mu2 = mu.dec(m - 1)
        w_mu2_back = w.get_back_projection(mu2, mu)
        # evaluate H1 semi-norm of projection error
        error2 = w_mu2_back - w[mu]
        a2 = inner(grad(error2.f), grad(error2.f)) * dx
        zeta2 = beta[-1] * assemble(a2)

        zeta1 *= ainfty
        zeta2 *= ainfty
        return (zeta1, zeta2)
