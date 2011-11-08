"""EGSZ a posteriori residual estimator

The residual estimator consists of a volume term :math:`\eta_{\mu,T}`, an edge term
:math:`\eta_{\mu,S}` and the projection error :math:`\delta_\mu`. The former two terms
are based on the flux of the discrete solution while the latter term measures the
projection error between different FE meshes.

In an extended form for the more generic orthonormal polynomials in spuq, the three
terms are defined for some discrete :math:`w_N\in\mathcal{V}_N` by


.. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m} ||_{L^2(S)}\\
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


from spuq.fem.multi_vector import MultiVector
from spuq.utils.type_check import *
from spuq.application.egsz.coefficient_field import CoefficientField


class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.


    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
    """

    @takes(MultiVector, CoefficientField, optional(MultiVector))
    def evaluateResidualEstimator(self, wN, CF, projected_wN):
        """Evaluate the residual error according to EGSZ (5.7) which consists of volume terms (5.3) and jump terms (5.5).

        ..math::
            \eta_{\mu,T}(w_N) &:= ... \\
            \eta_{\mu,S}(w_N) &:= ...
        """


    @takes(MultiVector, CoefficientField)
    def evaluateFlux(self, wN, CF):
        """

        """
        newDelta = extend(Delta)

        for mu in newDelta:
            sigma_x = a[0]( w[mu].mesh.nodes ) * w[mu].dx() 
            for m in xrange(1,100):
                mu1 = mu.add( (m,1) )
                if mu1 in Delta:
                    sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]+1) *\
                        w[mu1].project( w[mu].mesh ).dx()
                    mu2 = mu.add( (m,-1) )
                if mu2 in Delta:
                    sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]) *\
                        w[mu2].project( w[mu].mesh ).dx()

    @takes()
    def evaluateGradFlux(self, wN, CF):
        """

        """


    @takes(MultiVector, CoefficientField, optional(MultiVector)
    def evaluateProjectionError(self, wN, CF, projected_wN):
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
        pass


    @takes(MultiVector, MultiindexSet, int, CoefficientField, optional(MultiVector))
    def evaluateLocalProjectionError(self, wN, mu, m, CF, projected_wN=None):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        Both errors, :math:`\zeta_{\mu,T,m}^{\mu+e_m}` and :math:`\zeta_{\mu,T,m}^{\mu-e_m}` are returned.
        """

        # determine ||a_m/\overline{a}||_{L\infty(D)}
        a = CF[0]
        am = CF[m]
        amin = min(a(wN[mu].mesh.coordinates()))
        ammax = max(am(wN[mu].mesh.coordinates()))
        ainfty = amax//amin

        # prepare polynom coefficients
        (a, b, c) = p.recurrence_coefficients(mu[m])
        beta = (a/b, 1/b, c/b)

        # prepare projections of wN
        zeta1, zeta2 = _evaluateProjections(wN, mu, m, projected_wN)
        zeta1 *= ainfty
        zeta2 *= ainfty

        return (zeta1, zeta2)


    @takes(MultiVector, MultiindexSet, int, MultiVector)
    _evaluateProjections(wN, mu, m, projected_wN, pt=FEniCSBasis.PROJECTION.INTERPOLATION)
        """Evaluate and store projection of wN from source mu to destination mu. Checks and doesn't overwrite previously determined vectors."""

        if not projected_wN:
            projected_wN = MultiVector()

        if mu not in projected_wN.keys():
            projected_wN[mu] = MultiVector()

        # prepare polynom coefficients
        (a, b, c) = p.recurrence_coefficients(mu[m])
        beta = (a/b, 1/b, c/b)

        # mu+1
        mu1 = mu.add( (m,1) )
        if mu1 not in projected_wN[mu].keys():
            projected_wN[mu][mu1] = wN[mu1].functionspace.project(wN[mu].functionspace.project(wN[mu1], pt), pt)
        # evaluate H1 semi-norm of projection error
        error1 = projected_wN[mu][mu1] - wN[mu]
        a1 = inner(grad(error1), grad(error1))*dx
        zeta1 = beta[1]*assemble(a1)

        # mu -1
        mu2 = mu.add( (m,-1) )
        if mu2 not in projected_wN[mu].keys():
            projected_wN[mu][mu2] = wN[mu2].functionspace.project(wN[mu].functionspace.project(wN[mu2], pt), pt)
            # evaluate H1 semi-norm of projection error
            error2 = projected_wN[mu][mu2] - wN[mu]
            a2 = inner(grad(error2), grad(error2))*dx
            zeta2 = beta[-1]*assemble(a2)

        return (zeta1, zeta2)
