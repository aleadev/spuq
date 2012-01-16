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

from spuq.application.egsz.projection_cache import ProjectionCache
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.fem.fenics.fenics_function import FEniCSExpression, FEniCSFunction
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.multi_vector import MultiVector
from spuq.utils.type_check import *
from spuq.utils.multiindex_set import MultiindexSet

from dolfin import assemble, inner, dot, grad, dx, avg, ds, dS, sqrt
from dolfin import FunctionSpace, TestFunction, CellSize, FacetNormal


class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the projection error between different FE meshes.

    Note: In order to reduce computational costs, projected vectors are stored and reused at the expense of memory.
    fenics/dolfin implementation is based on
    https://answers.launchpad.net/dolfin/+question/177108
    """

    @takes(anything, CoefficientField, (FEniCSExpression, FEniCSFunction), MultiVector, optional(int))
    def __init__(self, CF, f, wN, maxm=10, ptype=FEniCSBasis.PROJECTION.INTERPOLATION):
        assert CF.length >= maxm
        self._CF = CF
        self._f = f
        self._maxm = maxm
        self._ptype = ptype
        self.wN = wN                # this also initialises the projection cache 

    @property
    def wN(self):
        return self._wN
    
    @wN.setter
    def wN(self, value):
        self._wN = value
        self._wN_cache = ProjectionCache(self._wN, self._ptype)

    @property
    def wN_cache(self):
        return self._wN_cache

    def evaluateResidualEstimator(self):
        """Evaluate residual estimator EGSZ (5.7) for all active mu of wN."""

        # evaluate residual estimator for all multi indices
        eta = MultiVector()
        for mu in self._wN.active_set():
            eta[mu] = self._evaluateResidualEstimator()
        return eta

    @takes(any, MultiindexSet)
    def _evaluateResidualEstimator(self, mu):
        """Evaluate the residual error according to EGSZ (5.7) which consists of volume terms (5.3) and jump terms (5.5).

            .. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
        """

        # get mean field of coefficient
        a0_f, _ = self._CF[0]

        # prepare FEM
        wN = self.wN
        wN_cache = self.wN_cache
        V = wN[mu].functionspace
        mesh = V.mesh
        DG = FunctionSpace(mesh, "DG", 0)
        w = TestFunction(DG)
        h = CellSize(mesh)
        nu = FacetNormal(mesh)
        # initialise volume and edge residual with deterministic part
        R_T = dot(grad(a0_f),grad(wN[mu].f))
        if mu.is_zero:
            R_T = R_T + self.f
        else:
            R_T = None  
        R_E = a0_f * grad(wN[mu].f)
        
        # iterate m
        Delta = wN.active_indices()
        for m in range(self.maxm):
            am_f, am_rv = self.CF[m]

            # prepare polynom coefficients
            # TODO: check order of coefficients
            am_p = am_rv.orth_poly
            (a, b, c) = am_p.recurrence_coefficients(mu[m])
            beta = (a/b, 1/b, c/b)

            # mu
            res = -beta[0]*wN[mu]

            # mu+1
            mu1 = mu.add( (m,1) )
            if mu1 in Delta:
                wNmu1 = wN_cache[mu1, mu, False]
                res += beta[1]*wNmu1

            # mu-1
            mu2 = mu.add( (m,-1) )
            if mu2 in Delta:
                wNmu2 = wN_cache[mu2, mu, False]
                res += beta[-1]*wNmu2


            # add volume contribution for m
            r_t = dot( grad(am_f), grad(res) ) 
            R_T = R_T + r_t 
            # add edge contribution for m
            r_e = a*dot( grad(res), grad(nu) )
            R_E = R_E + r_e

        # scaling of residual terms and definition of residual form
        R_T = 1/a * R_T**2
        R_E = 1/a * R_E**2
        res_form = (h**2*R_T*w*dx + avg(h)*avg(R_E)*2*avg(w)*dS + h*R_E*w*ds)  

        # FEM evaluate residual on mesh
        eta = assemble(res_form)
        error = sqrt(sum(i for i in eta.array()))
        return (eta, error)


    def evaluateProjectionError(self):
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
        for mu in self.wN.active_set():
            dmu = 0
            for m in range(1, self._maxm):
                dmu += self.evaluateLocalProjectionError(mu, m)
            delta[mu] = dmu
        return delta


    @takes(anything, MultiindexSet, int)
    def evaluateLocalProjectionError(self, mu, m):
        """Evaluate the local projection error according to EGSZ (6.4).

        Localisation of the global projection error (4.8) by (6.4)
        ..math::
            \zeta_{\mu,T,m}^{\mu\pm e_m} := ||a_m/\overline{a}||_{L^\infty(D)} \alpha_{\mu_m\pm 1}\int_T | \nabla( \Pi_{\mu\pm e_m}^\mu(\Pi_\mu^{\mu\pm e_m} w_{N,mu\pm e_)m})) - w_{N,mu\pm e_)m} |^2\;dx

        Both errors, :math:`\zeta_{\mu,T,m}^{\mu+e_m}` and :math:`\zeta_{\mu,T,m}^{\mu-e_m}` are returned.
        """

        @takes(any, MultiindexSet, int)
        def _evaluateLocalProjectionError(self, mu, m):
            """Evaluate and store projections of wN from source mu to destination mu. Checks and doesn't overwrite previously determined vectors."""
    
            # prepare polynom coefficients
            _, am_rv = self._CF[m]
            p = am_rv.orth_poly
            (a, b, c) = p.recurrence_coefficients(mu[m])
            beta = (a/b, 1/b, c/b)
    
            # mu+1
            mu1 = mu.add( (m,1) )
            _, wNmu1_back = self.wN_cache[mu1, mu, True]
            # evaluate H1 semi-norm of projection error
            error1 = wNmu1_back - self.wN[mu]
            a1 = inner(grad(error1), grad(error1))*dx
            zeta1 = beta[1]*assemble(a1)
    
            # mu -1
            mu2 = mu.add( (m,-1) )
            _, wNmu2_back = self.wN_cache[mu2, mu, True]
            # evaluate H1 semi-norm of projection error
            error2 = wNmu2_back - self.wN[mu]
            a2 = inner(grad(error2), grad(error2))*dx
            zeta2 = beta[-1]*assemble(a2)
            
            return (zeta1, zeta2)

        # determine ||a_m/\overline{a}||_{L\infty(D)}
        a0_f, _ = self._CF[0]
        am_f, _ = self._CF[m]
        mesh_points = self.wN[mu].mesh.coordinates()
        amin = min(a0_f(mesh_points))
        ammax = max(am_f(mesh_points))
        ainfty = ammax//amin

        # prepare projections of wN
        zeta1, zeta2 = self._evaluateLocalProjectionError(mu, m)
        zeta1 *= ainfty
        zeta2 *= ainfty
        return (zeta1, zeta2)
