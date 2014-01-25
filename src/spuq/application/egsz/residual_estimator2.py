"""EGSZ2 a posteriori residual estimator (FEniCS centric implementation)

The residual estimator consists of a volume term :math:`\eta_{\mu,T}` and an edge term
:math:`\eta_{\mu,S}`. These terms are based on the flux of the discrete solution.

In an extended form for the more generic orthonormal polynomials in spuq, the three
terms are defined for some discrete :math:`w_N\in\mathcal{V}_N` by


.. math:: \eta_{\mu,T}(w_N) &:= h_T || \overline{a}^{-1/2} (f\delta_{\mu,0} + \nabla\overline{a}\cdot\nabla w_{N,\mu}
                                + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \alpha^m_{\mu_m+1}\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m}
                                - \alpha_{\mu_m}^m w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu_m-e_m} w_{N,\mu-e_m} ||_{L^2(T)}\\
          \eta_{\mu,S}(w_N) &:= h_S^{-1/2} || \overline{a}^{-1/2} [(\overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla
                                  ( \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m} w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                  + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m} w_{N,\mu-e_m})\cdot\nu] ||_{L^2(S)}\\
          \zeta_\mu(w_N) &:= ... TODO ...


The coefficients :math:`\alpha_j` follow from the recurrence coefficients
:math:`a_n,b_m,c_m` of the orthonormal polynomials by

.. math::
        \alpha_{n-1} &:= c_n/b_n \\
        \alpha_n &:= a_n/b_n \\
        \alpha_{n+1} &:= 1/b_n
"""

from __future__ import division
import numpy as np
from dolfin import (assemble, dot, nabla_grad, dx, avg, dS, sqrt, norm, VectorFunctionSpace, cells,
                    Constant, FunctionSpace, TestFunction, CellSize, FacetNormal, parameters, inner)

from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorSharedBasis, supp
#from spuq.fem.fenics.fenics_utils import weighted_H1_norm
from spuq.linalg.vector import FlatVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything, list_of, optional
from spuq.utils.decorators import cache

import logging
logger = logging.getLogger(__name__)


class ResidualEstimator(object):
    """Evaluation of the residual error estimator which consists of volume/edge terms and the upper tail bound.
    """

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, optional(int))
    def evaluateResidualEstimator(cls, w, coeff_field, pde, f, quadrature_degree= -1):
        """Evaluate residual estimator EGSZ2 (4.1) for all active mu of w."""
        # evaluate residual estimator for all multiindices
        eta_local = MultiVector()
        eta = {}
        for mu in w.active_indices():
            eta[mu], eta_local[mu] = cls._evaluateResidualEstimator(mu, w, coeff_field, pde, f, quadrature_degree)
        global_eta = sqrt(sum([v ** 2 for v in eta.values()]))
        return global_eta, eta, eta_local


    @classmethod
    @takes(anything, Multiindex, MultiVector, CoefficientField, anything, anything, int)
    def _evaluateResidualEstimator(cls, mu, w, coeff_field, pde, f, quadrature_degree):
        """Evaluate the residual error according to EGSZ2 (4.1) which consists of volume terms and jump terms.

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
        r_T = pde.volume_residual
        r_E = pde.edge_residual
        r_Nb = pde.neumann_residual
        
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
                w_mu1 = w[mu1]
                res += beta[1] * w_mu1

            # mu-1
            mu2 = mu.dec(m)
            if mu2 in Lambda:
                w_mu2 = w[mu2]
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
                res_form = res_form + (1 / a0_s) * h * s * inner(rj, rj) * dsj

        # FEM evaluate residual estimator on mesh
        eta = assemble(res_form)
        eta_indicator = np.array([sqrt(e) for e in eta])
        # map DG dofs to cell indices
        dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
        eta_indicator = eta_indicator[dofs]
        global_error = sqrt(sum(e for e in eta))

#        # debug ---        
#        print "==========RESIDUAL ESTIMATOR============"
#        print "eta", eta.array()
#        print "eta_indicator", eta_indicator
#        print "global =", global_error
#        # ---debug
        
        # restore quadrature degree
        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_old

        return global_error, FlatVector(eta_indicator)


    @classmethod
    @takes(anything, MultiVectorSharedBasis, CoefficientField, anything, optional(float), optional(int))
    def evaluateUpperTailBound(cls, w, coeff_field, pde, maxh=1 / 10, add_maxm=10):
        """Estimate upper tail bounds according to Section 3.2."""
        
        @cache
        def get_ainfty(m, V):
            a0_f = coeff_field.mean_func
            if isinstance(a0_f, tuple):
                a0_f = a0_f[0]
            # determine min \overline{a} on D (approximately)
            f = FEniCSVector.from_basis(V, sub_spaces=0)
            f.interpolate(a0_f)
            min_a0 = f.min_val
            am_f, _ = coeff_field[m]
            if isinstance(am_f, tuple):
                am_f = am_f[0]
            # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
            try:
                # use exact bounds if defined
                max_am = am_f.max_val
            except:
                # otherwise interpolate
                f.interpolate(am_f)
                max_am = f.max_val
            ainftym = max_am / min_a0
            assert isinstance(ainftym, float)
            return ainftym
        
        def prepare_norm_w(energynorm, w):
            normw = {}
            for mu in w.active_indices():
                normw[mu] = energynorm(w[mu]._fefunc)        
            return normw
        
        def LambdaBoundary(Lambda):
            suppLambda = supp(Lambda)
            for mu in Lambda:
                for m in suppLambda:
                    mu1 = mu.inc(m)
                    if mu1 not in Lambda:
                        yield mu1
                        
                    mu2 = mu.dec(m)
                    if mu2 not in Lambda and mu2 is not None:
                        yield mu2

        # evaluate (3.15)
        def eval_zeta_bar(mu, suppLambda, coeff_field, normw, V, M):
            assert mu in normw.keys()
            zz = 0
#            print "====zeta bar Z1", mu, M
            for m in range(M):
                if m in suppLambda:
                    continue
                _, am_rv = coeff_field[m]
                beta = am_rv.orth_polys.get_beta(mu[m])
                ainfty = get_ainfty(m, V)
                zz += (beta[1] * ainfty) ** 2
            return normw[mu] * sqrt(zz)
        
        # evaluate (3.11)
        def eval_zeta(mu, Lambda, coeff_field, normw, V, M=None, this_m=None):
            z = 0
            if this_m is None:
                for m in range(M):
                    _, am_rv = coeff_field[m]
                    beta = am_rv.orth_polys.get_beta(mu[m])
                    ainfty = get_ainfty(m, V)
                    mu1 = mu.inc(m)
                    if mu1 in Lambda:
#                        print "====zeta Z1", ainfty, beta[1], normw[mu1], " == ", ainfty * beta[1] * normw[mu1]
                        z += ainfty * beta[1] * normw[mu1]
                    mu2 = mu.dec(m)
                    if mu2 in Lambda:
#                        print "====zeta Z2", ainfty, beta[-1], normw[mu2], " == ", ainfty * beta[-1] * normw[mu2]
                        z += ainfty * beta[-1] * normw[mu2]
                return z
            else:
                m = this_m
                _, am_rv = coeff_field[m]
                beta = am_rv.orth_polys.get_beta(mu[m])
                ainfty = get_ainfty(m, V)
#                print "====zeta Z3", m, ainfty, beta[1], normw[mu], " == ", ainfty * beta[1] * normw[mu]
                return ainfty * beta[1] * normw[mu]
        
        # prepare some variables
        energynorm = pde.energy_norm
        Lambda = w.active_indices()
        suppLambda = supp(w.active_indices())
#        M = min(w.max_order + add_maxm, len(coeff_field))
        M = w.max_order + add_maxm
        normw = prepare_norm_w(energynorm, w)
        # retrieve (sufficiently fine) function space for maximum norm evaluation
        V = w[Multiindex()].basis.refine_maxh(maxh)[0]
        # evaluate estimator contributions of (3.16)
        from collections import defaultdict
        # === (a) zeta ===
        zeta = defaultdict(int)
        # iterate multiindex extensions
#        print "===A1 Lambda", Lambda
        for nu in LambdaBoundary(Lambda):
            assert nu not in Lambda
#            print "===A2 boundary nu", nu
            zeta[nu] += eval_zeta(nu, Lambda, coeff_field, normw, V, M)
        # === (b) zeta_bar ===
        zeta_bar = {}
        # iterate over active indices
        for mu in Lambda:
            zeta_bar[mu] = eval_zeta_bar(mu, suppLambda, coeff_field, normw, V, M)

        # evaluate summed estimator (3.16)
        global_zeta = sqrt(sum([v ** 2 for v in zeta.values()]) + sum([v ** 2 for v in zeta_bar.values()]))
        # also return zeta evaluation for single m (needed for refinement algorithm)
        eval_zeta_m = lambda mu, m: eval_zeta(mu=mu, Lambda=Lambda, coeff_field=coeff_field, normw=normw, V=V, M=M, this_m=m)
        logger.debug("=== ZETA  %s --- %s --- %s", global_zeta, zeta, zeta_bar)
        return global_zeta, zeta, zeta_bar, eval_zeta_m
