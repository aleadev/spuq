"""EM1 a posteriori global mixed equilibration estimator (FEniCS centric implementation)"""

from __future__ import division
import numpy as np
from dolfin import (assemble, dot, nabla_grad, dx, avg, dS, sqrt, norm, VectorFunctionSpace, cells,
                    Constant, FunctionSpace, TestFunction, CellSize, FacetNormal, parameters, inner,
                    TestFunctions, TrialFunctions, div, Function, solve, plot)

from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorSharedBasis, supp
from spuq.application.egsz.fem_discretisation import element_degree
from spuq.linalg.vector import FlatVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything, list_of, optional
from spuq.utils.decorators import cache

import logging
logger = logging.getLogger(__name__)


class EquilibrationEstimator(object):
    """Evaluation of the equilibration error estimator based on the solution of global mixed problems."""

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, optional(int))
    def evaluateEquilibrationEstimator(cls, w, coeff_field, pde, f, quadrature_degree= -1):
        """Evaluate equilibration estimator for all active mu of w."""
        eta_local = MultiVector()
        eta = {}
        for mu in w.active_indices():
            eta[mu], eta_local[mu] = cls._evaluateGlobalMixedEstimator(mu, w, coeff_field, pde, f, quadrature_degree)
        global_eta = sqrt(sum([v ** 2 for v in eta.values()]))
        return global_eta, eta, eta_local


    @classmethod
    @takes(anything, Multiindex, MultiVector, CoefficientField, anything, anything, int)
    def _evaluateGlobalMixedEstimator(cls, mu, w, coeff_field, pde, f, quadrature_degree):
        """Evaluation of global mixed equilibrated estimator."""
        # set quadrature degree
        quadrature_degree_old = parameters["form_compiler"]["quadrature_degree"]
        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
        logger.debug("residual quadrature order = " + str(quadrature_degree))

        # evaluate numerical flux
        # #######################

        # determine numerical flux sigma_nu with solution w
        Lambda = w.active_indices()
        maxm = w.max_order
        if len(coeff_field) < maxm:
            logger.warning("insufficient length of coefficient field for MultiVector (%i < %i)", len(coeff_field), maxm)
            maxm = len(coeff_field)

        # get mean field of coefficient and initialise sigma
        a0_f = coeff_field.mean_func
        sigma_mu = a0_f * nabla_grad(w[mu]._fefunc)

        # iterate m
        for m in range(maxm):
            am_f, am_rv = coeff_field[m]

            # prepare polynomial coefficients
            beta = am_rv.orth_polys.get_beta(mu[m])

            # mu
            r_mu = -beta[0] * w[mu]

            # mu+1
            mu1 = mu.inc(m)
            if mu1 in Lambda:
                w_mu1 = w[mu1]
                r_mu += beta[1] * w_mu1

            # mu-1
            mu2 = mu.dec(m)
            if mu2 in Lambda:
                w_mu2 = w[mu2]
                r_mu += beta[-1] * w_mu2

            # add flux contribution
            sigma_mu = sigma_mu + am_f * nabla_grad(r_mu._fefunc)

        # initialise f
        # TODO: do we want to project to f\vert_T and split off oscillations?
        if not mu:
            f_mu = f
        else:
            f_mu = Constant(0.0)


        # ###################
        # ## MIXED PROBLEM ##
        # ###################

        # get setup data for mixed problem
        V = w[mu]._fefunc.function_space()
        mesh = V.mesh()
        degree = element_degree(w[mu]._fefunc)

        # create function spaces
        DG = FunctionSpace(mesh, 'DG', 0)
        RT = FunctionSpace(mesh, 'RT', degree)  # TODO: degree-1 ?
        W = RT*DG

        # setup boundary conditions
        bcs = pde.create_dirichlet_bcs(W.sub(1))

        # TODO: treat Neumann boundary

        # debug ===
        # from dolfin import DOLFIN_EPS, DirichletBC
        # def boundary(x):
        #     return x[0] < DOLFIN_EPS or x[0] > 1.0 + DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 + DOLFIN_EPS
        # bcs = [DirichletBC(W.sub(1), Constant(0.0), boundary)]
        # === debug

        # create trial and test functions
        (sigma, u) = TrialFunctions(W)
        (tau, v) = TestFunctions(W)

        # define variational form
        a_eq = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
        L_eq = (-f_mu*v + dot(sigma_mu, tau))*dx

        # compute solution
        w_eq = Function(W)
        solve(a_eq == L_eq, w_eq, bcs)
        (sigma_mixed, u_mixed) = w_eq.split()


        # #############################
        # ## EQUILIBRATION ESTIMATOR ##
        # #############################

        # define error estimator
        z = TestFunction(DG)
        sigma_error = inner(sigma_mu-sigma_mixed, sigma_mu-sigma_mixed)
        eta_mu = sigma_error * z * dx

        # assemble error estimator
        eta_T = assemble(eta_mu)
        eta_T = np.array([sqrt(e) for e in eta_T])

        # eta_f = Function(DG)
        # eta_f.vector().array()[:] = eta_T
        # plot(eta_f, interactive=True)

        # evaluate global error
        eta = sqrt(sum(i**2 for i in eta_T))
        # reorder array entries for local estimators
        dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
        eta_T = eta_T[dofs]

        # restore quadrature degree
        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_old

        return eta, FlatVector(eta_T)


#     @classmethod
#     @takes(anything, MultiVectorSharedBasis, CoefficientField, anything, optional(float), optional(int))
#     def evaluateUpperTailBound(cls, w, coeff_field, pde, maxh=1 / 10, add_maxm=10):
#         """Estimate upper tail bounds according to Section 3.2."""
#
#         @cache
#         def get_ainfty(m, V):
#             a0_f = coeff_field.mean_func
#             if isinstance(a0_f, tuple):
#                 a0_f = a0_f[0]
#             # determine min \overline{a} on D (approximately)
#             f = FEniCSVector.from_basis(V, sub_spaces=0)
#             f.interpolate(a0_f)
#             min_a0 = f.min_val
#             am_f, _ = coeff_field[m]
#             if isinstance(am_f, tuple):
#                 am_f = am_f[0]
#             # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
#             try:
#                 # use exact bounds if defined
#                 max_am = am_f.max_val
#             except:
#                 # otherwise interpolate
#                 f.interpolate(am_f)
#                 max_am = f.max_val
#             ainftym = max_am / min_a0
#             assert isinstance(ainftym, float)
#             return ainftym
#
#         def prepare_norm_w(energynorm, w):
#             normw = {}
#             for mu in w.active_indices():
#                 normw[mu] = energynorm(w[mu]._fefunc)
#             return normw
#
#         def LambdaBoundary(Lambda):
#             suppLambda = supp(Lambda)
#             for mu in Lambda:
#                 for m in suppLambda:
#                     mu1 = mu.inc(m)
#                     if mu1 not in Lambda:
#                         yield mu1
#
#                     mu2 = mu.dec(m)
#                     if mu2 not in Lambda and mu2 is not None:
#                         yield mu2
#
#         # evaluate (3.15)
#         def eval_zeta_bar(mu, suppLambda, coeff_field, normw, V, M):
#             assert mu in normw.keys()
#             zz = 0
# #            print "====zeta bar Z1", mu, M
#             for m in range(M):
#                 if m in suppLambda:
#                     continue
#                 _, am_rv = coeff_field[m]
#                 beta = am_rv.orth_polys.get_beta(mu[m])
#                 ainfty = get_ainfty(m, V)
#                 zz += (beta[1] * ainfty) ** 2
#             return normw[mu] * sqrt(zz)
#
#         # evaluate (3.11)
#         def eval_zeta(mu, Lambda, coeff_field, normw, V, M=None, this_m=None):
#             z = 0
#             if this_m is None:
#                 for m in range(M):
#                     _, am_rv = coeff_field[m]
#                     beta = am_rv.orth_polys.get_beta(mu[m])
#                     ainfty = get_ainfty(m, V)
#                     mu1 = mu.inc(m)
#                     if mu1 in Lambda:
# #                        print "====zeta Z1", ainfty, beta[1], normw[mu1], " == ", ainfty * beta[1] * normw[mu1]
#                         z += ainfty * beta[1] * normw[mu1]
#                     mu2 = mu.dec(m)
#                     if mu2 in Lambda:
# #                        print "====zeta Z2", ainfty, beta[-1], normw[mu2], " == ", ainfty * beta[-1] * normw[mu2]
#                         z += ainfty * beta[-1] * normw[mu2]
#                 return z
#             else:
#                     m = this_m
#                     _, am_rv = coeff_field[m]
#                     beta = am_rv.orth_polys.get_beta(mu[m])
#                     ainfty = get_ainfty(m, V)
# #                    print "====zeta Z3", m, ainfty, beta[1], normw[mu], " == ", ainfty * beta[1] * normw[mu]
#                     return ainfty * beta[1] * normw[mu]
#
#         # prepare some variables
#         energynorm = pde.energy_norm
#         Lambda = w.active_indices()
#         suppLambda = supp(w.active_indices())
#         M = min(w.max_order + add_maxm, len(coeff_field))
#         normw = prepare_norm_w(energynorm, w)
#         # retrieve (sufficiently fine) function space for maximum norm evaluation
#         V = w[Multiindex()].basis.refine_maxh(maxh)[0]
#         # evaluate estimator contributions of (3.16)
#         from collections import defaultdict
#         # === (a) zeta ===
#         zeta = defaultdict(int)
#         # iterate multiindex extensions
# #        print "===A1 Lambda", Lambda
#         for nu in LambdaBoundary(Lambda):
#             assert nu not in Lambda
# #            print "===A2 boundary nu", nu
#             zeta[nu] += eval_zeta(nu, Lambda, coeff_field, normw, V, M)
#         # === (b) zeta_bar ===
#         zeta_bar = {}
#         # iterate over active indices
#         for mu in Lambda:
#             zeta_bar[mu] = eval_zeta_bar(mu, suppLambda, coeff_field, normw, V, M)
#
#         # evaluate summed estimator (3.16)
#         global_zeta = sqrt(sum([v ** 2 for v in zeta.values()]) + sum([v ** 2 for v in zeta_bar.values()]))
#         # also return zeta evaluation for single m (needed for refinement algorithm)
#         eval_zeta_m = lambda mu, m: eval_zeta(mu=mu, Lambda=Lambda, coeff_field=coeff_field, normw=normw, V=V, M=M, this_m=m)
#         logger.debug("=== ZETA  %s --- %s --- %s", global_zeta, zeta, zeta_bar)
#         return global_zeta, zeta, zeta_bar, eval_zeta_m
