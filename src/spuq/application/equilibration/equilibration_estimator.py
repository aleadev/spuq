"""EM1 a posteriori global mixed equilibration estimator (FEniCS centric implementation)"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import itertools as iter
from operator import itemgetter
from dolfin import (assemble, dot, nabla_grad, dx, avg, dS, sqrt, norm, VectorFunctionSpace, cells,
                    Constant, FunctionSpace, TestFunction, CellSize, FacetNormal, parameters, inner,
                    TestFunctions, TrialFunctions, TrialFunction, div, Function, CellFunction, Measure,
                    vertices, Vertex, FacetFunction, Cell, facets, jump, avg, project, solve, plot)

from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorSharedBasis, supp
from spuq.application.egsz.fem_discretisation import element_degree
from spuq.linalg.vector import FlatVector
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.type_check import takes, anything, list_of, optional

import logging
logger = logging.getLogger(__name__)


def evaluate_oscillations(f, mesh, degree, dg0, osc_quad_degree = 15):
    # project f and evaluate oscillations
    dx = Measure('dx')
    PV = FunctionSpace(mesh, 'CG', degree) if degree > 0 else FunctionSpace(mesh, 'DG', 0)
    Pf = project(f, PV)
    h = CellSize(mesh)
    osc_form = h**2 * ((f-Pf)**2) * dg0 * dx
    osc_local = assemble(osc_form, form_compiler_parameters={'quadrature_degree': osc_quad_degree})
    osc_global = np.sqrt(np.sum(osc_local.array()))
    osc_local = [np.sqrt(o) for o in osc_local.array()]
    return osc_global, osc_local, Pf


def evaluate_numerical_flux(w, mu, coeff_field, f):
    '''determine numerical flux sigma_nu with solution w'''
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
    if not mu:
        f_mu = f
    else:
        f_mu = Constant(0.0)

    return sigma_mu, f_mu


class GlobalEquilibrationEstimator(object):
    """Evaluation of the equilibration error estimator based on the solution of global mixed problems."""

    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, optional(int))
    def evaluateEstimator(cls, w, coeff_field, pde, f, quadrature_degree= -1, osc_quadrature_degree = 15):
        """Evaluate equilibration estimator for all active mu of w."""

        # TODO: determine oscillations of coeff_field and calculate with projected coefficients?!

        # determine rhs oscillations
        mu0 = Multiindex()
        mesh = w[mu0]._fefunc.function_space().mesh()
        degree = element_degree(w[mu0]._fefunc)
        DG0 = FunctionSpace(mesh, 'DG', 0)
#        DG0_dofs = dict([(c.index(),DG0.dofmap().cell_dofs(c.index())[0]) for c in cells(mesh)])
        dg0 = TestFunction(DG0)
        osc_global, osc_local, Pf = evaluate_oscillations(f, mesh, degree - 1, dg0, osc_quadrature_degree)

        # evaluate global equilibration estimators
        eta_local = MultiVector()
        eta = {}
        for mu in w.active_indices():
            eta[mu], eta_local[mu] = cls._evaluateGlobalMixedEstimator(mu, w, coeff_field, pde, Pf, quadrature_degree)
        global_eta = sqrt(sum([v ** 2 for v in eta.values()]))
        return global_eta, eta, eta_local, osc_global, osc_local


    @classmethod
    @takes(anything, Multiindex, MultiVector, CoefficientField, anything, anything, int)
    def _evaluateGlobalMixedEstimator(cls, mu, w, coeff_field, pde, f, quadrature_degree, vectorspace_type='BDM'):
        """Evaluation of global mixed equilibrated estimator."""
        # set quadrature degree
#        quadrature_degree_old = parameters["form_compiler"]["quadrature_degree"]
#        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
#        logger.debug("residual quadrature order = " + str(quadrature_degree))

        # prepare numerical flux and f
        sigma_mu, f_mu = evaluate_numerical_flux(w, mu, coeff_field, f)

        # ###################
        # ## MIXED PROBLEM ##
        # ###################

        # get setup data for mixed problem
        V = w[mu]._fefunc.function_space()
        mesh = V.mesh()
        degree = element_degree(w[mu]._fefunc)

        # create function spaces
        DG0 = FunctionSpace(mesh, 'DG', 0)
        DG0_dofs = [DG0.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
        RT = FunctionSpace(mesh, vectorspace_type, degree)
        W = RT * DG0

        # setup boundary conditions
#        bcs = pde.create_dirichlet_bcs(W.sub(1))

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
        a_eq = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
        L_eq = (- f_mu * v + dot(sigma_mu, tau)) * dx

        # compute solution
        w_eq = Function(W)
        solve(a_eq == L_eq, w_eq)
        (sigma_mixed, u_mixed) = w_eq.split()

        # #############################
        # ## EQUILIBRATION ESTIMATOR ##
        # #############################

        # evaluate error estimator
        dg0 = TestFunction(DG0)
        eta_mu = inner(sigma_mu, sigma_mu) * dg0 * dx
        eta_T = assemble(eta_mu, form_compiler_parameters={'quadrature_degree': quadrature_degree})
        eta_T = np.array([sqrt(e) for e in eta_T])

        # evaluate global error
        eta = sqrt(sum(i**2 for i in eta_T))
        # reorder array entries for local estimators
        eta_T = eta_T[DG0_dofs]

        # restore quadrature degree
#        parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_old

        return eta, FlatVector(eta_T)


class LocalEquilibrationEstimator(object):
    @classmethod
    @takes(anything, MultiVector, CoefficientField, anything, anything, optional(int))
    def evaluateEstimator(cls, w, coeff_field, pde, f, quadrature_degree= -1, osc_quadrature_degree = 15):
        """Evaluate patch local equilibration estimator for all active mu of w."""

        # TODO: determine oscillations of coeff_field and calculate with projected coefficients?!

        # use uBLAS backend for conversion to scipy sparse matrices
        backup_backend = parameters.linear_algebra_backend
        parameters.linear_algebra_backend = "uBLAS"

        # determine rhs oscillations
        mu0 = Multiindex()
        mesh = w[mu0]._fefunc.function_space().mesh()
        degree = element_degree(w[mu0]._fefunc)
        DG0 = FunctionSpace(mesh, 'DG', 0)
#        DG0_dofs = dict([(c.index(),DG0.dofmap().cell_dofs(c.index())[0]) for c in cells(mesh)])
        dg0 = TestFunction(DG0)
        osc_global, osc_local, Pf = evaluate_oscillations(f, mesh, degree - 1, dg0, osc_quadrature_degree)

        # evaluate global equilibration estimators
        eta_local = MultiVector()
        eta = {}
        for mu in w.active_indices():
            eta[mu], eta_local[mu] = cls._evaluateLocalEstimator(mu, w, coeff_field, pde, Pf, quadrature_degree)
        global_eta = sqrt(sum([v ** 2 for v in eta.values()]))

        # restore backend and return estimator
        parameters.linear_algebra_backend = backup_backend
        return global_eta, eta, eta_local, osc_global, osc_local


    @classmethod
    @takes(anything, Multiindex, MultiVector, CoefficientField, anything, anything, int)
    def _evaluateLocalEstimator(cls, mu, w, coeff_field, pde, f, quadrature_degree, epsilon=1e-5):
        """Evaluation of patch local equilibrated estimator."""

        # prepare numerical flux and f
        sigma_mu, f_mu = evaluate_numerical_flux(w, mu, coeff_field, f)

        # ###################
        # ## MIXED PROBLEM ##
        # ###################

        # get setup data for mixed problem
        V = w[mu]._fefunc.function_space()
        mesh = V.mesh()
        mesh.init()
        degree = element_degree(w[mu]._fefunc)

        # data for nodal bases
        V_dm = V.dofmap()
        V_dofs = dict([(i, V_dm.cell_dofs(i)) for i in range(mesh.num_cells())])
        V1 = FunctionSpace(mesh, 'CG', 1)   # V1 is to define nodal base functions
        phi_z = Function(V1)
        phi_coeffs = np.ndarray(V1.dim())
        vertex_dof_map = V1.dofmap().vertex_to_dof_map(mesh)
        # vertex_dof_map = vertex_to_dof_map(V1)
        dof_list = vertex_dof_map.tolist()
        # DG0 localisation
        DG0 = FunctionSpace(mesh, 'DG', 0)
        DG0_dofs = dict([(c.index(),DG0.dofmap().cell_dofs(c.index())[0]) for c in cells(mesh)])
        dg0 = TestFunction(DG0)
        # characteristic function of patch
        xi_z = Function(DG0)
        xi_coeffs = np.ndarray(DG0.dim())
        # mesh data
        h = CellSize(mesh)
        n = FacetNormal(mesh)
        cf = CellFunction('size_t', mesh)
        # setup error estimator vector
        eq_est = np.zeros(DG0.dim())

        # setup global equilibrated flux vector
        DG = VectorFunctionSpace(mesh, "DG", degree)
        DG_dofmap = DG.dofmap()

        # define form functions
        tau = TrialFunction(DG)
        v = TestFunction(DG)

        # define global tau
        tau_global = Function(DG)
        tau_global.vector()[:] = 0.0

        # iterate vertices
        for vertex in vertices(mesh):
            # get patch cell indices
            vid = vertex.index()
            patch_cid, FF_inner, FF_boundary = get_vertex_patch(vid, mesh, layers=1)

            # set nodal base function
            phi_coeffs[:] = 0
            phi_coeffs[dof_list.index(vid)] = 1
            phi_z.vector()[:] = phi_coeffs

            # set characteristic function and mark patch
            cf.set_all(0)
            xi_coeffs[:] = 0
            for cid in patch_cid:
                xi_coeffs[DG0_dofs[int(cid)]] = 1
                cf[int(cid)] = 1
            xi_z.vector()[:] = xi_coeffs

            # determine local dofs
            lDG_cell_dofs = dict([(cid, DG_dofmap.cell_dofs(cid)) for cid in patch_cid])
            lDG_dofs = [cd.tolist() for cd in lDG_cell_dofs.values()]
            lDG_dofs = list(iter.chain(*lDG_dofs))

            # print "\nlocal DG subspace has dimension", len(lDG_dofs), "degree", degree, "cells", len(patch_cid), patch_cid
            # print "local DG_cell_dofs", lDG_cell_dofs
            # print "local DG_dofs", lDG_dofs

            # create patch measures
            dx = Measure('dx')[cf]
            dS = Measure('dS')[FF_inner]

            # define forms
            alpha = Constant(1 / epsilon) / h
            a = inner(tau,v) * phi_z * dx(1) + alpha * div(tau) * div(v) * dx(1) + avg(alpha) * jump(tau,n) * jump(v,n) * dS(1)\
                + avg(alpha) * jump(xi_z * tau,n) * jump(v,n) * dS(2)
            L = -alpha * (div(sigma_mu) + f) * div(v) * phi_z * dx(1)\
                - avg(alpha) * jump(sigma_mu,n) * jump(v,n) * avg(phi_z)*dS(1)

    #        print "L2 f + div(sigma)", assemble((f + div(sigma)) * (f + div(sigma)) * dx(0))

            # assemble forms
            lhs = assemble(a, form_compiler_parameters={'quadrature_degree': quadrature_degree})
            rhs = assemble(L, form_compiler_parameters={'quadrature_degree': quadrature_degree})

            # convert DOLFIN representation to scipy sparse arrays
            rows, cols, values = lhs.data()
            lhsA = sps.csr_matrix((values, cols, rows)).tocoo()

            # slice sparse matrix and solve linear problem
            lhsA = coo_submatrix_pull(lhsA, lDG_dofs, lDG_dofs)
            lx = spsolve(lhsA, rhs.array()[lDG_dofs])
            # print ">>> local solution lx", type(lx), lx
            local_tau = Function(DG)
            local_tau.vector()[lDG_dofs] = lx
            # print "div(tau)", assemble(inner(div(local_tau),div(local_tau))*dx(1))

            # add up local fluxes
            tau_global.vector()[lDG_dofs] += lx

        # evaluate estimator
        # maybe TODO: re-define measure dx
        eq_est = assemble( inner(tau_global, tau_global) * dg0 * (dx(0)+dx(1)),\
                           form_compiler_parameters={'quadrature_degree': quadrature_degree})

        # reorder according to cell ids
        eq_est = eq_est[DG0_dofs.values()].array()
        global_est = np.sqrt(np.sum(eq_est))
        # eq_est_global = assemble( inner(tau_global, tau_global) * (dx(0)+dx(1)), form_compiler_parameters={'quadrature_degree': quadrature_degree} )
        # global_est2 = np.sqrt(np.sum(eq_est_global))
        return global_est, np.sqrt(eq_est)#, tau_global


def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of a sparse.coo_matrix.
    https://gist.github.com/dwf/828099
    """
    if type(rows) != np.ndarray:
        rows = np.array(rows)
        cols = np.array(cols)

    if type(matr) != sps.coo_matrix:
        raise TypeError('matrix must be sparse COOrdinate format')

    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])

    lr = len(rows)
    lc = len(cols)

    ar = np.arange(0, lr)
    ac = np.arange(0, lc)

    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sps.coo_matrix((matr.data[newelem], np.array([gr[newrows], gc[newcols]])), (lr, lc)).tocsr()


def get_vertex_patch(vid, mesh, layers=1):
    # find patch cells
    patch_cid = []
    for l in range(layers):
        if l == 0:
            patch_cid = Vertex(mesh,vid).entities(2).tolist()
        else:
            new_cid = []
            for cid in patch_cid:
                for cid in Cell(mesh,cid).entities(2).tolist():
                    if cid not in patch_cid:
                        new_cid.append(cid)
            patch_cid = patch_cid + new_cid

    # determine all facets
    FF_inner = FacetFunction('size_t', mesh)
    inner_facets = [Cell(mesh,cid).entities(1).tolist() for cid in patch_cid]
    inner_facets = set(iter.chain(*inner_facets))
    for fid in inner_facets:
        FF_inner[int(fid)] = 1

    # determine boundary facets
    FF_boundary = FacetFunction('size_t', mesh)
    for cid in patch_cid:
        c = Cell(mesh,cid)
        for f in facets(c):
            flist = f.entities(2).tolist()
            for fcid in flist:
                if fcid not in patch_cid or len(flist)==1:
                    FF_boundary[f.index()] = 1
                    FF_inner[f.index()] = 2 if len(flist)==2 else 3 # mark patch boundary facet (3 for outer Dirichlet boundary)
                    break
    return patch_cid, FF_inner, FF_boundary
