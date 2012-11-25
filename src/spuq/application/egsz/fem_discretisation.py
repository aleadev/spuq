"""FEniCS FEM discretisation implementation for Poisson model problem"""
from dolfin import (TrialFunction, TestFunction, FunctionSpace, VectorFunctionSpace, Identity, Measure, FacetFunction,
                    dot, nabla_grad, div, tr, sym, inner, assemble, dx, Constant, DirichletBC, assemble_system, cells)
import dolfin

from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
from spuq.fem.fenics.fenics_utils import get_dirichlet_mask, set_dirichlet_bc_entries
#from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.utils.type_check import takes, anything, optional, returns

import numpy as np
import collections
from abc import ABCMeta, abstractmethod

default_Dirichlet_boundary = lambda x, on_boundary: on_boundary

CoefficientFunction = (dolfin.Expression, dolfin.GenericFunction)
FormFunction = (dolfin.Argument, dolfin.Function)

@takes(anything, optional(int))
def make_list(x, length=None):
    """Make a sequence type out of some item if it not already is one"""
    if not isinstance(x, collections.Sequence):
        x = [x]
    if length is not None and len(x)==1:
        x = x * length
    return x


@takes(dolfin.FunctionSpaceBase)
def zero_function(V):
    if V.num_sub_spaces():
        d = V.mesh().geometry().dim()
        return Constant([0] * d)
    else:
        return Constant(0)


@takes((dolfin.Argument, dolfin.Function))
def element_degree(u):
    """Returns true if u has an ufl_element and at least element degree 2"""
    return u.function_space().ufl_element().degree()


class FEMDiscretisation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def assemble_operator(self, coeff, basis, withDirichletBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        raise NotImplementedError

    @abstractmethod
    def assemble_solve_operator(self, coeff, basis, withDirichletBC=True):
        """Assemble the discrete problem and return a SolveOperator."""
        raise NotImplementedError

    @abstractmethod
    def assemble_operator_inner_dofs(self, coeff, basis):
        """Assemble the discrete problem and return as Operator
        (projected on the inner DOFs, i.e. all Dirichlet BC entries set to zero)."""
        raise NotImplementedError

    @abstractmethod
    def assemble_rhs(self, coeff, basis, withDirichletBC=True, withNeumannBC=True, f=None):
        """Assemble the discrete right-hand side."""
        raise NotImplementedError


class FEMDiscretisationBase(FEMDiscretisation):
    def __init__(self, weak_form, coeff, f,
                 dirichlet_boundary, uD,
                 neumann_boundary, g):
        self.weak_form = weak_form
        self.coeff = coeff
        self.f = f
        self.dirichlet_boundary = dirichlet_boundary
        self.uD = uD
        self.neumann_boundary = neumann_boundary
        self.g = g

    @takes(anything, dolfin.Mesh, int)
    def function_space(self, mesh, degree=1):
        """Return the FunctionSpace V used from the weak form"""
        return self.weak_form.function_space(mesh, degree)
        

    @takes(anything, CoefficientFunction, FEniCSBasis, optional(bool))
    def assemble_lhs(self, coeff, basis, withDirichletBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs

        a = self.weak_form.bilinear_form(V, coeff)
        L = self.weak_form.loading_linear_form(V, self.f)
        bcs = []
        if withDirichletBC:
            bcs = self.create_dirichlet_bcs(V, self.uD, self.dirichlet_boundary)
        A, _ = assemble_system(a, L, bcs)
        return A

    @takes(anything, CoefficientFunction, FEniCSBasis, optional(bool), optional(bool), optional(FormFunction))
    def assemble_rhs(self, coeff, basis, withDirichletBC=True, withNeumannBC=True, f=None):
        """Assemble the discrete right-hand side."""
        f = f or self.f
        Dirichlet_boundary = self.dirichlet_boundary
        uD = self.uD

        # get FEniCS function space
        V = basis._fefs
        a = self.weak_form.bilinear_form(V, coeff)
        L = self.weak_form.loading_linear_form(V, self.f)

        # treat Neumann boundary
        if withNeumannBC and self.neumann_boundary is not None:
            g, ds = self._prepare_neumann(self.neumann_boundary, self.g, V.mesh())

            for gj, dsj in zip(g, ds):
                L += dot(gj, v) * dsj

        # treat Dirichlet boundary
        bcs = []
        if withDirichletBC:
            bcs = self.create_dirichlet_bcs(V, self.uD, self.dirichlet_boundary)

        # assemble linear form
        _, F = assemble_system(a, L, bcs)
        return F

    def assemble_operator(self, coeff, basis, withDirichletBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = self.assemble_lhs(coeff, basis, withDirichletBC=withDirichletBC)
        return FEniCSOperator(matrix, basis)

    def assemble_solve_operator(self, coeff, basis, withDirichletBC=True):
        matrix = self.assemble_lhs(coeff, basis, withDirichletBC=withDirichletBC)
        return FEniCSSolveOperator(matrix, basis)

    def assemble_operator_inner_dofs(self, coeff, basis):
        """Assemble the discrete problem and return as Operator
        (projected on the inner DOFs, i.e. all Dirichlet BC entries set to zero)."""
        matrix = self.assemble_lhs(coeff, basis)
        bcs = self.create_dirichlet_bcs(basis, self.uD, self.dirichlet_boundary)
        mask = get_dirichlet_mask(matrix, bcs)
        return FEniCSOperator(matrix, basis, mask)

    def set_dirichlet_bc_entries(self, u, homogeneous=False):
        bcs = self.create_dirichlet_bcs(u.basis, self.uD, self.dirichlet_boundary)
        set_dirichlet_bc_entries(u.coeffs, bcs, homogeneous)

    def create_dirichlet_bcs(self, V, uD=None, boundary=None):
        """Create list of FEniCS boundary condition objects."""
        if uD is None:
            uD = self.uD
            if self.dirichlet_boundary is not None:
                boundary = self.dirichlet_boundary
        try:
            V = V._fefs
        except:
            pass
        
        boundary = make_list(boundary)
        uD = make_list(uD, len(boundary))
        
        bcs = [DirichletBC(V, cuD, cDb) for cuD, cDb in zip(uD, boundary)]
        return bcs

    def apply_dirichlet_bc(self, V, A=None, b=None, uD=None, Dirichlet_boundary=default_Dirichlet_boundary):
        """Apply Dirichlet boundary conditions."""
        bcs = self.create_dirichlet_bcs(V, uD, Dirichlet_boundary)
        val = []
        if not A is None:
            for bc in bcs:
                bc.apply(A)
            val.append(A)
        if not b is None:
            for bc in bcs:
                bc.apply(b)
            val.append(b)
        if len(val) == 1:
            val = val[0]
        return val

    def volume_residual(self, u, coeff):
        """Volume residual r_T."""
        return self.weak_form.flux_derivative(u, coeff)

    def edge_residual(self, coeff, v, nu):
        """Edge residual r_E."""
        return dot(self.weak_form.flux(v, coeff), nu)

    @staticmethod
    def _prepare_neumann(self, boundaries, g, mesh):
        boundaries = make_list(boundaries)
        g = make_list(g, len(boundaries))

        parts = FacetFunction("sizet", mesh, 0)
        for j, bnd_domain in enumerate(boundaries):
            bnd_domain.mark(parts, j + 1)
        ds = Measure("ds")[parts]
        return g, ds

    def neumann_residual(self, coeff, v, nu, mesh, homogeneous=False):
        """Neumann boundary residual."""
        form = []
        boundaries = self.neumann_boundary
        g = self.g
        if boundaries is not None:
            if homogeneous:
                g = zero_function(v.function_space())

            g, ds = self.prepare_neumann(boundaries, g, mesh)
            for gj, dsj in zip(g, ds):
                Nbres = gj - dot(self.weak_form.flux(v, coeff), nu)
                form.append((inner(Nbres, Nbres), dsj))
        return form


class FEMPoisson(FEMDiscretisationBase):
    def __init__(self, a0=Constant(1.0), f=Constant(1.0), 
                 dirichlet_boundary=default_Dirichlet_boundary, uD=Constant(0.0),
                 neumann_boundary=None, g=None):
        super(FEMPoisson, self).__init__(PoissonWeakForm(), a0, f, 
                                         dirichlet_boundary, uD,
                                         neumann_boundary, g)


class FEMNavierLame(FEMDiscretisationBase):
    def __init__(self, mu, lmbda, f=Constant(1.0),
                 dirichlet_boundary=default_Dirichlet_boundary, uD=None,
                 neumann_boundary=None, g=None):
        super(FEMNavierLame, self).__init__(NavierLameWeakForm(), (mu, lmbda), f, 
                                            dirichlet_boundary, uD,
                                            neumann_boundary, g)
        


class WeakForm(object):
    """Base class for WeakForms, that can be assembled into a discrete
    form using FEMDiscretisation methods."""
    __metaclass__ = ABCMeta

    @abstractmethod
    @takes(anything, dolfin.Mesh, int)
    def function_space(self, mesh, degree=1):
        """Return the FunctionSpace V used for this weak form"""
        raise NotImplementedError

    @abstractmethod
    @takes(anything, dolfin.FunctionSpaceBase, CoefficientFunction)
    def bilinear_form(self, V, coeff):
        """Return the bilinear a(u,v) form for the operator"""
        raise NotImplementedError

    @abstractmethod
    @takes(anything, dolfin.FunctionSpaceBase, FormFunction)
    def loading_linear_form(self, V, f):
        """Return the linear form L(v) for the loading"""
        raise NotImplementedError


class EllipticWeakForm(WeakForm):
    """Helper class for WeakForms, implementing some basic methods"""

    @takes(anything, dolfin.FunctionSpaceBase, CoefficientFunction)
    def bilinear_form(self, V, coeff):
        """Return the bilinear form for the operator"""
        u = dolfin.TrialFunction(V)
        v = dolfin.TestFunction(V)
        return inner(self.flux(u, coeff), self.differential_op(v)) * dx

    @abstractmethod
    @takes(anything, FormFunction)
    def differential_op(self, u):
        """Return the differential operator (Du) for this elliptic weak form (strain?)"""
        raise NotImplementedError

    @abstractmethod
    @takes(anything, FormFunction, CoefficientFunction)
    def flux(self, u, coeff):
        """Return the flux term (sigma(u)) for this elliptic weak form"""
        raise NotImplementedError

    @abstractmethod
    @takes(anything, FormFunction, CoefficientFunction)
    def flux_derivative(self, u, coeff):
        """First derivative of flux (Dsigma(u))."""
        raise NotImplementedError

    @takes(anything, dolfin.FunctionSpaceBase, CoefficientFunction)
    def loading_linear_form(self, V, f):
        """Return the linear form L(v) for the loading"""
        v = dolfin.TrialFunction(V)
        return inner(f, v) * dx


class PoissonWeakForm(EllipticWeakForm):
    """Weak form for the Poisson problem."""

    @takes(anything, dolfin.Mesh, int)
    def function_space(self, mesh, degree=1):
        return FunctionSpace(mesh, "CG", degree=degree)

    @takes(anything, FormFunction)
    def differential_op(self, u):
        return nabla_grad(u)

    @takes(anything, FormFunction, CoefficientFunction)
    def flux(self, u, coeff):
        a = coeff
        Du = self.differential_op(u)
        return a * Du

    @takes(anything, FormFunction, CoefficientFunction)
    def flux_derivative(self, u, coeff):
        a = coeff
        Du = self.differential_op(u)
        Dsigma = dot(nabla_grad(a), Du)
        if element_degree(u)>=2:
            Dsigma += a * div(Du)
        return Dsigma


class NavierLameWeakForm(EllipticWeakForm):
    """Weak form for the Navier-Lame problem."""

    @takes(anything, dolfin.Mesh, int)
    def function_space(self, mesh, degree=1):
        return VectorFunctionSpace(mesh, "CG", degree=degree)

    @takes(anything, FormFunction)
    def differential_op(self, u):
        return sym(nabla_grad(u))

    @takes(anything, FormFunction, CoefficientFunction)
    def flux(self, u, coeff):
        lmbda, mu = coeff
        Du = self.differential_op(u)
        I = Identity(u.cell().d)
        return 2.0 * mu * Du + lmbda * tr(Du)

    @takes(anything, FormFunction, CoefficientFunction)
    def flux_derivative(self, u, coeff):
        """First derivative of flux."""
        lmbda, mu = coeff
        Du = self.differential_op(u)
        I = Identity(u.cell().d)
        Dsigma = 2.0 * mu * div(Du) + dot(nabla_grad(lmbda), tr(Du) * I)
        if element_degree(u)>=2:
            Dsigma += lmbda * div(tr(Du) * I)
        return Dsigma
