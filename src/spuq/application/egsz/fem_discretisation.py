"""FEniCS FEM discretisation implementation for Poisson model problem"""
from dolfin import (TrialFunction, TestFunction, FunctionSpace, VectorFunctionSpace, Identity, Measure, FacetFunction,
                    dot, nabla_grad, div, tr, sym, inner, assemble, dx, Constant, DirichletBC, assemble_system, cells)
import dolfin
import ufl.expr

from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
from spuq.fem.fenics.fenics_utils import get_dirichlet_mask, set_dirichlet_bc_entries
#from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.utils.type_check import takes, anything, optional, returns, tuple_of, takes_verbose, sequence_of

import numpy as np
import collections
from abc import ABCMeta, abstractmethod

default_Dirichlet_boundary = lambda x, on_boundary: on_boundary

CoefficientFunction = (dolfin.Expression, dolfin.GenericFunction, ufl.expr.Expr, tuple_of((dolfin.Expression, dolfin.GenericFunction, ufl.expr.Expr, float, int)))
FormFunction = (dolfin.Argument, dolfin.Function)
LoadingFunction = (dolfin.Coefficient)
BoundaryType = (anything)
BoundaryFunction = (dolfin.Coefficient)

###################################################
# Helper functions
###################################################

@takes(anything, optional(int))
def make_list(x, length=None):
    """Make a sequence type out of some item if it not already is one"""
    if not isinstance(x, collections.Sequence):
        x = [x]
    if length is not None:
        if len(x) == 1:
            x = x * length
        assert len(x) == length

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
    """Returns degree of ufl_element of u"""
    return u.function_space().ufl_element().degree()


def get_default(x, default_x):
    """Returns the first parameter if not None, otherwise the second."""
    return x if x is not None else default_x

###################################################
# FEniCS Hacks
###################################################

NEEDS_HACK = False
if NEEDS_HACK:
    def _assemble_system(a, L, bcs, facet_function):
        # Mean hack to work around a bug in the FEniCS assemble_system
        # function that doesn't treat the exterior facet domains correctly
        # (Note: we assemble twice to get the right type and size of the
        # matrix A and vector b. This is pretty inefficient, but remember:
        # it's a hack that should be obsolete when the FEniCS guys have
        # their code fixed)
        A = dolfin.assemble(a)
        b = dolfin.assemble(L)
        a = dolfin.fem.Form(a, subdomains={'exterior_facet': facet_function})
        L = dolfin.fem.Form(L, subdomains={'exterior_facet': facet_function})
        # somehow the SystemAssembler seems to work in contrast to
        # assemble_system
        sa = dolfin.SystemAssembler()
        sa.assemble(A, b, a, L, bcs)
        return A, b
else:
    def _assemble_system(a, L, bcs, facet_function):
        return assemble_system(a, L, bcs, exterior_facet_domains=facet_function)

###################################################
# Weak Forms
###################################################

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

    @takes(anything, dolfin.FunctionSpaceBase, CoefficientFunction)
    def loading_linear_form(self, V, f):
        """Return the linear form L(v) for the loading"""
        v = dolfin.TrialFunction(V)
        return inner(f, v) * dx

    @takes(anything, dolfin.FunctionSpaceBase, collections.Sequence, collections.Sequence)
    def neumann_linear_form(self, V, neumann_boundary, g, L=None):
        """Return or add up the linear form L(v) coming from the Neumann boundary"""
        v = dolfin.TrialFunction(V)
        for g_j, ds_j in self.neumann_form_list(neumann_boundary, g, V.mesh()):
            if L is None:
                L = dot(g_j, v) * ds_j
            else:
                L += dot(g_j, v) * ds_j
        return L

    @takes(anything, anything, anything, dolfin.Mesh)
    def neumann_facet_function(self, boundaries, g, mesh):
        boundaries = make_list(boundaries)
        g = make_list(g, len(boundaries))
        # create FacetFunction to mark different Neumann boundaries with ids 0, 1, ...
        parts = FacetFunction("size_t", mesh, 0)
        for j, bnd_domain in enumerate(boundaries, 1):
            bnd_domain.mark(parts, j)
        return parts

    @takes(anything, anything, anything, dolfin.Mesh)
    def neumann_form_list(self, boundaries, g, mesh):
        boundaries = make_list(boundaries)
        g = make_list(g, len(boundaries))
        parts = self.neumann_facet_function(boundaries, g, mesh)
        # create Neumann measures wrt Neumann boundaries
        ds = Measure("ds")[parts]
        # return Neumann data together with boundary measures
        return [(gj, ds(j)) for j, gj in enumerate(g, 1)]


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
        if element_degree(u) >= 2:
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
        return 2.0 * mu * Du + lmbda * tr(Du) * I

    @takes(anything, FormFunction, CoefficientFunction)
    def flux_derivative(self, u, coeff):
        """First derivative of flux."""
        lmbda, mu = coeff
        Du = self.differential_op(u)
        I = Identity(u.cell().d)
        Dsigma = 2.0 * mu * div(Du) + dot(nabla_grad(lmbda), tr(Du) * I)
        if element_degree(u) >= 2:
            Dsigma += lmbda * div(tr(Du) * I)
        return Dsigma


###################################################
# FEM Discretisation
###################################################

class FEMDiscretisation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def assemble_operator(self, basis, coeff, withDirichletBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        raise NotImplementedError

    @abstractmethod
    def assemble_solve_operator(self, basis, coeff, withDirichletBC=True):
        """Assemble the discrete problem and return a SolveOperator."""
        raise NotImplementedError

    @abstractmethod
    def assemble_operator_inner_dofs(self, basis, coeff):
        """Assemble the discrete problem and return as Operator
        (projected on the inner DOFs, i.e. all Dirichlet BC entries set to zero)."""
        raise NotImplementedError

    @abstractmethod
    def assemble_rhs(self, basis, coeff, withDirichletBC=True, withNeumannBC=True, f=None):
        """Assemble the discrete right-hand side."""
        raise NotImplementedError


class FEMDiscretisationBase(FEMDiscretisation):
    @takes_verbose(anything, WeakForm, CoefficientFunction, optional(LoadingFunction),
           optional(sequence_of(BoundaryType)), optional(sequence_of(BoundaryFunction)),
           optional(sequence_of(BoundaryType)), optional(sequence_of(BoundaryFunction)))
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

    @takes(anything, FEniCSBasis, optional(CoefficientFunction), optional(bool))
    def assemble_lhs(self, basis, coeff=None, withDirichletBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs
        coeff = get_default(coeff, self.coeff)

        a = self.weak_form.bilinear_form(V, coeff)
        L = self.weak_form.loading_linear_form(V, self.f)
        bcs = []
        if withDirichletBC:
            bcs = self.create_dirichlet_bcs(V, self.uD, self.dirichlet_boundary)
        A, _ = assemble_system(a, L, bcs)
        return A

    @takes_verbose(anything, FEniCSBasis, optional(CoefficientFunction), optional(bool), optional(bool), optional(FormFunction))
    def assemble_rhs(self, basis, coeff=None, withDirichletBC=True, withNeumannBC=True, f=None):
        """Assemble the discrete right-hand side."""
        coeff = get_default(coeff, self.coeff)
        f = get_default(f, self.f)
        Dirichlet_boundary = self.dirichlet_boundary
        uD = self.uD

        # get FEniCS function space
        V = basis._fefs
        a = self.weak_form.bilinear_form(V, coeff)
        L = self.weak_form.loading_linear_form(V, f)

        # treat Neumann boundary
        if withNeumannBC and self.neumann_boundary:
            L += self.weak_form.neumann_linear_form(V, self.neumann_boundary, self.g)

        # treat Dirichlet boundary
        bcs = []
        if withDirichletBC:
            bcs = self.create_dirichlet_bcs(V, self.uD, self.dirichlet_boundary)

        # assemble linear form
        if True:    # activate quick hack for system assembler
            facet_function = self.weak_form.neumann_facet_function(self.neumann_boundary, self.g, V.mesh())
            _, F = _assemble_system(a, L, bcs, facet_function)
        else:
            _, F = assemble_system(a, L, bcs)
        return F

    @takes_verbose(anything, FEniCSBasis, optional(CoefficientFunction), optional(bool))
    def assemble_operator(self, basis, coeff=None, withDirichletBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        coeff = get_default(coeff, self.coeff)
        matrix = self.assemble_lhs(basis, coeff, withDirichletBC=withDirichletBC)
        return FEniCSOperator(matrix, basis)

    @takes_verbose(anything, FEniCSBasis, optional(CoefficientFunction), optional(bool))
    def assemble_solve_operator(self, basis, coeff=None, withDirichletBC=True):
        coeff = get_default(coeff, self.coeff)
        matrix = self.assemble_lhs(basis, coeff, withDirichletBC=withDirichletBC)
        return FEniCSSolveOperator(matrix, basis)

    @takes_verbose(anything, FEniCSBasis, optional(CoefficientFunction))
    def assemble_operator_inner_dofs(self, basis, coeff=None):
        """Assemble the discrete problem and return as Operator
        (projected on the inner DOFs, i.e. all Dirichlet BC entries set to zero)."""
        coeff = get_default(coeff, self.coeff)
        matrix = self.assemble_lhs(basis, coeff)
        bcs = self.create_dirichlet_bcs(basis._fefs, self.uD, self.dirichlet_boundary)
        mask = get_dirichlet_mask(matrix, bcs)
        return FEniCSOperator(matrix, basis, mask)

    @takes_verbose(anything, FEniCSVector, optional(bool))
    def set_dirichlet_bc_entries(self, u, homogeneous=False):
        bcs = self.create_dirichlet_bcs(u.basis._fefs, self.uD, self.dirichlet_boundary)
        set_dirichlet_bc_entries(u.coeffs, bcs, homogeneous)

    @takes_verbose(anything, dolfin.FunctionSpaceBase, optional(list), optional(list))
    def create_dirichlet_bcs(self, V, uD=None, boundary=None):
        """Create list of FEniCS boundary condition objects."""
        if uD is None:
            uD = self.uD
            if self.dirichlet_boundary is not None:
                boundary = self.dirichlet_boundary
        
        boundary = make_list(boundary)
        uD = make_list(uD, len(boundary))
        
        bcs = [DirichletBC(V, cuD, cDb) for cuD, cDb in zip(uD, boundary)]
        return bcs

    def apply_dirichlet_bc(self, V, A=None, b=None, uD=None, Dirichlet_boundary=None):
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

    @takes(anything, CoefficientFunction, FormFunction)
    def volume_residual(self, coeff, u):
        """Volume residual r_T."""
        return self.weak_form.flux_derivative(u, coeff)

    @takes(anything, CoefficientFunction, FormFunction)
    def edge_residual(self, coeff, v, nu):
        """Edge residual r_E."""
        return dot(self.weak_form.flux(v, coeff), nu)

    @takes(anything, CoefficientFunction, FormFunction)
    def neumann_residual(self, coeff, v, nu, mesh, homogeneous=False):
        """Neumann boundary residual."""
        form = []
        a = coeff
        boundaries = self.neumann_boundary
        g = self.g
        if boundaries is not None:
            if homogeneous:
                g = zero_function(v.function_space())

            for g_j, ds_j in self.weak_form.neumann_form_list(boundaries, g, mesh):
                r_j = g_j - a * dot(self.weak_form.flux(v, coeff), nu)
                form.append((r_j, ds_j))
        return form

    @property
    def energy_norm(self):
        return self.get_energy_norm()

    def get_energy_norm(self, mesh=None):
        '''Energy norm wrt operator, i.e. (\sigma(v),\eps(v))=||C^{1/2}\eps(v)||.'''
        if mesh is None:
            def energy_norm(v):
                return np.sqrt(assemble(inner(self.weak_form.flux(v, self.coeff), self.weak_form.differential_op(v)) * dx))
            return energy_norm
        else:
            DG = FunctionSpace(mesh, "DG", 0)
            s = TestFunction(DG)
            def energy_norm(v):
                ae = np.sqrt(assemble(inner(self.weak_form.flux(v, self.coeff), self.weak_form.differential_op(v)) * s * dx))
                # reorder DG dofs wrt cell indices
                dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
                norm_vec = ae[dofs]
                return norm_vec
            return energy_norm


class FEMPoisson(FEMDiscretisationBase):
    def __init__(self, a=Constant(1.0), f=Constant(1.0),
                 dirichlet_boundary=[default_Dirichlet_boundary], uD=[Constant(0.0)],
                 neumann_boundary=None, g=None):
        super(FEMPoisson, self).__init__(PoissonWeakForm(), a, f,
                                         dirichlet_boundary, uD,
                                         neumann_boundary, g)


class FEMNavierLame(FEMDiscretisationBase):
    def __init__(self, mu, lmbda, f=Constant((0.0, 0.0)),
                 dirichlet_boundary=[default_Dirichlet_boundary], uD=[Constant((0.0, 0.0))],
                 neumann_boundary=None, g=None):
        super(FEMNavierLame, self).__init__(NavierLameWeakForm(), (mu, lmbda), f,
                                            dirichlet_boundary, uD,
                                            neumann_boundary, g)

    @takes(anything, CoefficientFunction, FormFunction)
    def neumann_residual(self, coeff, v, nu, mesh, homogeneous=False):
        # the coefficient field does not influence the Neumann boundary for Navier-Lame!
        """Neumann boundary residual."""
        form = []
        boundaries = self.neumann_boundary
        g = self.g
        if boundaries is not None:
            if homogeneous:
                g = zero_function(v.function_space())

            for g_j, ds_j in self.weak_form.neumann_form_list(boundaries, g, mesh):
                r_j = g_j - dot(self.weak_form.flux(v, coeff), nu)
                form.append((r_j, ds_j))
        return form
