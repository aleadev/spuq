"""FEniCS FEM discretisation implementation for Poisson model problem"""
from dolfin import (TrialFunction, TestFunction, FunctionSpace, VectorFunctionSpace, Identity, Measure, FacetFunction,
                    dot, nabla_grad, div, tr, sym, inner, assemble, dx, Constant, DirichletBC, assemble_system)

from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
from spuq.fem.fenics.fenics_utils import remove_boundary_entries, set_dirichlet_bc_entries
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.utils.type_check import takes, anything

import numpy as np

default_Dirichlet_boundary = lambda x, on_boundary: on_boundary


class FEMDiscretisationBase(FEMDiscretisation):

    def create_dirichlet_bcs(self, V, uD, boundary):
        """Apply Dirichlet boundary conditions."""
        if uD is None:
            uD = self._uD
            if self._dirichlet_boundary is not None:
                boundary = self._dirichlet_boundary
        try:
            V = V._fefs
        except:
            pass
        
        if not isinstance(boundary, (tuple, list)):
            boundary = [boundary]
        if not isinstance(uD, (tuple, list)):
            uD = [uD]
        if len(uD) == 1:
            uD *= len(boundary)
        
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


    def assemble_operator_inner_dofs(self, coeff, basis):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = self.assemble_lhs(coeff, basis)
        bcs = self.create_dirichlet_bcs(basis, self._uD, self._dirichlet_boundary)
        remove_boundary_entries(matrix, bcs)
        return FEniCSOperator(matrix, basis)

    def set_dirichlet_bc_entries(self, u, homogeneous=False):
        bcs = self.create_dirichlet_bcs(u.basis, self._uD, self._dirichlet_boundary)
        set_dirichlet_bc_entries(u.coeffs, bcs, homogeneous)

    def copy_dirichlet_bc(self, d, b):
        for mu in d.active_indices():
            bcs = self.create_dirichlet_bcs(d[mu].basis, self._uD, self._dirichlet_boundary)
            for bc in bcs:
                dofs = bc.get_boundary_values().keys()
                print b[mu], d[mu]
                b[mu].coeffs[dofs] = d[mu].coeffs[dofs]

    
class FEMPoisson(FEMDiscretisationBase):
    """FEM discrete Laplace operator with coefficient :math:`a` on domain :math:`\Omega:=[0,1]^2` with homogeneous Dirichlet boundary conditions.

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    def __init__(self, f=Constant(1.0), a0=1,
                 dirichlet_boundary=default_Dirichlet_boundary, uD=None,
                 neumann_boundary=None, g=None):
        self._f = f
        self._a0 = a0
        self._dirichlet_boundary = dirichlet_boundary
        self._uD = uD
        self._neumann_boundary = neumann_boundary
        self._g = g

    @property
    def f(self):
        return self._f

    @property
    def norm(self):
        '''Energy norm wrt operator.'''
        return lambda v: np.sqrt(assemble(self._a0 * inner(nabla_grad(v), nabla_grad(v)) * dx))

    def function_space(self, mesh, degree=1):
        return FunctionSpace(mesh, "CG", degree=degree)

    def assemble_operator(self, coeff, basis, withBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = self.assemble_lhs(coeff, basis, withBC=withBC)
        return FEniCSOperator(matrix, basis)

    def assemble_solve_operator(self, coeff, basis, withBC=True):
        matrix = self.assemble_lhs(coeff, basis, withBC=withBC)
        return FEniCSSolveOperator(matrix, basis)

    def assemble_lhs(self, coeff, basis, withBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        if withBC:
            bcs = self.create_dirichlet_bcs(V, self._uD, self._dirichlet_boundary)
        else:
            bcs = []
        A, _ = assemble_system(a, v * dx, bcs)
        return A

    def assemble_rhs(self, coeff, basis, withBC=True, f=None):
        """Assemble the discrete right-hand side."""
        if f is None:
            f = self._f
        Dirichlet_boundary = self._dirichlet_boundary
        uD = self._uD

        # get FEniCS function space
        V = basis._fefs
        # assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)

        l = (f * v) * dx
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx

        # treat Neumann boundary
        if self._neumann_boundary is not None:
            Ng, ds = self._prepareNeumann(V.mesh())
            for j in range(len(Ng)):
                l -= dot(Ng[j], v) * ds(j + 1)
        
        if withBC:
            bcs = self.create_dirichlet_bcs(V, self._uD, self._dirichlet_boundary)
        else:
            bcs = []

        # assemble linear form
        _, F = assemble_system(a, l, bcs)
        return F
            
    def _prepareNeumann(self, mesh):
        assert self._g is not None
        boundary = self._neumann_boundary
        g = self._g
        # mark boundary
        if not isinstance(boundary, (tuple, list)):
            boundary = [boundary]
        if not isinstance(g, (tuple, list)):
            g = [g]
        if len(g) == 1:
            g *= len(boundary)
        parts = FacetFunction("uint", mesh, mesh.topology().dim() - 1)
        parts.set_all(0)
        for j, bnd in enumerate(boundary):
            bnd.mark(parts, j + 1)
        # evaluate boundary flux terms
        ds = Measure("ds")[parts]
        return g, ds

    def sigma(self, a, v):
        """Flux."""
        return a * nabla_grad(v)
    
    def Dsigma(self, a, v):
        """First derivative of flux."""
        if v.ufl_element().degree() < 2:
            return dot(nabla_grad(a), nabla_grad(v))
        else:
            return dot(nabla_grad(a), nabla_grad(v)) + a * div(nabla_grad(v))

    def r_T(self, a, v):
        """Volume residual."""
        return self.Dsigma(a, v)

    def r_E(self, a, v, nu):
        """Edge residual."""
        return a * dot(nabla_grad(v), nu)

    def r_Nb(self, a, v, nu, mesh):
        """Neumann boundary residual."""
        form = None
        if self._neumann_boundary is not None:
            form = []
            g, ds = self._prepareNeumann(mesh)
            for j, gj in enumerate(g):
                # TODO: check sign!
                Nbres = gj - dot(nabla_grad(v), nu)
                form.append((a * inner(Nbres, Nbres), ds(j + 1)))
        return form


class FEMNavierLame(FEMDiscretisationBase):
    """FEM discrete Navier-Lame equation (linearised elasticity) with parameters :math:`E` and :math:`\nu` with provided boundary conditions.

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    def __init__(self, mu, lmbda0,
                 f=Constant(1.0),
                 dirichlet_boundary=default_Dirichlet_boundary, uD=None,
                 neumann_boundary=None, g=None):
        self.lmbda0 = lmbda0
        self.mu = mu
        self._f = f
        self._dirichlet_boundary = dirichlet_boundary
        self._uD = uD
        self._neumann_boundary = neumann_boundary
        self._g = g

    @property
    def f(self):
        return self._f

    @property
    def norm(self):
        '''Energy norm wrt operator, i.e. (\sigma(v),\eps(v))=||C^{1/2}\eps(v)||.'''
        return lambda v: np.sqrt(assemble(inner(self.sigma(self.lmbda0, self.mu, v), sym(nabla_grad(v))) * dx))

    def function_space(self, mesh, degree=1):
        return VectorFunctionSpace(mesh, "CG", degree=degree)

    def assemble_operator(self, lmbda, basis, withBC=True):
        """Assemble the discrete problem and return as Operator."""
        matrix = self.assemble_lhs(lmbda, basis, withBC=withBC)
        return FEniCSOperator(matrix, basis)

    def assemble_solve_operator(self, lmbda, basis, withBC=True):
        matrix = self.assemble_lhs(lmbda, basis, withBC=withBC)
        return FEniCSSolveOperator(matrix, basis)

    def assemble_lhs(self, lmbda, basis, withBC=True):
        """Assemble the discrete operator."""
        f = self._f
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(self.sigma(lmbda, self.mu, u), sym(nabla_grad(v))) * dx
        l = inner(f, v) * dx

        if withBC:
            bcs = self.create_dirichlet_bcs(V, self._uD, self._dirichlet_boundary)
        else:
            bcs = []
        A, _ = assemble_system(a, l, bcs)
        return A

    def assemble_rhs(self, lmbda, basis, withBC=True):
        """Assemble the discrete right-hand side."""
        f = self._f
        Dirichlet_boundary = self._dirichlet_boundary
        uD = self._uD

        # get FEniCS function space
        V = basis._fefs
        # define linear form
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(self.sigma(lmbda, self.mu, u), sym(nabla_grad(v))) * dx
        l = inner(f, v) * dx
        
        # treat Neumann boundary
        if self._neumann_boundary is not None:
            Ng, ds = self._prepareNeumann(V.mesh())            
            for j in range(len(Ng)):
                l -= dot(Ng[j], v) * ds(j + 1)
                        
        if withBC:
            bcs = self.create_dirichlet_bcs(V, self._uD, self._dirichlet_boundary)
        else:
            bcs = []

        # assemble linear form
        _, F = assemble_system(a, l, bcs)
        return F
            
    def _prepareNeumann(self, mesh):
        assert self._g is not None
        boundary = self._neumann_boundary
        g = self._g
        # mark boundary
        if not isinstance(boundary, (tuple, list)):
            boundary = [boundary]
        if not isinstance(g, (tuple, list)):
            g = [g]
        if len(g) == 1:
            g *= len(boundary)
        parts = FacetFunction("uint", mesh, mesh.topology().dim() - 1)
        parts.set_all(0)
        for j, bnd in enumerate(boundary):
            bnd.mark(parts, j + 1)
        # evaluate boundary flux terms
        ds = Measure("ds")[parts]
        return g, ds

    def sigma(self, lmbda, mu, v):
        """Flux."""
        return 2.0 * mu * sym(nabla_grad(v)) + lmbda * tr(sym(nabla_grad(v))) * Identity(v.cell().d)
    
    def Dsigma(self, lmbda, mu, v):
        """First derivative of flux."""
        if v.ufl_element().degree() < 2:
            return 2.0 * mu * div(sym(nabla_grad(v))) + dot(nabla_grad(lmbda), tr(sym(nabla_grad(v))) * Identity(v.cell().d))
        else:
            return 2.0 * mu * div(sym(nabla_grad(v))) + dot(nabla_grad(lmbda), tr(sym(nabla_grad(v))) * Identity(v.cell().d)) + lmbda * div(tr(sym(nabla_grad(v))) * Identity(v.cell().d))

    def r_T(self, lmbda, v):
        """Volume residual."""
        return self.Dsigma(lmbda, self.mu, v)

    def r_E(self, lmbda, v, nu):
        """Edge residual."""
        return dot(self.sigma(lmbda, self.mu, v), nu)

    def r_Nb(self, lmbda, v, nu, mesh):
        """Neumann boundary residual."""
        form = None
        if self._neumann_boundary is not None:
            form = []
            g, ds = self._prepareNeumann(mesh)
            for j, gj in enumerate(g):
                # TODO: check sign!
                Nbres = gj - dot(self.sigma(lmbda, self.mu, v), nu)
                form.append((inner(Nbres, Nbres), ds(j + 1)))


class FEMPoissonOld(FEMPoisson):

    def assemble_lhs(self, coeff, basis, withBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        # apply Dirichlet bc
        if withBC:
            A = self.apply_dirichlet_bc(V, A=A, uD=self._uD, Dirichlet_boundary=self._dirichlet_boundary)
        return A

    def assemble_rhs(self, basis, withBC=True):
        """Assemble the discrete right-hand side."""
        f = self._f
        Dirichlet_boundary = self._dirichlet_boundary
        uD = self._uD

        # get FEniCS function space
        V = basis._fefs
        # assemble and apply boundary conditions
        v = TestFunction(V)
        l = (f * v) * dx

        # treat Neumann boundary
        if self._neumann_boundary is not None:
            Ng, ds = self._prepareNeumann(V.mesh())
            for j in range(len(Ng)):
                l -= dot(Ng[j], v) * ds(j + 1)
        
        # assemble linear form
        F = assemble(l)
        # apply Dirichlet bc
        if withBC:
            F = self.apply_dirichlet_bc(V, b=F, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return F
