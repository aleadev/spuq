"""FEniCS FEM discretisation implementation for Poisson model problem"""

from dolfin import (TrialFunction, TestFunction, FunctionSpace, VectorFunctionSpace, Identity,
                    dot, nabla_grad, div, tr, sym, inner, assemble, dx, Constant, DirichletBC)

from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.utils.type_check import takes, anything

default_Dirichlet_boundary = lambda x, on_boundary: on_boundary

class FEMPoisson(FEMDiscretisation):
    """FEM discrete Laplace operator with coefficient :math:`a` on domain :math:`\Omega:=[0,1]^2` with homogeneous Dirichlet boundary conditions.

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    def __init__(self, f=Constant(1.0), 
                 dirichlet_boundary=default_Dirichlet_boundary, uD=None, 
                 neumann_boundary=None, g=None):
        self._f = f
        self._dirichlet_boundary = dirichlet_boundary
        self._uD = uD
        self._neumann_boundary = neumann_boundary
        self._g = g

    def f(self):
        return self._f

    def function_space(self, mesh, degree=1):
        return FunctionSpace(mesh, "CG", degree=degree)

    def assemble_operator(self, coeff, basis, withBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = self.assemble_lhs(coeff, basis, uD=None, withBC=withBC, Dirichlet_boundary=self._dirichlet_boundary)
        return FEniCSOperator(matrix, basis)

    def assemble_solve_operator(self, coeff, basis, withBC=True):
        matrix = self.assemble_lhs(coeff, basis, uD=None, withBC=withBC, Dirichlet_boundary=self._dirichlet_boundary)
        return FEniCSSolveOperator(matrix, basis)

    def apply_dirichlet_bc(self, V, A=None, b=None, uD=None, Dirichlet_boundary=default_Dirichlet_boundary):
        """Apply Dirichlet boundary conditions."""
        if uD is None:
            uD = Constant(0.0)
        try:
            V = V._fefs
        except:
            pass
        
        if not isinstance(Dirichlet_boundary, (tuple, list)):
            Dirichlet_boundary = [Dirichlet_boundary]
        if not isinstance(uD, (tuple, list)):
            uD = [uD]
        if len(uD) == 1:
            uD *= len(Dirichlet_boundary)
        
        bcs = [DirichletBC(V, cuD, cDb) for cuD, cDb in zip(uD, Dirichlet_boundary)]
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

    def assemble_lhs(self, coeff, basis, uD=None, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        if withBC:
            A = self.apply_dirichlet_bc(V, A=A, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return A

    def assemble_rhs(self, basis, withBC=True):
        """Assemble the discrete right-hand side."""
        f = self._f
        Dirichlet_boundary = self._dirichlet_boundary
        uD = self._uD
        Neumann_boundary = self._neumann_boundary
        g = self._g

        # get FEniCS function space
        V = basis._fefs
        # assemble and apply boundary conditions
        v = TestFunction(V)
        l = (f * v) * dx
        if Neumann_boundary is not None:
            assert g is not None
            ds = Measure("ds")[Neumann_boundary]
            if not isinstance(Neumann_boundary, (tuple, list)):
                Neumann_boundary = [Neumann_boundary]
            if not isinstance(g, (tuple, list)):
                g = [g]
            for b in range(len(Neumann_boundary)):
                l -= dot(g, v) * ds(b)
        F = assemble(l)
        if withBC:
            F = self.apply_dirichlet_bc(V, b=F, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return F

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

    def r_Nb(self, a, v, nu):
        """Neumann boundary residual."""
        pass


class FEMNavierLame(FEMDiscretisation):
    """FEM discrete Navier-Lame equation (linearised elasticity) with parameters :math:`E` and :math:`\nu` with provided boundary conditions.

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    def __init__(self, mu):
        self.mu = mu

    def f(self, type=0):
        assert type == 0
        return Constant((0.0, 0.0))

    def function_space(self, mesh, degree=1):
        return VectorFunctionSpace(mesh, "CG", degree=degree)

    def assemble_operator(self, lmbda, basis, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        """Assemble the discrete problem and return as Operator."""
        matrix = self.assemble_lhs(lmbda, basis, uD=None, withBC=withBC, Dirichlet_boundary=Dirichlet_boundary)
        return FEniCSOperator(matrix, basis)

    def assemble_solve_operator(self, lmbda, basis, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        matrix = self.assemble_lhs(lmbda, basis, uD=None, withBC=withBC, Dirichlet_boundary=Dirichlet_boundary)
        return FEniCSSolveOperator(matrix, basis)

    def apply_dirichlet_bc(self, V, A=None, b=None, uD=None, Dirichlet_boundary=default_Dirichlet_boundary):
        """Apply Dirichlet boundary conditions."""
        if uD is None:
            uD = Constant((0.0, 0.0))
        try:
            V = V._fefs
        except:
            pass
            
        if not isinstance(Dirichlet_boundary, (tuple, list)):
            Dirichlet_boundary = [Dirichlet_boundary]
        if not isinstance(uD, (tuple, list)):
            uD = [uD]
        if len(uD) == 1:
            uD *= len(Dirichlet_boundary)
        
        bcs = [DirichletBC(V, cuD, cDb) for cuD, cDb in zip(uD, Dirichlet_boundary)]
        val = []
        if not A is None:
            for i, bc in enumerate(bcs):
#                print '#######LHS BC', i, "START"
                bc.apply(A)
#                print '#######LHS BC', i, "END"
            val.append(A)
        if not b is None:
            for i, bc in enumerate(bcs):
#                print '#######RHS BC', i, "START"
                bc.apply(b)
#                print '#######RHS BC', i, "END"
            val.append(b)
            
        if len(val) == 1:
            val = val[0]
        return val

    def assemble_lhs(self, lmbda, basis, uD=None, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        """Assemble the discrete operator."""
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(self.sigma(lmbda, self.mu, u), sym(nabla_grad(v))) * dx
        A = assemble(a)
        if withBC:
            A = self.apply_dirichlet_bc(V, A=A, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return A

    def assemble_rhs(self, f, basis, uD=None, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary, g=None, Neumann_boundary=None):
        """Assemble the discrete right-hand side."""
        # get FEniCS function space
        V = basis._fefs
        # define linear form
        v = TestFunction(V)
        l = inner(f, v) * dx
        
        # treat Neumann boundary
        if Neumann_boundary is not None:
            assert g is not None
            ds = Measure("ds")[Neumann_boundary]
            if not isinstance(Neumann_boundary, (tuple, list)):
                Neumann_boundary = [Neumann_boundary]
            if not isinstance(g, (tuple, list)):
                g = [g]
            for b in range(len(Neumann_boundary)):
                l -= dot(g, v) * ds(b)
                
        # assemble linear form
        F = assemble(l)
        # apply Dirichlet boundary conditions
        if withBC:
            F = self.apply_dirichlet_bc(V, b=F, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return F

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
        return lmbda * dot(self.sigma(lmbda, self.mu, v), nu)

    def r_Nb(self, lmbda, v, nu):
        """Neumann boundary residual."""
        pass
