"""FEniCS FEM discretisation implementation for Poisson model problem"""

from dolfin import (nabla_grad, TrialFunction, TestFunction,
                    inner, assemble, dx, Constant, DirichletBC)

from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.utils.type_check import takes, anything

class FEMPoisson(FEMDiscretisation):
    """FEM discrete Laplace operator with coefficient :math:`a` on domain :math:`\Omega:=[0,1]^2` with homogeneous Dirichlet boundary conditions.

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    @classmethod
    def assemble_operator(cls, coeff, basis, withBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = cls.assemble_lhs(coeff, basis, withBC)
        return FEniCSOperator(matrix, basis)

    @classmethod
    def assemble_solve_operator(cls, coeff, basis, withBC=True):
        matrix = cls.assemble_lhs(coeff, basis, withBC)
        return FEniCSSolveOperator(matrix, basis)

    @classmethod
    def apply_dirichlet_bc(cls, V, A=None, b=None, uD=None):
        """Apply Dirichlet boundary conditions."""
        # define boundary conditions
        def uD_boundary(x, on_boundary):
            return on_boundary
        if uD is None:
            uD = Constant(0.0)
        try:
            V = V._fefs
        except:
            pass
        bc = DirichletBC(V, uD, uD_boundary)
        val = []
        if not A is None:
            bc.apply(A)
            val.append(A)
        if not b is None:
            bc.apply(b)
            val.append(b)
        if len(val) == 1:
            val = val[0]
        return val

    @classmethod
    def assemble_lhs(cls, coeff, basis, uD=None, withBC=True):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        if withBC:
            A = cls.apply_dirichlet_bc(V, A=A, uD=uD)
        return A

    @classmethod
    def assemble_rhs(cls, f, basis, uD=None, withBC=True):
        """Assemble the discrete right-hand side."""
        # get FEniCS function space
        V = basis._fefs
        # assemble and apply boundary conditions
        v = TestFunction(V)
        l = (f * v) * dx
        F = assemble(l)
        if withBC:
            F = cls.apply_dirichlet_bc(V, b=F, uD=uD)
        return F
