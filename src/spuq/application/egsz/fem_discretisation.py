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
    def assemble_operator(cls, coeff, basis):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = cls.assemble_lhs(coeff, basis)
        return FEniCSOperator(matrix, basis)

    @classmethod
    def assemble_solve_operator(cls, coeff, basis):
        matrix = cls.assemble_lhs(coeff, basis)
        return FEniCSSolveOperator(matrix, basis)

    @classmethod
    def assemble_lhs(cls, coeff, basis, uD=None):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs

        # define boundary conditions
        def uD_boundary(x, on_boundary):
            return on_boundary
        if uD is None:
            uD = Constant(0.0)
        bc = DirichletBC(V, uD, uD_boundary)

        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        bc.apply(A)
        return A

    @classmethod
    def assemble_rhs(cls, f, basis, uD=None):
        """Assemble the discrete right-hand side."""

        # get FEniCS function space
        V = basis._fefs

        # define boundary conditions
        def uD_boundary(x, on_boundary):
            return on_boundary
        if uD is None:
            uD = Constant(0.0)
        bc = DirichletBC(V, uD, uD_boundary)

        # assemble and apply boundary conditions
        v = TestFunction(V)
        l = (f * v) * dx
        F = assemble(l)
        bc.apply(F)
        return F
