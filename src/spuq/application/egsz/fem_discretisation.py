"""FEniCS FEM discretisation implementation for Poisson model problem"""

from dolfin import (nabla_grad, TrialFunction, TestFunction,
                    inner, assemble, dx, Constant, DirichletBC)

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

    @classmethod
    def assemble_operator(cls, coeff, basis, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        """Assemble the discrete problem (i.e. the stiffness matrix) and return as Operator."""
        matrix = cls.assemble_lhs(coeff, basis, uD=None, withBC=withBC, Dirichlet_boundary=Dirichlet_boundary)
        return FEniCSOperator(matrix, basis)

    @classmethod
    def assemble_solve_operator(cls, coeff, basis, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        matrix = cls.assemble_lhs(coeff, basis, uD=None, withBC=withBC, Dirichlet_boundary=Dirichlet_boundary)
        return FEniCSSolveOperator(matrix, basis)

    @classmethod
    def apply_dirichlet_bc(cls, V, A=None, b=None, uD=None, Dirichlet_boundary=default_Dirichlet_boundary):
        """Apply Dirichlet boundary conditions."""
        if uD is None:
            uD = Constant(0.0)
        try:
            V = V._fefs
        except:
            pass
        bc = DirichletBC(V, uD, Dirichlet_boundary)
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
    def assemble_lhs(cls, coeff, basis, uD=None, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        """Assemble the discrete problem (i.e. the stiffness matrix)."""
        # get FEniCS function space
        V = basis._fefs
        # setup problem, assemble and apply boundary conditions
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        if withBC:
            A = cls.apply_dirichlet_bc(V, A=A, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return A

    @classmethod
    def assemble_rhs(cls, f, basis, uD=None, withBC=True, Dirichlet_boundary=default_Dirichlet_boundary):
        """Assemble the discrete right-hand side."""
        # get FEniCS function space
        V = basis._fefs
        # assemble and apply boundary conditions
        v = TestFunction(V)
        l = (f * v) * dx
        F = assemble(l)
        if withBC:
            F = cls.apply_dirichlet_bc(V, b=F, uD=uD, Dirichlet_boundary=Dirichlet_boundary)
        return F
