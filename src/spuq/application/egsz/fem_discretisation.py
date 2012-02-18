"""FEniCS FEM discretisation implementation for Poisson model problem"""

from dolfin import nabla_grad, TrialFunction, TestFunction, inner, assemble, dx, Constant, DirichletBC, Matrix
import numpy as np

from spuq.fem.fenics.fenics_vector import FEniCSBasis
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.linalg.operator import Operator
from spuq.utils.type_check import takes, anything

#        def apply(self, vec):
#            return self._a * vec
#        @property
#        def domain(self):
#            return self._basis
#        @property
#        def codomain(self):
#            return self._basis

class FEMPoisson(FEMDiscretisation):
    """FEM discrete Laplace operator with coefficient :math:`a` on domain :math:`\Omega:=[0,1]^2` with homogeneous Dirichlet boundary conditions

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    def assemble_operator(self, coeff, basis):
        """Assemble the discrete problem (i.e. the stiffness matrix)"""
        
        class MatrixWrapper(Operator):
            @takes(anything, Matrix, FEniCSBasis)
            def __init__(self, matrix):
                self._matrix = matrix
                self._basis = basis
            @takes(anything, np.array)
            def apply(self, vec):
                new_vec = vec.copy()
                new_vec.coeffs = self._matrix * vec.coeffs
                return new_vec
            @property
            def domain(self):
                return self._basis
            @property
            def codomain(self):
                return self._basis
        
        # get FEniCS function space
        V = basis._fefs

# TODO: what about boundary conditions?
#        # define boundary conditions
#        def u0_boundary(x, on_boundary):
#            return on_boundary
#        u0 = Constant(0.0)        
#        bc = DirichletBC(V, u0, u0_boundary)

        # setup problem
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
        A = assemble(a)
        return MatrixWrapper(A)
