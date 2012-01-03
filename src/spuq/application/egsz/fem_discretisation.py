"""FEM discretisation implementation"""

from dolfin import *
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fenics.fenics_function import FEniCSFunction, FEniCSExpression

class FEMPoisson(FEMDiscretisation):
    """FEM discrete Laplace operator with coefficient :math:`a`

        ..math:: -\mathrm{div}a \nabla u

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    def assemble_operator(self, data, basis):
        """Assemble discrete Poisson operator, i.e., the stiffness matrix"""

        if isinstance(basis, FEniCSBasis):
            u = TrialFunction(basis.functionspace)
            v = TestFunction(basis.functionspace)
        else:
            u = TrialFunction(basis)
            v = TestFunction(basis)
        a = None
        try:
            c = data['a']
            if isinstance(c, FEniCSExpression):
                c = c.fex
            elif isinstance(c, FEniCSFunction):
                c = c.f
        except Exception:
            print "FEMPoisson: no coefficient data given, assuming c=1"
            a = inner(grad(u),grad(v))*dx
        else:
            a = c*inner(grad(u),grad(v))*dx
        A = assemble(a)
        return A
