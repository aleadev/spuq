""" """

from dolfin import *
from spuq.fem.fem_discretisation import FEMDiscretisation

class FEMPoisson(FEMDiscretisation):

        def assemble_operator(self, data, basis):
            """Assemble discrete Poisson operator, i.e., the stiffness matrix"""
            u = TrialFunction(basis.functionspace)
            v = TestFunction(basis.functionspace)
            a = None
            try:
                c = data['a']
            except Exception:
                print "FEMPoisson: no coefficient data given, assuming c=1"
                a = inner(grad(u)*grad(v))*dx
            else:
                a = c*inner(grad(u)*grad(v))*dx
            A = assemble(a)
            return A
