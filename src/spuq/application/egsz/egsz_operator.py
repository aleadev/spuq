"""EGSZ discrete operator A"""

from spuq.linalg.operator import Operator
from spuq.utils.type_check import *
from spuq.application.egsz.egsz_coefficient_field import CoefficientField 
from spuq.fem.multi_vector import MultiVector
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.fem.fenics.fenics_basis import FEniCSBasis

class MultiOperator(Operator):
    """Discrete operator according to EGSZ (2.6)"""
    
    @takes(FEMDiscretisation, CoefficientField)
    def __init__(self, FEM, CF):
        """Initialise discrete operator with FEM discretisation and coefficient field of the diffusion coefficient"""
        self._FEM = FEM
        self._CF = CF
            
    @takes(MultiVector)
    def apply(self, w, maxm=10, pt=FEniCSBasis.PROJECTION.INTERPOLATION):
        "Apply operator to vec which should be in the same domain"
        
        v = MultiVector()
        Delta = w.active_set()
        for mu in Delta:
            A0 = self._FEM.assemble_operator( {'a':self._CF[0][0]}, w[mu].basis )
            v[mu] = A0 * w[mu] 
            for m in xrange(1,maxm):
                Am = self._FEM.assemble_operator( {'a':self._CF[m][0]}, w[mu].basis )
                mu1 = mu.add( (m,1) )
                if mu1 in Delta:
#                    v[mu] += Am * beta(m, mu[m] + 1) * w[mu1].project(w[mu].mesh)
                    a, b, c = self._CF[m][1].recurrence_coefficients(mu[m] + 1)
                    # TODO!
                    v[mu] += Am * a*b*c * w[mu].functionspace.project(w[mu1], ptype=pt)
                mu2 = mu.add( (m,-1) )
                if mu2 in Delta:
#                    v[mu] += Am * beta(m, mu[m]) * w[mu2].project(w[mu].mesh)
                    a, b, c = self._CF[m][1].recurrence_coefficients(mu[m])
                    # TODO!
                    v[mu] += Am * a*b*c * w[mu].functionspace.project(w[mu2], ptype=pt)
        return v
        
    def domain(self):
        "Returns the basis of the domain"
        # TODO
        return None

    def codomain(self):
        "Returns the basis of the codomain"
        # TODO
        return None
