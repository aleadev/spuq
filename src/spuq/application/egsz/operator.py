"""EGSZ discrete operator A


According to the representation of orthogonal polynomials in spuq, the operator A defined in EGSZ (2.6) has the more general form

.. math:: A(u_N) := \overline{A}u_{N,\mu} + \sum_{m=1}^infty A_m\left(\alpha^m_{\mu_m+1}u_{N,\mu+e_m} - \alpha^m_{\mu_m}u_{N,\mu} + \alpha^m_{\mu_m-1}u_{N,\mu-e_m}\right)

where the coefficients :math:`(\alpha^m_{n-1},\alpha^m_n,\alpha^m_{n+1})` are obtained from the the three recurrence coefficients :math:`(a_n,b_n,c_n)` of the orthogonal polynomial by

.. math::
        \alpha_{n-1} &:= c_n/b_n \\
        \alpha_n &:= a_n/b_n \\
        \alpha_{n+1} &:= 1/b_n
"""

from spuq.linalg.operator import Operator
from spuq.utils.type_check import *
from spuq.application.egsz.coefficient_field import CoefficientField 
from spuq.fem.multi_vector import MultiVector
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.stochastics.random_variable import RandomVariable

class MultiOperator(Operator):
    """Discrete operator according to EGSZ (2.6), generalised for spuq orthonormal polynomials"""
    
    @takes(FEMDiscretisation, CoefficientField)
    def __init__(self, FEM, CF, maxm=10, pt=FEniCSBasis.PROJECTION.INTERPOLATION):
        """Initialise discrete operator with FEM discretisation and coefficient field of the diffusion coefficient"""
        self._FEM = FEM
        self._CF = CF
        self.maxm = ...
            
    @takes(RandomVariable, MultiVector, int)
    def apply(self, w):
        "Apply operator to vec which should be in the same domain"
        
        v = MultiVector()           # result vector
        Delta = w.active_set()
        for mu in Delta:
            # deterministic part
            am_f, am_rf = self._CF[0]
            A0 = self._FEM.assemble_operator( {'a':am_f}, w[mu].basis )
            v[mu] = A0 * w[mu] 
            for m in range(1, maxm):
                # assemble A for \mu and a_m
                am_f, am_rv = self._CF[m]
                Am = self._FEM.assemble_operator( {'a':am_f}, w[mu].basis )

                # prepare polynom coefficients
                p = am_rv.orth_poly
                (a, b, c) = p.recurrence_coefficients(mu[m])
                beta = (a/b, 1/b, c/b)

                # mu
                wN = -beta[0]*w[mu]

                # mu+1
                mu1 = mu.add( (m,1) )
                if mu1 in Delta:
                    wN += beta[1] * w[mu].functionspace.project(w[mu1], ptype=pt)

                # mu-1
                mu2 = mu.add( (m,-1) )
                if mu2 in Delta:
                    wN += beta[-1] * w[mu].functionspace.project(w[mu2], ptype=pt)

                # apply discrete operator
                v[mu] = Am * wN
        return v
        
    def domain(self):
        "Returns the basis of the domain"
        # TODO
        return None

    def codomain(self):
        "Returns the basis of the codomain"
        # TODO
        return None
