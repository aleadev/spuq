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
from spuq.application.egsz.projection_cache import ProjectionCache 
from spuq.fem.multi_vector import MultiVector
from spuq.fem.fem_discretisation import FEMDiscretisation
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.stochastics.random_variable import RandomVariable
from numpy import dot

class MultiOperator(Operator):
    """Discrete operator according to EGSZ (2.6), generalised for spuq orthonormal polynomials"""
    
    @takes(any, FEMDiscretisation, CoefficientField, int)
    def __init__(self, FEM, CF, maxm=10, domain=None, codomain=None):
        """Initialise discrete operator with FEM discretisation and coefficient field of the diffusion coefficient"""
        self._FEM = FEM
        self._CF = CF
        self._maxm = maxm

    @takes(any, MultiVector, ProjectionCache)
    def apply(self, wN=None, wN_projection_cache=None, ptype=FEniCSBasis.PROJECTION.INTERPOLATION):
        """Apply operator to vector which has to live in the same domain"""

        if wN_projection_cache:
            assert wN==None or wN==wN_projection_cache.wN
            wN = wN_projection_cache.wN
            wN_cache = wN_projection_cache
        else:
            wN_cache = ProjectionCache(wN, ptype=ptype)
        
        v = MultiVector()           # result vector
        Delta = wN.active_set()
        for mu in Delta:
            # deterministic part
            a0_f, _ = self._CF[0]
            A0 = self._FEM.assemble_operator( {'a':a0_f}, wN[mu].basis )
            v[mu] = A0 * wN[mu] 
            for m in range(1, self._maxm):
                # assemble A for \mu and a_m
                am_f, am_rv = self._CF[m]
                Am = self._FEM.assemble_operator( {'a':am_f}, wN[mu].basis )

                # prepare polynom coefficients
                p = am_rv.orth_poly
                (a, b, c) = p.recurrence_coefficients(mu[m])
                beta = (a/b, 1/b, c/b)

                # mu
                cur_wN = -beta[0]*wN[mu]

                # mu+1
                mu1 = mu.add( (m,1) )
                if mu1 in Delta:
                    cur_wN += beta[1] * wN_cache[mu1, mu, False]

                # mu-1
                mu2 = mu.add( (m,-1) )
                if mu2 in Delta:
                    cur_wN += beta[-1] * wN_cache[mu2, mu, False]

                # apply discrete operator
                v[mu] = dot(Am, cur_wN)
        return v
        
    def domain(self):
        """Returns the basis of the domain"""
        # TODO: this could be extracted from the discrete domains (meshes) of the vectors or provided by the user
        return NotImplemented

    def codomain(self):
        """Returns the basis of the codomain"""
        # TODO
        return NotImplemented
