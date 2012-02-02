"""EGSZ discrete operator A


According to the representation of orthogonal polynomials in spuq, the operator A defined in EGSZ (2.6) has the more general form

.. math:: A(u_N) := \overline{A}u_{N,\mu} + \sum_{m=1}^infty A_m\left(\alpha^m_{\mu_m+1}u_{N,\mu+e_m} - \alpha^m_{\mu_m}u_{N,\mu} + \alpha^m_{\mu_m-1}u_{N,\mu-e_m}\right)

where the coefficients :math:`(\alpha^m_{n-1},\alpha^m_n,\alpha^m_{n+1})` are obtained from the the three recurrence coefficients :math:`(a_n,b_n,c_n)` of the orthogonal polynomial by

.. math::
        \alpha_{n-1} &:= c_n/b_n \\
        \alpha_n &:= a_n/b_n \\
        \alpha_{n+1} &:= 1/b_n


# init
FEM = FEMPoisson()
mo = MultiOperator( CF, FEM.assemble_operator )


# ....
ptype = FEniCSBasis.PROJECTION.INTERPOLATION
if wN_projection_cache:
    assert wN==None or wN==wN_projection_cache.wN
    wN = wN_projection_cache.wN
    wN_cache = wN_projection_cache
else:
    wN_cache = ProjectionCache(wN, ptype=ptype)
        
mo.set_project( lambda mu1: wN_cache[mu1, mu, False])
mo.apply(self, wN)

"""

from spuq.linalg.basis import Basis
from spuq.linalg.operator import Operator
from spuq.utils.type_check import takes, anything, optional
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVectorWithProjection


class MultiOperator(Operator):
    """Discrete operator according to EGSZ (2.6), generalised for spuq
    orthonormal polynomials"""

    @takes(anything, CoefficientField, callable, optional(Basis), optional(Basis))
    def __init__(self, CF, assemble, domain=None, codomain=None):
        """Initialise discrete operator with FEM discretisation and
        coefficient field of the diffusion coefficient"""
        self._assemble = assemble
        self._CF = CF

    @takes(any, MultiVectorWithProjection)
    def apply(self, wN):
        """Apply operator to vector which has to live in the same
        domain"""

        v = 0 * wN
        Delta = wN.active_indices()
        for mu in Delta:
            # deterministic part
            a0_f, _ = self._CF[0]
            A0 = self._assemble({'a':a0_f}, wN[mu].basis)
            v[mu] = A0 * wN[mu]
            for m in range(1, len(mu)):
                # assemble A for \mu and a_m
                am_f, am_rv = self._CF[m]
                Am = self._assemble({'a':am_f}, wN[mu].basis)

                # prepare polynom coefficients
                p = am_rv.orth_polys
                (a, b, c) = p.recurrence_coefficients(mu[m])
                beta = (a / b, 1 / b, c / b)

                # mu
                cur_wN = -beta[0] * wN[mu]

                # mu+1
                mu1 = mu.add(m, 1)
                if mu1 in Delta:
                    cur_wN += beta[1] * wN.get_projection(mu1, mu)

                # mu-1
                mu2 = mu.add(m, -1)
                if mu2 in Delta:
                    cur_wN += beta[-1] * wN.get_projection(mu2, mu)

                # apply discrete operator
                v[mu] = Am * cur_wN
        return v

    def domain(self):
        """Returns the basis of the domain"""
        # TODO: this could be extracted from the discrete domains
        # (meshes) of the vectors or provided by the user
        raise NotImplementedError

    def codomain(self):
        """Returns the basis of the codomain"""
        # TODO
        raise NotImplementedError



