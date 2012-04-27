"""EGSZ discrete operator A.

According to the representation of orthogonal polynomials in spuq, the operator A defined in EGSZ (2.6) has the more general form

.. math:: A(u_N) := \overline{A}u_{N,\mu} + \sum_{m=1}^infty A_m\left(\alpha^m_{\mu_m+1}u_{N,\mu+e_m} - \alpha^m_{\mu_m}u_{N,\mu} + \alpha^m_{\mu_m-1}u_{N,\mu-e_m}\right)

where the coefficients :math:`(\alpha^m_{n-1},\alpha^m_n,\alpha^m_{n+1})` are obtained from the the three recurrence coefficients :math:`(a_n,b_n,c_n)` of the orthogonal polynomial by

.. math::
        \alpha_{n-1} &:= c_n/b_n \\
        \alpha_n &:= a_n/b_n \\
        \alpha_{n+1} &:= 1/b_n
"""

from spuq.linalg.basis import Basis
from spuq.linalg.operator import Operator
from spuq.utils.type_check import takes, anything, optional
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.application.egsz.multi_vector import MultiVector, MultiVectorWithProjection

import logging

logger = logging.getLogger(__name__)

class MultiOperator(Operator):
    """Discrete operator according to EGSZ (2.6), generalised for spuq orthonormal polynomials."""

    @takes(anything, CoefficientField, callable, optional(Basis), optional(Basis))
    def __init__(self, coeff_field, assemble, domain=None, codomain=None):
        """Initialise discrete operator with FEM discretisation and
        coefficient field of the diffusion coefficient"""
        self._assemble = assemble
        self._coeff_field = coeff_field
        self._domain = domain
        self._codomain = codomain

    @takes(any, MultiVectorWithProjection)
    def apply(self, w):
        """Apply operator to vector which has to live in the same domain."""

        v = 0 * w
        Delta = w.active_indices()
        maxm = w.max_order
        if len(self._coeff_field) < maxm:
            logger.warning("insufficient length of coefficient field for MultiVector (%i instead of %i",
                len(self._coeff_field),
                maxm)
            maxm = len(self._coeff_field)
            #        assert self._coeff_field.length >= maxm        # ensure coeff_field expansion is sufficiently long
        for mu in Delta:
            logger.debug("apply on mu = %s", str(mu))
            # deterministic part
            a0_f = self._coeff_field.mean_func
            A0 = self._assemble(a0_f, w[mu].basis)
            v[mu] = A0 * w[mu]
            for m in range(maxm):
                logger.debug("with m = %i", m)
                # assemble A for \mu and a_m
                am_f, am_rv = self._coeff_field[m]
                Am = self._assemble(am_f, w[mu].basis)

                # prepare polynom coefficients
                beta = am_rv.orth_polys.get_beta(mu[m])

                # mu
                cur_w = -beta[0] * w[mu]

                # mu+1
                mu1 = mu.inc(m)
                if mu1 in Delta:
                    cur_w += beta[1] * w.get_projection(mu1, mu)

                # mu-1
                mu2 = mu.dec(m)
                if mu2 in Delta:
                    cur_w += beta[-1] * w.get_projection(mu2, mu)

                # apply discrete operator
                v[mu] += Am * cur_w
        return v

    @property
    def domain(self):
        """Returns the basis of the domain."""
        return self._domain

    @property
    def codomain(self):
        """Returns the basis of the codomain."""
        return self._codomain


class PreconditioningOperator(Operator):
    """Preconditioning operator according to EGSZ section 7.1."""

    @takes(anything, anything, callable, optional(Basis), optional(Basis))
    def __init__(self, mean_func, assemble_solver, domain=None, codomain=None):
        """Initialise operator with FEM discretisation and
        mean diffusion coefficient"""
        self._assemble_solver = assemble_solver
        self._mean_func = mean_func
        self._domain = domain
        self._codomain = codomain

    @takes(any, MultiVector)
    def apply(self, w):
        """Apply operator to vector which has to live in the same domain."""
        v = 0 * w
        Delta = w.active_indices()

        for mu in Delta:
            a0_f = self._mean_func
            A0 = self._assemble_solver(a0_f, w[mu].basis)
            v[mu] = A0 * w[mu]
        return v

    @property
    def domain(self):
        """Returns the basis of the domain."""
        return self._domain

    @property
    def codomain(self):
        """Returns the basis of the codomain."""
        return self._codomain
