"""Preconditioned conjugate gradient method for the EGSZ application"""

from spuq.utils.type_check import takes, optional, returns, tuple_of

from spuq.linalg.operator import Operator
from spuq.linalg.vector import Vector, inner
from spuq.utils.forgetful_vector import ForgetfulVector

__all__ = ["pcg"]

import logging
logger = logging.getLogger(__name__)

@takes(Operator, Vector, Operator, Vector, optional(float), optional(int))
def pcg(A, f, P, w0, eps=1e-4, maxiter=100):
    # for most quantities in PCG (except zeta) only the most recent 
    # values needs to be kept in memory
    w = ForgetfulVector(1)
    rho = ForgetfulVector(1)
    s = ForgetfulVector(1)
    v = ForgetfulVector(1)
    z = ForgetfulVector(1)
    alpha = ForgetfulVector(1)
    zeta = ForgetfulVector(2)

    w[0] = w0
    rho[0] = f - A * w[0]
    s[0] = P * rho[0]
    v[0] = s[0]
    zeta[0] = inner(rho[0], s[0])
    for i in xrange(1, maxiter):
        logger.info("pcg iter: %s -> zeta=%s, rho^2=%s" % (i, zeta[i - 1], inner(rho[i - 1], rho[i - 1])))
        if zeta[i - 1] < 0:
            for mu in rho[i-1].active_indices():
                print i, mu, inner(rho[i-1][mu], s[i-1][mu])
            raise Exception("Preconditioner for PCG is not positive definite (%s)" % zeta[i - 1])
        if zeta[i - 1] <= eps ** 2:
            return (w[i - 1], zeta[i - 1], i)
        z[i - 1] = A * v[i - 1]
        alpha[i - 1] = inner(z[i - 1], v[i - 1])
        w[i] = w[i - 1] + zeta[i - 1] / alpha[i - 1] * v[i - 1]
        rho[i] = rho[i - 1] - zeta[i - 1] / alpha[i - 1] * z[i - 1]
        s[i] = P * rho[i]
        zeta[i] = inner(rho[i], s[i])
        v[i] = s[i] + zeta[i] / zeta[i - 1] * v[i - 1]

    raise Exception("PCG did not converge")


