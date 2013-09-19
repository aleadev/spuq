"""Preconditioned conjugate gradient method for the EGSZ application"""

from spuq.utils.type_check import takes, optional, returns, tuple_of

from spuq.linalg.operator import Operator
from spuq.linalg.vector import Vector, inner, Flat
from spuq.utils.forgetful_vector import ForgetfulVector

__all__ = ["pcg"]

import logging
logger = logging.getLogger(__name__)

def _identity(x):
    return x

@takes(Operator, Vector, Operator, Vector, optional(float), optional(int))
def pcg(A, b, P, x0, eps=1e-4, maxiter=100, truncate_func=_identity):

    in_tensor_format = not isinstance(b, Flat)

    # for most quantities in PCG (except rho) only the most recent
    # values needs to be kept in memory
    x = ForgetfulVector(1)
    r = ForgetfulVector(1)
    q = ForgetfulVector(1)
    z = ForgetfulVector(1)
    p = ForgetfulVector(1)

    qp = ForgetfulVector(1)
    rho = ForgetfulVector(2)
    alpha = ForgetfulVector(2)
    beta = ForgetfulVector(2)

    x[0] = x0
    r[0] = b - A * x[0]
    r[0] = truncate_func(r[0])
    z[0] = P * r[0]
    p[0] = z[0]
    rho[0] = inner(r[0], z[0])

    for i in xrange(1, maxiter):
        logger.info("pcg iter: %s -> rho=%s, r^2=%s" % (i, rho[i - 1], inner(r[i - 1], r[i - 1])))
        if rho[i - 1] < 0:
            for mu in r[i - 1].active_indices():
                print i, mu, inner(r[i - 1][mu], z[i - 1][mu])
            raise Exception("Preconditioner for PCG is not positive definite (%s)" % rho[i - 1])
        if rho[i - 1] <= eps ** 2:
            return (x[i - 1], rho[i - 1], i)

        q[i - 1] = A * p[i - 1]
        q[i - 1] = truncate_func(q[i - 1])

        qp[i - 1] = inner(q[i - 1], p[i - 1])
        if qp[i - 1] == 0:
            raise Exception("Matrix for PCG is singular (%s)" % qp[i - 1])
        elif qp[i - 1] < 0:
            raise Exception("Matrix for PCG is not positive definite (%s)" % qp[i - 1])

        alpha[i - 1] = rho[i - 1] / qp[i - 1]
        x[i] = x[i - 1] + alpha[i - 1] * p[i - 1]
        x[i] = truncate_func(x[i])

        if in_tensor_format:
            r[i] = b - A * x[i]
            r[i] = truncate_func(r[i])
        else:
            r[i] = r[i - 1] - alpha[i - 1] * q[i - 1]

        z[i] = P * r[i]
        z[i] = truncate_func(z[i])

        rho[i] = inner(r[i], z[i])
        beta[i] = rho[i] / rho[i - 1]

        p[i] = z[i] + beta[i] * p[i - 1]
        p[i] = truncate_func(p[i])

        if False and in_tensor_format:
            print x[i].rank
            print z[i].rank
            print r[i].rank
            print p[i].rank
            print q[i-1].rank



    raise Exception("PCG did not converge")
