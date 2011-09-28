r""" This is a private module that implements the basic recurrence
relations for orthgonal polynomials and methods for converting between
different formats for those recurrences


There are different forms for the format of the recurrence
coefficients. The one used in (Abramowitz & Stegun, p. 782) is

.. math:: a_{1n} p_{n + 1} = (a_{2n} + x a_{3n})p_n - a_{4n} p_{n - 1}

It is common to divide both sides by :math:`a_{1n}` in which case the 
following recurrence results

.. math:: p_{n + 1}(x) = (a_n + x b_n) p_n(x) - c_n p_{n - 1}(x)

where 

.. math::  a_n = a_{2n}/a_{1n}, b_n = a_{3n}/a_{1n}, c_n = a_{4n}/a_{1n}

.. note:: Note that in many publications the role of :math:`a_n` 
          and :math:`b_n` is interchanged.

The square of the norm :math:`h_n` of the polynomials is given by

.. math:: h_n = \int p_n ^ 2 w(x) \mathrm{d}x

where :math:`w` is the weight function for the family of
polynomials. Note: we require the weights to be given such that
:math:`h_0=1`, which is sensible for stochastics applications, but may
be at variance to common usage for the orthogonal
polynomials. According to (add citation) this can also be computed via
the recurrence coefficients

.. math:: h_n = \frac{b_0}{b_n} c_1 c_2 \cdots c_n

The normalised polynomials can be computed via

.. math:: \hat{p}_n = p_n / \sqrt{h_n}

For the normalised polynomials we need only two coefficients where we
stick to the definition in Gittelson (which is probably from Gautschi)

.. math:: \beta_n \hat{p}_n(x) = (x-\alpha_{n-1}) \hat{p}_{n-1}(x) - \beta_{n-1} \hat{p}_{n-2}(x)

or 

.. math:: \beta_{n+1} \hat{p}_{n+1}(x) = (x-\alpha_{n}) \hat{p}_{n}(x) - \beta_{n} \hat{p}_{n-1}(x)

from which the three term recurrence can be computed via

.. math:: 
    a_n &= -\alpha_n / \beta_{n+1}, \\
    b_n &= 1 / \beta_{n+1}, \\
    c_n &= \beta_n / \beta_{n+1}

(Note: it follows easily that for these recurrence coefficients :math:`h_n=1`)

Describe: what needs to be provided for a polynomial family and what
can be computed and how

(Abramowitz & Stegun, p. 774 - 775) AS: defs p. 773,
orth relations

inf = float('inf')
Legendre P_n(x), [ - 1, 1], w(x) = 1, 2 / (2n + 1), (n + 1, 0, 2n + 1, n)
Stoch. Hermite He_n(x), [ - inf, inf], w(x) = 1, (sqrt(2 * pi)n!), (1, 0, 1, n)
"""

import numpy as np
import scipy as sp



def rc4_to_rc3(rc4_func):
    def rc3_func(n):
        (a1, a2, a3, a4) = rc4_func(n)
        a1 = float(a1)
        return (a2 / a1, a3 / a1, a4 / a1)
    return rc3_func


def rc_orth_to_rc3(rc_orth_func):
    def rc3_func(n):
        print n
        (alpha0, beta0) = rc_orth_func(n)
        (alpha1, beta1) = rc_orth_func(n + 1)
        a = -float(alpha0) / float(beta1)
        b = 1.0 / float(beta1)
        c = float(beta0) / float(beta1)
        return (a, b, c)
    return rc3_func


def rc_stoch_hermite(n):
    """Return the recurrence coefficients for the stochastic Hermite
    polynomials.

    AS page 782

    Returns `(a2n, a3n, a4n)`
    """
    return (0.0, 1.0, float(n))


def sqnorm_stoch_hermite(n):
    """AS page 782 (s.t. h0 == 1)"""
    return sp.factorial(n)


@rc4_to_rc3
def rc_legendre(n):
    """AS page 782 """
    return (n + 1.0, 0.0, 2.0 * n + 1.0, float(n))


def rc_orth_legendre(n):
    """ """
    if n > 0:
        beta = 1.0 / np.sqrt(4.0 - 1.0 / float(n) ** 2)
        beta2 = (4 - float(n) ** - 2) ** - 0.5
        assert abs(beta - beta2) < 1e-8
    else:
        beta = 0.0
        #beta = 1.0
    return (0.0, beta)


def sqnorm_legendre(n):
    """AS page 782 (divided by 2, s.t. h0 == 1)"""
    return 1.0 / (2.0 * n + 1.0)


def sqnorm_from_rc(rc_func, n):
    """Compute norm from recurrence.

    Normalizing Orthogonal Polynomials by Using their Recurrence Coefficients
    Alan G. Law and M. B. Sledd
    Proceedings of the American Mathematical Society, Vol. 48, No. 2
    (Apr., 1975), pp. 505 - 507
    Stable URL: http: // www.jstor.org / stable / 2040291 .
    """
    (a0, b0, c0) = rc_func(0)
    (an, bn, cn) = rc_func(n)
    h = b0 / bn
    for i in range(1, n + 1):
        (a, b, c) = rc_func(i)
        h *= c
    return h


def compute_poly(rc_func, n, x):
    f = [0 * x, 0 * x + 1]
    for i in xrange(n):
        (a, b, c) = rc_func(float(i))
        fn = (a + x * b) * f[i + 1] - c * f[i]
        f.append(fn)
    return f[1:]

######################################################


def testit():
    x = np.poly1d([1, 0])
    rc3_orth_legendre = rc_orth_to_rc3(rc_orth_legendre)
    P1 = compute_poly(rc3_orth_legendre, 5, x)
    P2 = compute_poly(rc_legendre, 5, x)
    for i in range(5):
        print sqnorm_legendre(i), sqnorm_legendre(i) ** 0.5
        print P1[i].coeffs * sqnorm_legendre(i) ** 0.5
        print P2[i].coeffs
        print


def foo():
    P = compute_poly(rc_legendre, 5, x)
    z = compute_poly(rc_legendre, 5, 1.0)
    if False:
        for i in range(0 * 5 - 1):
            print P[i]
            print P[i](1.0)
            print sqnorm_legendre(i)
    for i in range(1, 5):
        print sqnorm_legendre(i)
        j = i
        h2 = sqnorm_legendre(j + 1) ** 0.5
        h1 = sqnorm_legendre(j) ** 0.5
        h0 = sqnorm_legendre(j - 1) ** 0.5
        t = rc_legendre(i)
        print (t[0] * h1 / h2, t[1] * h1 / h2, t[2] * h0 / h2)
        print (t[0] * h2 / h1, t[1] * h2 / h1, t[2] * h2 / h0)
        print rc_orth_legendre(i)
        print

#testit()
#raise ValueError()

######################################################
import numpy as np
from numpy.testing import *

from spuq.polynomials.polynomials import *


class TestRecurrences(TestCase):
    def test_legendre(self):
        x = np.poly1d([1, 0])
        P = compute_poly(rc_legendre, 5, x)
        assert_array_equal(P[0].coeffs, [1])
        assert_array_equal(P[1].coeffs, np.array([1, 0]))
        assert_array_equal(P[2].coeffs, np.array([3, 0, - 1]) / 2.0)
        assert_array_equal(P[3].coeffs, np.array([5, 0, - 3, 0]) / 2.0)
        assert_array_equal(P[4].coeffs, np.array([35, 0, - 30, 0, 3]) / 8.0)

    def test_stoch_hermite(self):
        x = np.poly1d([1, 0])
        P = compute_poly(rc_stoch_hermite, 5, x)
        assert_array_equal(P[0].coeffs, [1])
        assert_array_equal(P[1].coeffs, np.array([1, 0]))
        assert_array_equal(P[2].coeffs, np.array([1, 0, - 1]))
        assert_array_equal(P[3].coeffs, np.array([1, 0, - 3, 0]))
        assert_array_equal(P[4].coeffs, np.array([1, 0, - 6, 0, 3]))

    def test_sqnorm_from_rc(self):
        assert_array_almost_equal(sqnorm_legendre(0), 1)
        h1 = [sqnorm_from_rc(rc_legendre, i) for i in range(7)]
        h2 = [sqnorm_legendre(i) for i in range(7)]
        assert_array_almost_equal(h1, h2)

        assert_array_almost_equal(sqnorm_stoch_hermite(0), 1)
        h1 = [sqnorm_from_rc(rc_stoch_hermite, i) for i in range(7)]
        h2 = [sqnorm_stoch_hermite(i) for i in range(7)]
        assert_array_almost_equal(h1, h2)

    def test_rc3_to_orth(self):
        x = np.poly1d([1, 0])
        rc3_orth_legendre = rc_orth_to_rc3(rc_orth_legendre)
        P1 = compute_poly(rc3_orth_legendre, 4, x)
        P2 = compute_poly(rc_legendre, 4, x)
        hs = [sqnorm_legendre(i) ** 0.5 for i in range(5)]

        assert_array_almost_equal(P1[0] * hs[0], P2[0])
        assert_array_almost_equal(P1[1] * hs[1], P2[1])
        assert_array_almost_equal(P1[2] * hs[2], P2[2])
        assert_array_almost_equal(P1[3] * hs[3], P2[3])
        assert_array_almost_equal(P1[4] * hs[4], P2[4])


if __name__ == "__main__":
    run_module_suite()
