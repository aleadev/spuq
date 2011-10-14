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

.. math::

   \beta_n \hat{p}_n(x) = (x-\alpha_{n-1}) \hat{p}_{n-1}(x) -
   \beta_{n-1} \hat{p}_{n-2}(x)

or

.. math::

   \beta_{n+1} \hat{p}_{n+1}(x) = (x-\alpha_{n}) \hat{p}_{n}(x) -
   \beta_{n} \hat{p}_{n-1}(x)

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

_0 = lambda x: 0 * x
_1 = lambda x: 0 * x + 1


def rc4_to_rc3(a):
    """Convert recurrence coeffs from a 4 to a 3 coeffs form."""
    a0 = float(a[0])
    return (a[1] / a0, a[2] / a0, a[3] / a0)


def rc_norm_to_rc3(rc_norm_func):
    def rc3_func(n):
        (alpha0, beta0) = rc_norm_func(n)
        (alpha1, beta1) = rc_norm_func(n + 1)
        a = -float(alpha0) / float(beta1)
        b = 1.0 / float(beta1)
        c = float(beta0) / float(beta1)
        return (a, b, c)
    return rc3_func


def normalise_rc(rc_func, sqnorm_func=None):
    """Return a function that returns the recurrence coefficients for
    the normalised polynomials."""
    if sqnorm_func is None:
        sqnorm_func = lambda n: sqnorm_from_rc(rc_func, n)

    def rc_norm_func(n):
        (a, b, c) = rc_func(n)
        h2 = sqnorm_func(n + 1) ** 0.5
        h1 = sqnorm_func(n) ** 0.5
        if n > 0:
            h0 = sqnorm_func(n - 1) ** 0.5
        else:
            h0 = 0
        return (a * h1 / h2, b * h1 / h2, c * h0 / h2)
    return rc_norm_func


def sqnorm_from_rc(rc_func, n):
    """Compute norm from recurrence.

    Normalizing Orthogonal Polynomials by Using their Recurrence Coefficients
    Alan G. Law and M. B. Sledd
    Proceedings of the American Mathematical Society, Vol. 48, No. 2
    (Apr., 1975), pp. 505 - 507
    Stable URL: http: // www.jstor.org / stable / 2040291 .
    """
    assert type(n) == type(1)

    (a0, b0, c0) = rc_func(0)
    (an, bn, cn) = rc_func(n)
    h = b0 / bn
    for i in range(1, n + 1):
        (a, b, c) = rc_func(i)
        h *= c
    return h

def rc_shift_scale(rc_func, shift, scale):
    """Return a function that computes recurrence coefficients for
    polynomials with a weight function that's shifted and scaled.

    I.e. w'(x) = w((x-shift)/scale) where w is the original weight
    function.
    """
    def rc_shifted_scaled(n):
        (a, b, c) = rc_func(n)
        return (a - b * shift / scale, b / scale, c)
    return rc_shifted_scaled


def rc_window_trans(rc_func, old_domain, new_domain):
    (a0, b0) = old_domain
    (a1, b1) = new_domain
    shift = 0.5 * ((a1 + b1) - (a0 + b0))
    scale = float(a1 - b1) / (a0 - b0)
    return rc_shift_scale(rc_func, shift, scale)

def compute_poly(rc_func, n, x):
    f = [_0(x), _1(x)]
    for i in xrange(n):
        (a, b, c) = rc_func(i)
        fn = (a + x * b) * f[i + 1] - c * f[i]
        f.append(fn)
    return f[1:]


def _compute_poly2(rc_func, n, x): # pragma: no cover
    if n == 0:
        # ensure return value has the right format (e.g. if x is a
        # vector, ndarray or poly1d)
        return _1(x)
    else:
        h1, h0 = 1, 0
        for i in xrange(0, n):
            (a, b, c) = rc_func(i)
            h1, h0 = (a + b * x) * h1 - c * h0, h1
        return h1


def eval_clenshaw(rc_func, coeffs, x):
    """Evaluate the polynomial using Clenshaw's algorithm"""
    q1 = q2 = _0(x)
    n = len(coeffs) - 1
    for k in reversed(xrange(n + 1)):
        (ak, bk, ck) = rc_func(k)
        (ak2, bk2, ck2) = rc_func(k + 1)
        q0 = coeffs[k] + (ak + bk * x) * q1 - ck2 * q2
        q1, q2 = q0, q1
    return q0


def eval_forsythe(rc_func, coeffs, x):
    """Evaluate the polynomial using Forsythe's algorithm"""
    n = len(coeffs) - 1

    t0 = _1(x)
    f0 = coeffs[0]
    if n < 1:
        return f0
    (a0, b0, c0) = rc_func(0)
    t1 = a0 + b0 * x
    f1 = f2 = f0 + coeffs[1] * t1

    for k in xrange(1, n):
        (ak, bk, ck) = rc_func(k)
        t2 = (ak + bk * x) * t1 - ck * t0
        f2 = f1 + coeffs[k + 1] * t2
        t0, t1, f1 = t1, t2, f2
    return f2


# Monomials
def rc_monomials(n):
    return (0.0, 1.0, 0.0)


# Stochatic Hermite polynomials
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


def stc_stoch_hermite(a, b, c):
    n = max((a, b, c))
    s = a + b + c
    if bool(s % 2) or a <= b + c or b <= a + c or c <= a + b:
        c = 0
    else:
        s /= 2
        fac = scipy.factorial
        c = (fac(s - a) * fac(s - b) * fac(s - c) /
             (fac(a) * fac(b) * fac(c)))


# Legendre polynomials
def rc_legendre(n):
    """AS page 782 """
    return rc4_to_rc3((max(0, n) + 1.0, 0.0, 2.0 * n + 1.0, float(n)))


def rc_norm_legendre(n):
    """Recurrence coefficients of the normalised Legendre polys on [-1, 1]."""
    if n > 0:
        beta = (4 - float(n) ** -2) ** -0.5
    else:
        beta = 0.0
    return (0.0, beta)


def sqnorm_legendre(n):
    """Square of the norm of the legendre polynomials of [-1, 1].

    AS page 782 (divided by 2, s.t. h0 == 1)"""
    return 1.0 / (2.0 * n + 1.0)
