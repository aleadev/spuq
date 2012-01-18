from __future__ import division
from spuq.linalg.function import SimpleFunction, ConstFunction
from spuq.utils.testing import *
from numpy import log

f = SimpleFunction(f=lambda x: x ** 2, Df=lambda x: 2 * x)
g = SimpleFunction(f=lambda x: 2 * x, Df=lambda x: 2)
h = SimpleFunction(f=lambda x, y: 2 * x + 3 * y ** 3, domain_dim=2)
c3 = ConstFunction(3)

def test_function_operators():
    assert_equal(f(3), 9)
    assert_equal(g(3), 6)
    assert_equal(c3(7), 3)
    assert_equal(h(2, 3), 85)
    assert_raises(Exception, f, 3, 3)
    assert_raises(Exception, c3, 3, 3)
    assert_raises(Exception, h, 1)
    assert_raises(Exception, h, 1, 2, 3)

    assert_equal(f.D(3), 6)
    assert_equal(g.D(3), 2)
    assert_equal(c3.D(3), 0)


def test_function_add_and_sub():
    assert_equal((+f)(3), f(3))
    assert_equal((f + g)(3), f(3) + g(3))
    assert_equal((f + 2)(3), f(3) + 2)
    assert_equal((2 + f)(3), 2 + f(3))

    assert_equal((-f)(3), -f(3))
    assert_equal((f - g)(3), f(3) - g(3))
    assert_equal((f - 2)(3), f(3) - 2)
    assert_equal((2 - f)(3), 2 - f(3))


def test_derivative_add_and_sub():
    assert_equal((f + g).D(3), f.D(3) + g.D(3))
    assert_equal((f + 2).D(3), f.D(3))
    assert_equal((2 + f).D(3), f.D(3))

    assert_equal((f - g).D(3), f.D(3) - g.D(3))
    assert_equal((f - 2).D(3), f.D(3))
    assert_equal((2 - f).D(3), -f.D(3))


def test_function_mul_and_div():
    assert_equal((f * g)(3), f(3) * g(3))
    assert_equal((f * 2)(3), f(3) * 2)
    assert_equal((2 * f)(3), 2 * f(3))

    assert_equal((f / g)(3), f(3) / float(g(3)))
    assert_equal((f / 2)(3), f(3) / 2.0)
    assert_equal((2 / f)(3), 2.0 / f(3))


def test_derivative_mul_and_div():
    assert_equal((f * g).D(3), f.D(3) * g(3) + f(3) * g.D(3))
    assert_equal((f * 2).D(3), 2 * f.D(3))
    assert_equal((2 * f).D(3), 2 * f.D(3))

    assert_equal((f / g).D(3), (f.D(3) * g(3) - f(3) * g.D(3)) / (g(3) ** 2))
    assert_equal((f / 2).D(3), f.D(3) / 2.0)
    assert_equal((2 / f).D(3), -2.0 * f.D(3) / f(3) ** 2)


def test_function_pow():
    assert_equal((f ** g)(3), f(3) ** g(3))
    assert_equal((f ** 3)(3), f(3) ** 3)
    assert_equal((f ** -3)(3), f(3) ** -3.0)
    assert_equal((f ** 0.3)(3), f(3) ** 0.3)
    assert_equal((3 ** f)(3), 3 ** f(3))


def test_derivative_pow():
    #assert_equal((f ** g).D(3), f(3) ** g(3))
    assert_equal((f ** 3).D(3), 3 * f.D(3) * f(3) ** 2)
    assert_equal((f ** -3).D(3), -3 * f.D(3) * f(3) ** -4.0)
    assert_equal((f ** 0.3).D(3), 0.3 * f.D(3) * f(3) ** -0.7)
    #assert_equal((3 ** f).D(3), log(3) * f.D(3) * 3 ** f(3))


def test_function_compose():
    # composition
    assert_equal(f(g)(3), f(g(3)))
    assert_equal(g(f)(3), g(f(3)))
    assert_equal(f(f(f))(3), f(f(f(3))))
    assert_equal(g(g(g))(3), g(g(g(3))))
    assert_equal((f << g << f)(3), f(g(f(3))))


def test_derivative_compose():
    # composition
    assert_equal(f(g).D(3), f.D(g(3)) * g.D(3))
    assert_equal(f(f(f)).D(3), f.D(f(f(3))) * f.D(f(3)) * f.D(3))


def test_function_tensorise():
    h1 = f % g
    h2 = g % f
    assert_equal(h1(5, 7), f(5) * g(7))
    assert_equal(h2(6, 8), f(8) * g(6))
    assert_equal((h1 % h2)(5, 7, 6, 8), h1(5, 7) * h2(6, 8))


#    def test_function_vectorisation(self):
#        f1 = SimpleFunction(f=lambda x: x**2)
#        f2 = SimpleFunction(f=lambda x: x[0]**2+3*x[1])
#        print f1(range(3))

test_main()
