from spuq.linalg.function import SimpleFunction, ConstFunction
from spuq.utils.testing import *

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

def test_function_addition_and_multiplication():
    # 
    print 'add/mult'
    assert_equal((f + g)(3), f(3) + g(3))
    assert_equal((f - g)(3), f(3) - g(3))
    assert_equal((f + 2)(3), f(3) + 2)
    assert_equal((2 + f)(3), 2 + f(3))
    assert_equal((f - 2)(3), f(3) - 2)
    assert_equal((2 - f)(3), 2 - f(3))

def foooo():

    assert_equal((2 + f - g)(3), 5)
    assert_equal((2 * f + g)(3), 24)
    assert_equal((f * g)(3), 54)
    assert_equal((2 * f * g)(3), 108)
    assert_equal((2 * f * g / 2)(3), 54)
    assert_equal((f / 2 + f ** 3)(3), 4 + 9 ** 3)
    assert_equal((f / 2.0 + f ** 3)(3), 4.5 + 9 ** 3)

def test_function_compose():
    # composition
    assert_equal(f(g)(3), 36)
    assert_equal(g(f)(3), 18)
    assert_equal(f(f(f)(3)), 3 ** 8)
    assert_equal(g(g(g)(3)), 24)

def test_function_exponentiation():
    assert_equal((f ** g)(3), 9 ** 6)
    assert_equal((g ** f)(3), 6 ** 9)
    h1 = f ** g
    h2 = g ** f
    # FIXME
#        print 'D3---',(h1**h2)(5,7,5,7), 350*490

def foobar():
    # tensorisation
    print 'tensor'
    print 'E1---', (f % g)(5, 7), 350
    print 'E2---', (f % g)(7, 5), 490
    h1 = f % g
    h2 = g % f
    print 'E3---', (h1 % h2)(5, 7, 5, 7), 350 * 490


#    def test_function_vectorisation(self):
#        f1 = SimpleFunction(f=lambda x: x**2)
#        f2 = SimpleFunction(f=lambda x: x[0]**2+3*x[1])
#        print f1(range(3))

test_main()
