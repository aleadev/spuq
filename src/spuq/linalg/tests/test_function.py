from spuq.linalg.function import SimpleFunction
from spuq.utils.testing import *

class TestFunctions(TestCase):
    def test_function_operators(self):
        f = SimpleFunction(f=lambda x: x**2)
        g = SimpleFunction(f=lambda x: 2*x)
        print f(3)
        print f((3))
        print f(f)(3)
        print f(f(f))(3)

        # addition and multiplication
        print 'add/mult'
        print (f+g)(3)
        print (f-g)(3)
        print (2+f-g)(3)
        print (2*f+g)(3)
        print (2*f*g)(3)
        print (2*f*g/2)(3)
        print (f/2+f**3)(3)
        print (f/2*f**2)(3)
        print (2*f*g/2+f**2)(3)

        # composition
        print 'compose'
        print f(g)(3), 36
        print g(f)(3), 18
        
        # exponentiation
        print 'exp'
        print (f**g)(5,7), 350
        print (f**g)(7,5), 490
        h1=f**g
        h2=g**f
# FIXME!
#        print (h1**h2)(5,7,5,7), 350*490

        # tensorisation
        print 'tensor'
        print (f%g)(5,7), 350
        print (f%g)(7,5), 490
        h1=f%g
        h2=g%f
        print (h1%h2)(5,7,5,7), 350*490


#    def test_function_vectorisation(self):
#        f1 = SimpleFunction(f=lambda x: x**2)
#        f2 = SimpleFunction(f=lambda x: x[0]**2+3*x[1])
#        print f1(range(3))

test_main()
