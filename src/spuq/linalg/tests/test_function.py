from spuq.linalg.function import SimpleFunction
from spuq.utils.testing import *

class TestFunctions(TestCase):
    def test_function_operators(self):
        f = SimpleFunction(f=lambda x: x**2)
        g = SimpleFunction(f=lambda x: 2*x)
        print 'A1---',f(3)
        print 'A2---',f((3))
        print 'A3---',f(f)(3)
        print 'A4---',f(f(f))(3)

        # addition and multiplication
        print 'add/mult'
        print 'B1---',(f+g)(3)
        print 'B2---',(f-g)(3)
        print 'B3---',(f+1)(3)
        print 'B4---',(2+f-g)(3)
        print 'B5---',(2*f+g)(3)
        print 'B6---',(2*f*g)(3)
        print 'B7---',(2*f*g/2)(3)
        print 'B8---',(f/2+f**3)(3)
        print 'B9---',(f/2*f**2)(3)
        print 'B10---',(2*f*g/2+f**2)(3)

        # composition
        print 'compose'
        print 'C1---',f(g)(3), 36
        print 'C2---',g(f)(3), 18
        
        # exponentiation
        print 'exp'
        print 'D1---',(f**g)(5,7), 350
        print 'D2---',(f**g)(7,5), 490
        h1=f**g
        h2=g**f
        # FIXME
#        print 'D3---',(h1**h2)(5,7,5,7), 350*490

        # tensorisation
        print 'tensor'
        print 'E1---',(f%g)(5,7), 350
        print 'E2---',(f%g)(7,5), 490
        h1=f%g
        h2=g%f
        print 'E3---',(h1%h2)(5,7,5,7), 350*490


#    def test_function_vectorisation(self):
#        f1 = SimpleFunction(f=lambda x: x**2)
#        f2 = SimpleFunction(f=lambda x: x[0]**2+3*x[1])
#        print f1(range(3))

test_main()
