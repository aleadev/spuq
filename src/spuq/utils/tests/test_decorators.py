import numpy as np

from spuq.utils.testing import *
from spuq.utils.decorators import *


class TestCopyDocs(TestCase):
    
    class A(object):
        def x(self):
            """A.x"""
            pass
        def y(self):
            pass
        def z(self):
            pass

    @copydocs
    class Bc(A):
        def x(self):
            pass
        def y(self):
            """B.y"""
            pass

    def test_copy_doc1(self):
        assert_equal(self.A.x.__doc__, "A.x")
        assert_equal(self.Bc.x.__doc__, "A.x")
        assert_equal(self.Bc.y.__doc__, "B.y")

    class B(A):
        def x(self):
            pass
        def y(self):
            """B.y"""
            pass

    @copydocs
    class C(B):
        def x(self):
            pass
        def z(self):
            """C.z"""
            pass

    def test_copy_doc2(self):
        assert_equal(self.C.x.__doc__, "A.x")
        assert_equal(self.C.y.__doc__, "B.y")
        assert_equal(self.C.z.__doc__, "C.z")

    @copydocs
    class Cc(Bc):
        def x(self):
            """C.x"""
            pass
        def z(self):
            """C.z"""
            pass

    def test_copy_doc3(self):
        assert_equal(self.Cc.x.__doc__, "C.x")
        assert_equal(self.Cc.y.__doc__, "B.y")
        assert_equal(self.Cc.z.__doc__, "C.z")


class TestCopyDoc(TestCase):
    
    class A(object):
        def x(self):
            """A.x"""
            pass
        def y(self):
            """A.y"""

    class B(A):
        @copydoc
        def x(self):
            pass
        def y(self):
            pass

    def test_copy_doc1(self):
        assert_equal(self.B.x.__doc__, "A.x")
        assert_equal(self.B.y.__doc__, None)



class TestSimpleIntCache(TestCase):

    def test_int_cache(self):
        add = 0
        @simple_int_cache(10)
        def foo(n):
            return 10*n+add

        # first pass, should return the same as the function
        assert_equal(foo(-1), -10)
        assert_equal(foo(0), 0)
        assert_equal(foo(2), 20)
        assert_equal(foo(9), 90)
        assert_equal(foo(10), 100)

        # second pass should return the same the first pass
        assert_equal(foo(-1), -10)
        assert_equal(foo(0), 0)
        assert_equal(foo(2), 20)
        assert_equal(foo(9), 90)
        assert_equal(foo(10), 100)

        # third pass; changing add we can see what has truly been
        # cached; namely -1, 5 and 10 should not have been cached,
        # returning different values while the others should return
        # the same as in the second pass
        add = 1
        assert_equal(foo(-1), -10+1)
        assert_equal(foo(0), 0)
        assert_equal(foo(2), 20)
        assert_equal(foo(5), 50+1)
        assert_equal(foo(9), 90)
        assert_equal(foo(10), 100+1)


test_main()
