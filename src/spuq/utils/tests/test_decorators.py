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


def test_int_cache():
    global call_count
    call_count = 0
    @simple_int_cache(10)
    def foo(n):
        global call_count
        call_count = call_count + 1
        return 10*n

    # first pass, evaluate some arguments within and without range
    assert_equal(foo(-1), -10)
    assert_equal(foo(0), 0)
    assert_equal(foo(2), 20)
    assert_equal(foo(9), 90)
    assert_equal(foo(10), 100)

    # second pass, call_count should only increase for values outside range
    assert_equal(call_count, 5)
    assert_equal(foo(-1), -10)
    assert_equal(call_count, 6)
    assert_equal(foo(0), 0)
    assert_equal(call_count, 6)
    assert_equal(foo(2), 20)
    assert_equal(call_count, 6)
    assert_equal(foo(9), 90)
    assert_equal(call_count, 6)
    assert_equal(foo(10), 100)
    assert_equal(call_count, 7)



test_main()
