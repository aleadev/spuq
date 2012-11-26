from spuq.utils.testing import *
from spuq.utils.type_check import *

def test_simple():
    @takes(int)
    def foo(a):
        pass

    foo(1)
    assert_raises(TypeError, foo, "hallo")


def test_optional():
    @takes(int, optional(float),optional(int))
    def foo(a,b=None,c=None):
        return b

    assert_equal(foo(1),  None)
    assert_equal(foo(1, 3.0), 3)
    assert_equal(foo(1, b=3.0), 3)
    assert_equal(foo(1, c=7, b=3.0), 3)
    assert_equal(foo(1, b=3.0, c=7), 3)
    assert_equal(foo(1, c=7), None)
    assert_raises(TypeError, foo, 1, 3)


def test_optional2():
    @takes(int, b=optional(float))
    def foo(a,b=None):
        return b

    assert_equal(foo(1),  None)
    assert_equal(foo(1, 3.0), 3)
    assert_equal(foo(1, b=3.0), 3)
    # can't raise for this case, better avoid named
    # parameters in type checker list
    #assert_raises(TypeError, foo, 1, 3)

def test_strings():
    def checkit(types, args, str_expected):
        def foo(*args):
            pass
        str_actual = ""
        try:
            takes(*types)(foo)(*args)
        except TypeError as e:
            str_actual = e.message
        assert_equal(str_actual, "foo() " + str_expected)

    format = "got invalid parameter %d of %s instead of %s"
    # base types
    checkit([int], ["bla"], format % (1, str, int))
    checkit([float], ["bla"], format % (1, str, float))
    checkit([int], [1.0], format % (1, float, int))

    # choice of types
    checkit([(float, int)], ["bla"], format % (1, str, (float, int)))

    # type by string (StrChecker)
    checkit(["int"], [4.0], format % (1, float, int))

    # optional
    checkit([optional(int)], [4.0], format % (1, float, (int, type(None))))

    # TupleOfChecker
    checkit([tuple_of(int)], [1.0], format % (1, float, "tuple_of(<type 'int'>)"))

    # ListOfChecker
    checkit([list_of(int)], [(3.0,2.0)], format % (1, tuple, "list_of(<type 'int'>)"))

    # SequenceOfChecker
    checkit([sequence_of(int)], [(3.0,2.0)], format % (1, tuple, "sequence_of(<type 'int'>)"))

    # DictOfChecker
    checkit([dict_of(int, float)], [{"a":"b"}], format % (1, dict, "dict_of(<type 'int'>:<type 'float'>)"))

    # OneOfChecker
    checkit([one_of(1,2,3)], [4.0], format % (1, "4.0", "one_of(1, 2, 3)"))

    # ByRegexChecker
    checkit([by_regex("i.*")], [5], format % (1, "5", "by_regex(i.*)"))
    checkit([by_regex("i.*")], ["foo"], format % (1, "'foo'", "by_regex(i.*)"))


    

test_main()
