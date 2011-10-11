def strclass(cls, with_mod=False):
    if with_mod:
        return "%s.%s" % (cls.__module__, cls.__name__)
    else:
        return "%s" % cls.__name__


try:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
except ImportError:
    # silently ignore import errors here
    pass
