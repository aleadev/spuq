def strclass(cls, with_mod=False):
    if with_mod:
        return "%s.%s" % (cls.__module__, cls.__name__)
    else:
        return "%s" % cls.__name__


class InequalityMixin(object):
    """Supplies an default implementation for ``!=``."""

    def __ne__(self, other):
        """Return true if the objects are not equal."""
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return not eq

     
class EqualityMixin(InequalityMixin):
    """Supplies an default implementation for ``==`` and ``!=``."""
    
    def __eq__(self, other):
        """Return true if the objects are equal.

        Equality as defined here is when the class match exactly and
        if both __dict__'s are exactly the same.
        """

        return (type(self) is type(other) and self.__dict__ == other.__dict__)
        
       


try:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
except ImportError:
    # silently ignore import errors here
    pass
