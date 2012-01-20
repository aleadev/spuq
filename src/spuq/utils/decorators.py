"""Common decorators for spuq."""  


def copydocs(cls):
    """A decorators that copies docstrings of functions from base
    classes.

    When base class method are overridden in derived classes, it is
    often not necessary to alter the docstring of those methods, as
    usually only the implementation has changed, but not the
    interface. How the implementatin has changed or is now
    implemented, can usually be inferred from the class
    description. Using `copydocs` then, saves time and removes
    duplication of docstrings.

    `copydocs` applies to a class and copies all the functions
    docstrings from base classes if none were specified. The lookup is
    depth first through the inheritance tree.
    """

    def _find_doc(cls, attrname):
        """Helper function for copydoc/copydocs"""
        attr = getattr(cls, attrname, None)
        func = getattr(attr, "__func__", None) 
        doc = getattr(func, "__doc__", None) 
        if doc is not None:
            return doc

        for b in cls.__bases__:
            doc = _find_doc(b, attrname)
            if doc is not None:
                return doc

        return doc

    for attrname in cls.__dict__:
        attr = getattr(cls, attrname)
        func = getattr(attr, "__func__", None) 
        if func is not None and getattr(func, "__doc__") is None:
            func.__doc__ = _find_doc(cls, attrname)
    return cls



def simple_int_cache(size):
    class IntCache:
        Empty = object()

        def __init__(self, func, size):
            self.func = func
            self.cache = size * [self.Empty]
        def __call__(self, n):
            if n < 0 or n >= len(self.cache):
                return self.func(n)
            if self.cache[n] is self.Empty:
                self.cache[n] = self.func(n)
            return self.cache[n]

    def decorator(func):
        return IntCache(func, size)
    return decorator








# http://code.activestate.com/recipes/577689-enforce-__all__-outside-the-import-antipattern/
import sys
import types
import warnings

class EncapsulationWarning(RuntimeWarning): pass

class ModuleWrapper(types.ModuleType):
    def __init__(self, context):
        self.context = context
        super(ModuleWrapper, self).__init__(
                context.__name__,
                context.__doc__)

    def __getattribute__(self, key):
        context = object.__getattribute__(self, 'context')
        if key not in context.__all__:
            warnings.warn('%s not in %s.__all__' % (key, context.__name__),
                          EncapsulationWarning,
                          2)
        return context.__getattribute__(key)

#import example
#sys.modules['example'] = ModuleWrapper(example)
