"""deferred_binder module"""

"""http://code.activestate.com/recipes/577745/"""
"""http://code.activestate.com/recipes/577746-inherit-method-docstrings-using-only-function-deco/"""

"""docfunc module"""


class DeferredBinder(object):
    """A descriptor that defers binding for an object.

    The binding is delayed until the name to which the descriptor is
    bound is accessed.  This is also the name to which the object would
    have been bound if the descriptor hadn't been used.

    The last parameter to init is a method with the following description:
        Transform the target and return it.

          name - the name to which the target will be bound.
          context - namespace, and optionally the class, in which the
                target will be bound to the name.
          obj - the instance of the class that is involved in calling
                this method, if any.

    """

    def __init__(self, name, target, transform):
        self.name = name
        self.target = target
        self.transform = transform

    def __get__(self, obj, cls):
        context = (cls.__dict__, cls)
        target = self.transform(self.name, context, self.target, obj)
        setattr(cls, self.name, self.target)
        return target


def get_doc(cls, fname, member=True):
    """Returns the function docstring the method should inherit.

      cls - the class from which to start looking for the method.
      fname - the method name on that class
      member - is the target function already bound to cls?

    """

    bases = cls.__mro__[:]
    if member:
        bases = bases[1:]
    for base in bases:
        func = getattr(base, fname, None)
        if not func:
            continue
        doc = getattr(func, '__doc__', None)
        if doc is None:
            continue
        return doc
    return default      # is default defined?


def doc_func_transform(name, context, target, obj=None):
    """The DeferredBinder transform for this subclass.

      name - the attribute name to which the function will be bound.
      context - the class/namespace to which the function will be bound.
      target - the function that will be bound.
      obj - ignored.

    The DeferredBinder descriptor class will replace itself with the
    result of this method, when the name to which the descriptor is requested
    for the first time.  This can be on the class or an instances of the
    class.

    This way the class to which the method is bound is available so that the
    inherited docstring can be identified and set.

    """

    namespace, cls = context
    doc = target.__doc__
    if doc is None:
        doc = get_doc(cls, name)
    target.__doc__ = doc
    return target


def inherits_docstring(func):
    """A decorator that returns a new DocFunc object.

      func - the function to decorate.
    """

    return DeferredBinder(func.__name__, func, doc_func_transform)

