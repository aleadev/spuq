"""Common decorators for spuq."""  


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
    for attrname in cls.__dict__:
        attr = getattr(cls, attrname)
        func = getattr(attr, "__func__", None) 
        if func is not None and getattr(func, "__doc__") is None:
            func.__doc__ = _find_doc(cls, attrname)
    return cls


def copydoc(meth):
    """A decorators that copies docstrings of a method from base
    classes.

    `copydoc` copies the docstring of an overridden method from the
    method definition in the a base class. The lookup is depth first
    through the inheritance tree.
    """
    #func = meth.__func__
    func = meth
    funcname = func.__name__
    if getattr(func, "__doc__") is None:
        func.__doc__ = _find_doc(func.__class__, funcname)
    return func



import _decorator_contrib

copydoc = _decorator_contrib.inherits_docstring
