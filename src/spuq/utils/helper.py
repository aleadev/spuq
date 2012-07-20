

# source: http://blog.mathieu-leplatre.info/python-lazy-hasattr.html
def lazyhasattr(obj, name):
    return any(name in d for d in (obj.__dict__,
                                   obj.__class__.__dict__))
