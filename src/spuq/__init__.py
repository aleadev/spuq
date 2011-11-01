__all__ = [
    "linalg",
    "fem",
    "polyquad",
    "stoch",
    "utils",
    ]

import spuq.utils.fixes


try:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
except ImportError:
    # silently ignore import errors here
    pass

