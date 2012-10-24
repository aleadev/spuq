__all__ = [
    "linalg",
    "fem",
    "polyquad",
    "stoch",
    "utils",
    ]

import spuq.utils.fixes

import numpy as np
np.set_printoptions(suppress=True, linewidth=1000, precision=3, edgeitems=20)

try:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
except ImportError:
    # silently ignore import errors here
    pass

