__all__ = [
    "linalg",
    "fem",
    "polyquad",
    "stoch",
    "utils",
    ]

#import statistics
#from statistics import *

#import linalg
#from linalg import *

#import utils
#from utils import *

#import bases


try:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
except ImportError:
    # silently ignore import errors here
    pass
