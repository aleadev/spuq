__all__=["linear_operator", 
         "composed_operator", 
         "summed_operator", 
         "LinearOperator", 
         "FullVector",
         "FullLinearOperator"]


from .linear_operator import (LinearOperator, FullLinearOperator, FullVector)
from .composed_operator import *
from .summed_operator import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

def test():
    print  "Hallo"
    
