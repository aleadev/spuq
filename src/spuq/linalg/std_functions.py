from numpy import poly1d
from spuq.linalg.function import GenericFunction

class FunctionScalar(GenericFunction):
    def __init__(self, val=1):
        self.val = val
    
    
    
class FunctionPoly1d(GenericFunction):
    def __init__(self):
    
    