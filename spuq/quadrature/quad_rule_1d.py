#from numpy import array, ndarray, zeros, ones, diag, arange

class Quad_Rule_1d(object):
    """Abstract base class for 1d quadrature rules on [0,1]"""
    
    @staticmethod
    def getPointsWeights(np):
        return NotImplemented

    @staticmethod
    def transform(p,interval):
        """Transform points p from [0,1] to interval"""
        return (interval[1]-interval[0])*p + interval[0]
    