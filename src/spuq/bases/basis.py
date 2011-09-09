class Basis(object):
    def dim(self):
        return NotImplemented

class EuclideanBasis( Basis ):
    def __init__(self, dim):
        self._dim=dim
    
    def dim(self):
        return self._dim

    def __eq__(self,other):
        return isinstance(other, EuclideanBasis) and\
            self._dim==other._dim


class FunctionBasis(Basis):
    def getGramian(self):
        # return a LinearOperator, not necessarily a matrix
        return NotImplemented

    def evalAt(self,vector):
        return NotImplemented

    def integrateOver(self,function):
        # do we need that? What functions are general enough to be
        # included in a Basis base blass
        return NotImplemented

    # projection (oblige, orthogonal), 
    # arg dim/arg domain?
