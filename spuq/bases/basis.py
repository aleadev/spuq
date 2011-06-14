class Basis(object):
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
