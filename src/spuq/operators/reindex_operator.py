from spuq.operators.linear_operator import LinearOperator
from spuq.operators.linear_operator import AbstractLinearOperator

class ReindexOperator(AbstractLinearOperator):
    def __init__( self, index_map, domain_basis, codomain_basis ):
        AbstractLinearOperator( self, domain_basis, codomain_basis )
        self.index_map = index_map
        
    def apply( ):
        pass

    def transpose():
        pass

    def invert():
        # is size(domain_basis)==size(codomain_basis) && index_map is full
        pass

    
