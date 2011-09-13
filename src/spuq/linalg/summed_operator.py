from spuq.operators.linear_operator import LinearOperator

class SummedLinearOperator(LinearOperator):
    """Wrapper class for linear operators adding two operators
    """
    
    def __init__(self, operators, factors=None, \
                     trans=None, inv=None, invtrans=None):
        """Takes two operators and returns the sum of those operators"""
        op1=operators[0];
        for op2 in operators:
            assert( op1.domain_basis() == op2.domain_basis() )
            assert( op1.codomain_basis() == op2.codomain_basis() )
        self.operators = operators
        self.factors = factors
        self.trans = None
        self.inv = None
        self.invtrans = None

    def domain_basis(self):
        "Returns the basis of the domain"
        return self.operators[0].domain_basis()

    def codomain_basis(self):
        "Returns the basis of the codomain"
        return self.operators[0].codomain_basis()

    def apply(self, vec):
        "Apply operator to vec which should be in the domain of op"
        # TODO: implement zero vector
        r=None
        for i, op in enumerate(self.operators):
            r1 = op.apply( vec )
            if self.factors and self.factors[i]!=1.0:
                r1=self.factors[i]*r1
            if r:
                r=r+r1
            else:
                r=r1
        return r
    
    def can_transpose(self):
        "Return whether the operator can transpose itself"
        if self.trans:
            return True
        else:
            return all( map( lambda op: op.can_transpose(), self.operators ) )

    def is_invertible(self):
        "Return whether the operator is invertible"
        if self.inv:
            return True
        else:
            return False

    def transpose(self):
        """Transpose the operator"""
        # TODO: should go into AbstractLinOp, here only create_transpose
        if self.trans:
            return self.trans
        else:
            return SummedLinearOperator(
                map( lambda op: op.transpose(), self.operators ),
                self.factors,
                trans=self,
                inv=self.invtrans,
                invtrans=self.inv)

    def invert(self):
        """Return an operator that is the inverse of this operator"""
        if self.inv:
            return self.inv
        else:
            # Cannot do this, the inverse of a sum is not the sum of the inverses
            # throw exeception?
            # TODO: should go if only 1 operators
            return None

    def as_matrix(self):
        return sum( map( lambda op: op.as_matrix(), self.operators ) )
