from spuq.operators.linear_operator import LinearOperator

class ComposedLinearOperator(LinearOperator):
    """Wrapper class for linear operators that are composed of other
    linear operators
    """
    
    def __init__(self, op1, op2, trans=None, inv=None, invtrans=None):
        """Takes two operators and returns the composition of those operators"""
        assert( op1.codomain_basis() == op2.domain_basis() )
        self.op1 = op1
        self.op2 = op2
        self.trans = None
        self.inv = None
        self.invtrans = None

    def domain_basis(self):
        "Returns the basis of the domain"
        return self.op1.domain_basis()

    def codomain_basis(self):
        "Returns the basis of the codomain"
        return self.op2.codomain_basis()

    def apply(self, vec):
        "Apply operator to vec which should be in the domain of op"
        r = self.op1.apply( vec )
        r = self.op2.apply( r )
        return r
    
    def can_transpose(self):
        "Return whether the operator can transpose itself"
        if self.trans:
            return True
        else:
            return self.op1.can_transpose() and self.op2.can_transpose()

    def is_invertible(self):
        "Return whether the operator is invertible"
        if self.inv:
            return True
        else:
            return self.op1.is_invertible() and self.op2.is_invertible()

    def transpose(self):
        """Transpose the operator"""
        if self.trans:
            return self.trans
        else:
            return ComposedLinearOperator(
                self.op2.transpose(),
                self.op1.transpose(),
                trans=self,
                inv=self.invtrans,
                invtrans=self.inv)

    def invert(self):
        """Return an operator that is the inverse of this operator"""
        if self.inv:
            return self.inv
        else:
            return ComposedLinearOperator(
                self.op2.invert(),
                self.op1.invert(),
                inv = self,
                trans = self.invtrans,
                invtrans = self.trans)

    def as_matrix(self):
        return self.op2.as_matrix() * self.op1.as_matrix()

