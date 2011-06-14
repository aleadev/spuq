class ComposedOpertor(LinearOperator):
    def __init__(op1,op2):
        assert( op1.range_basis()==op2.domain_basis() );
        self.op1 = op1
        self.op2 = op2
