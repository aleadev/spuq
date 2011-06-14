class LinearOperator(object):
    def range_basis(self):
        "Returns the dimension of the range of Op"
        return NotImplemented
    def dim_range(self):
        "Returns the dimension of the range of Op"
        return NotImplemented
    def dim_domain(self):
        "Returns the dimension of the domain of Op"
        return NotImplemented
    def __call__(self, arg):
        return NotImplemented
    def apply(self):
        pass
    def transpose(self):
        pass
    def invert(self):
        pass
    pass
