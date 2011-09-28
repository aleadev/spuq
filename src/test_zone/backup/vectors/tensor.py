class Tensor(GVector):
    def __init__(self):
        self.x=1
    def __add__(self,other):
        t=Tensor()
        t.x=self.x+other.x
        return t
    def __repr__(self):
        return str(self.x)
