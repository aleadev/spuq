from spuq.linalg.basis import FunctionBasis


class FEMBasis(FunctionBasis):
    def __init__(self, mesh):
        self.mesh = mesh

    def refine(self, faces):
        (newmesh, prolongate, restrict) = self.mesh.refine(faces)
        newbasis = FEMBasis(newmesh)
        prolop = Operator(prolongate, self, newbasis)
        restop = Operator(restrict, newbasis, self)
        return (newbasis, prolop, restop)

    def evaluate(self, x):
        # pass to dolfin
        pass

    @classmethod
    def transfer(coeff, oldbasis, newbasis, type):
        # let the mesh class the transfer accoring to type
        pass
