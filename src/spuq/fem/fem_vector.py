from spuq.operators.full_vector import FullVector
from spuq.fem.fem_basis import FEMBasis

INTERPOLATE = "interpolate"

class FEMVector(FullVector):

  def __init__(self, coeff, basis ):
    assert isinstance( basis, FEMBasis )
    #super(self,FEMVector).__init__(coeff, basis)

  def transfer(self, basis, type=INTERPOLATE):
    assert isinstance( basis, FEMBasis )
    newcoeff = FEMBasis.transfer( self.coeff, self.basis, basis, type )
    return FEMVector( newcoeff, basis )
