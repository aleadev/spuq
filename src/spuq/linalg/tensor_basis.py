from spuq.utils.type_check import anything, sequence_of, takes
from spuq.linalg.basis import Basis


class TensorBasis(Basis):
    @takes(anything, sequence_of(Basis))
    def __init__(self, bases):
        super(Basis, self).__init__()
        self._bases = bases

    def __repr__(self):
        return super(TensorBasis, self).__repr__()

    def __eq__(self, other):
        return (
            type(self) is type(other) and
            len(self._bases) == len(other._bases) and
            all(basis1 == basis2 for basis1, basis2 in zip(self,_bases, other._bases))
        )

    def copy(self):
        return TensorBasis([basis.copy() for basis in self._bases])

    @property
    def dim(self):
        return np.prod([basis.dim for basis in self._bases])
