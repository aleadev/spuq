from spuq.stochastics.stochastic_basis import StochasticBasis

class RandomField(object):
    def __init__(self):
        # maybe this should go into TensorProductBasis
        self.Phi = SpatialBasis()
        self.Psi = StochasticBasis()
        # maybe this should go into derived classes
        self.coeffs = GramMatrix()

    def __add__(self, other):
        # add only if same spatial and stochastic basis are used
        # or add scalar, or spatial function
        return NotImplemented

    def __mult__(self, other):
        # mult by other field
        # mult by scalar
        return NotImplemented


class SeparatedRandomField(RandomField):
    # e.g. for KL representations
    pass


class FullRandomField(RandomField):
    # e.g. for full (PCE,gPC) representations
    pass


class KL(object):
    pass


class PCE(object):
    pass


class KL_PCE(object):
    pass
