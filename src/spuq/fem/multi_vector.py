class MultiVector(object):
    #map multiindex to Vector (=coefficients + basis)
    def __init__(self):
        self.mi2vec = dict()

    def extend(self, mi, vec):
        self.mi2vec[mi] = vec

    def active_indices(self):
        return self.mi2vec.keys()

    def get_vector(self, mi):
        return self.mi2vec[mi]

    def __add__(self, other):
        assert self.active_indices() == other.active_indices()
        newvec = FooVector()
        for mi in self.active_indices():
            newvec.extend(mi, self.get_vector(mi) + other.get_vector(mi))
        return newvec

    def __mul__():
        return NotImplemented

    def __sub__():
        return NotImplemented
