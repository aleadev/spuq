from spuq.fem.multi_vector import MultiVector
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import *

class ProjectionCache(object):
    """cache for projections of vectors in multivector wN onto different discrete spaces of other vectors in wN"""
    
    @takes(any, MultiVector)
    def __init__(self, wN, ptype=FEniCSBasis.PROJECTION.INTERPOLATION):
        """initialise cache with multivector"""
        
        assert wN
        self.wN = wN
        self._ptype = ptype
        self.clear()

    def clear(self, mu=None):
        """clear cache for alls or specific multiindex sets"""
        
        if MultiindexSet:
            self._projected_wN[mu] = MultiVector()
            self._projected_back_wN[mu] = MultiVector()
        else:
            self._projected_wN = MultiVector()        
            self._projected_back_wN = MultiVector()        

    @takes(any, MultiindexSet, MultiindexSet)
    def get_projection(self, mu_src, mu_dest, with_back_projection=True):
        """return projection (and back projection) of vector in multivector"""
        
        # projection of vector
        if mu_src not in self._projected_wN.keys():
            self._projected_wN[mu_src] = MultiVector()
        if mu_dest not in self._projected_wN[mu_src].keys():
            self._projected_wN[mu_src][mu_dest] = self.wN[mu_dest].functionspace.project(self.wN[mu_src], self._ptype)

        # return projected vector
        if not with_back_projection:
            return self._projected_wN[mu_src][mu_dest]
        
        # back projection of projected vector
        if mu_src not in self._projected_back_wN.keys():
            self._projected_back_wN[mu_src] = MultiVector()        
        if mu_dest not in self._projected_back_wN[mu_src].keys():
            self._projected_back_wN[mu_src][mu_dest] = self.wN[mu_src].functionspace.project(self._projected_wN[mu_src][mu_dest], self._ptype)

        # return projected vector and back projection
        if not with_back_projection:
            return self._projected_wN[mu_src][mu_dest], self._projected_back_wN[mu_src][mu_dest] 
            
    def __getitem__(self, mus):
        assert len(mus) >= 2 and len(mus) <= 3
        backp = True
        if len(mus)==3:
            backp = mus[2]
        return self.get_projection(mus[0], mus[1], backp)
