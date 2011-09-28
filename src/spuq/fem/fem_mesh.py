from abc import *

class FEMMesh(object):
    '''ABC for FEM meshes.'''

    __metaclass__ = ABCMeta
    
    def num_cells(self):
        '''return number cells'''
        return self.mesh.num_cells()
    
    def num_edges(self):
        '''return number edges'''
        return self.mesh.num_edges()
    
    def num_vertices(self):
        '''return number vertices'''
        return self.mesh.num_vertices()
    
    def cells(self):
        '''return cells'''
        return self.mesh.cells()
    
    def coordinates(self):
        '''return vertex coordinates'''
        return self.mesh.coordinates()
    
    @abstractmethod
    def refine(self, cells=None):
        '''refine mesh uniformly or wrt cells, returns new mesh'''
        return NotImplemented
    
    @abstractproperty
    def mesh(self):
        return NotImplemented
    