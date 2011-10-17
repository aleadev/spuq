from spuq.fem.fem_mesh import FEMMesh
from dolfin import Mesh

class FEniCSMesh(FEMMesh):
    '''Wrapper for FEniCS/dolfin Mesh'''
    
    def __init__(self, mesh=None, filename=None):
        '''Initialise either with an existing mesh (generate e.g. with UnitSquare) or load from file.'''
        if mesh is not None:
            assert(isinstance(mesh, Mesh))
            self._mesh = mesh
        else:
            self._mesh = Mesh(filename);

    def refine(self, cells=None):
        from dolfin.mesh.refinement import refine
#        from dolfin.fem.projection import project
#        from dolfin.fem.interpolation import interpolate
        return refine(self.mesh, cells)

    @property
    def mesh(self):
        return self._mesh
