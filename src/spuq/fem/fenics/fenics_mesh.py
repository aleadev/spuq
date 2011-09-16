from spuq.fem.fem_mesh import FEMMesh
from dolfin import Mesh

class FEniCSMesh(FEMMesh):
    '''wrapper for FEniCS/dolfin Mesh'''
    
    def __init__(self, mesh=None, filename=None):
        '''initialise either with an existing mesh (generate e.g. with UnitSquare) or load from file.'''
        if mesh is not None:
            assert(isinstance(mesh, Mesh))
            self.mesh = mesh
        else:
            self.mesh = Mesh(filename);

    def refine(self, cells=None):
        from dolfin.mesh.refinement import refine
#        from dolfin.fem.projection import project
#        from dolfin.fem.interpolation import interpolate
        return refine(self.mesh, cells)
#        prolongate = lambda vec: project(vec, self.mesh, new_mesh)
#        restrict = lambda vec: project(vec, new_mesh, self.mesh)
#        return (new_mesh, prolongate, restrict)

    @property
    def mesh(self):
        return self._mesh
    
    @mesh.setter
    def mesh(self, newmesh):
        self._mesh = newmesh
        