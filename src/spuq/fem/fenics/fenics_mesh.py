from spuq.fem.mesh import FEMMesh


class FEniCSMesh(FEMMesh):
    def __init__(self):
        from dolfin import Mesh
        self.fenics_mesh = Mesh()

    def refine(self, faces):
        print "warning: refine not implemented yet"
        return (None, None, None)
        new_fenics_mesh = self.fenics_mesh.refine(faces)
        prolongate = lambda x: fenics.project(x, fenics_mesh,
                                              new_fenics_mesh)
        restrict = lambda x: fenics.project(x, new_fenics_mesh,
                                            fenics_mesh)
        return (Mesh(new_fenics_mesh), prolongate, restrict)
