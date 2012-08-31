from dolfin import *
from time import time

def T(t):
    dt = time() - t
    t = time()
    print dt

N= 1000
t = time()
mesh = UnitSquare(N,N)
T(t)
info(mesh)
V = mesh.coordinates()
C = mesh.cells()
mesh2 = Mesh()
ME = MeshEditor()
ME.open(mesh2,2,2)
ME.init_vertices(V.shape[0])
ME.init_cells(C.shape[0])
for i,v in enumerate(V):
    ME.add_vertex(i,v[0],v[1])
for i,c in enumerate(C):
    ME.add_cell(i,c[0],c[1],c[2])
ME.close()
T(t)
info(mesh2)
