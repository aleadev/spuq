from __future__ import division
from dolfin import *
from numpy.random import randint

N=10
refs=5
frac=0.1
mesh = UnitSquare(N,N)
for r in range(refs):
    print "refinement", r
    cf = CellFunction("bool", mesh)
    cf.set_all(False)
    M = mesh.num_cells()
    ri = randint(0, M, round(M*frac))
    for i in ri:
        cf[i] = True
    newmesh = refine(mesh, cf)
    info(mesh)
    mesh = newmesh
    info(mesh)
plot(mesh, interactive=True)
