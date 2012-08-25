from __future__ import division
from dolfin import *
from numpy.random import randint

# create refined mesh
N=10
refs=20
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
    mesh = newmesh

destmesh = UnitSquare(N,N)
while True:
    cf = CellFunction("bool", destmesh)
    cf.set_all(False)
    rc = 0
    h = [c.diameter() for c in cells(destmesh)]
    for c in cells(mesh):
        p = c.midpoint()
        cid = destmesh.closest_cell(p)
        if h[cid] > c.diameter():
            cf[cid] = True
            rc += 1
    print "WILL REFINE", rc
    if rc:
        newmesh = refine(destmesh, cf)
        destmesh = newmesh
    else:
        break

plot(mesh)
plot(destmesh)
interactive()
