from __future__ import division
from dolfin import *
from numpy.random import randint
from collections import defaultdict

def test1():
    N = 10
    refs = 5
    frac = 0.1
    mesh = UnitSquare(N, N)
    for r in range(refs):
        print "refinement", r
        cf = CellFunction("bool", mesh)
        cf.set_all(False)
        M = mesh.num_cells()
        ri = randint(0, M, round(M * frac))
        for i in ri:
            cf[i] = True
        newmesh = refine(mesh, cf)
        info(mesh)
        mesh = newmesh
        info(mesh)
        plot(mesh, interactive=True)

def test2():
    from random import random
    P = 0.4
    refinements = 10
    mesh = UnitSquare(1, 1)
    PM = []
    for level in range(refinements):
        markers = CellFunction("bool", mesh)
        markers.set_all(False)
        # randomly refine mesh
        for i in range(mesh.num_cells()):
            if random() <= P:
                markers[i] = True
        mesh2 = refine(mesh, markers)

#        info(mesh, True)
#        info(mesh2.data(), True)
#        info(mesh2.data().mesh_function("parent_cell"), True)
#        info(mesh2.data().mesh_function("parent_facet"), True)

        # determine parent cell map
        pc = mesh2.data().mesh_function("parent_cell")
        pmap = defaultdict(list)
        for i in range(pc.size()):
            cellid = pc[i]
            pmap[cellid].append(i)
        PM.append(pmap)
        mesh = mesh2

    M = PM[0]
    for level in range(1, len(PM)):
        childids = []
        for cellid in M:
            childids.append(PM[level][cellid])
        M[k] = childids
    print "M", M

    # TODO: check association for each cell

    plot(mesh2, interactive=True)

#test1()
test2()
