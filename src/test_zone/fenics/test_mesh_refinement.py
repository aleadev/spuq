from __future__ import division
from dolfin import *
import math
import numpy as np
from numpy.linalg import norm
from numpy.random import randint
from collections import defaultdict

# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.
# http://pseentertainmentcorp.com/smf/index.php?topic=545.0
def point_inside_polygon(p, poly):
    n = len(poly)
    inside = False
    x, y = p[0], p[1]
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if p2x == x and p2y == y:
            return True
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def triangle_area(a, b, c) :
    return 0.5 * norm(np.cross(b - a, c - a))

def point_inside_triangle(p, tri):
    A1 = triangle_area(*tri)
    A = 0
    for i in range(3):
        p1 = tri[i % 3]
        p2 = tri[(i + 1) % 3]
        A += triangle_area(p, p1, p2)
    return A1 == A

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
    P = 0.3
    refinements = 10
    parentmesh = UnitSquare(2, 2)
    mesh = parentmesh
    # setup parent cells
    parents = {}
    for c in cells(mesh):
        parents[c.index()] = [c.index()]
    PM = []
    # refinement loop
    for level in range(refinements):
        # mark and refine
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
            pmap[pc[i]].append(i)
#        print "iteration", level, pmap
        PM.append(pmap)
        mesh = mesh2

    # determine association to parent cells
    for level in range(len(PM)):
        for parentid, childids in parents.iteritems():
            newchildids = []
            for cid in childids:
                for cid in PM[level][cid]:
                    newchildids.append(cid)
            parents[parentid] = newchildids
#        print "parents at level", level, parents
    print parents
    assert sum(len(c) for c in parents.itervalues()) == mesh.num_cells()

    # check association for each cell, i.e.
    # all triangles of the fine mesh have to lie within the respective parent triangle
    xy = parentmesh.coordinates()
    elems = parentmesh.cells()
    xy2 = mesh.coordinates()
    elems2 = mesh.cells()
    for pid, cids in parents.iteritems():
        tri = [xy[i] for i in elems[pid]]
        for cid in cids:
            for vid in elems2[cid]:
                p = xy2[vid]
#                print "TEST", p, "inside", tri
                assert point_inside_triangle(p, tri)

    plot(parentmesh)
    plot(mesh2, interactive=True)

#test1()
test2()
