from __future__ import division
from dolfin import UnitSquare, CellFunction, MeshEditor, Mesh, adapt, plot, interactive, refine, info
import meshpy.triangle as triangle
import numpy as np
from numpy.random import randint
from operator import itemgetter
from copy import copy

# NOTE: this implementation requires meshpy
# http://documen.tician.de/meshpy/index.html
# HOWEVER, this approach of complete new mesh generation does not work properly!
# Successive refinement as done in test_joint_mesh2.py is much more reliable
# and should be used instead!

def generate_meshes(N = 5, iterations = 3, refinements = 0.5):
    mesh1 = UnitSquare(N,N)
    mesh2 = UnitSquare(N,N)
    for ref in range(iterations):
        print "refinement ", ref+1
        info(mesh1)
        info(mesh2)
        cf1 = CellFunction("bool", mesh1)
        cf2 = CellFunction("bool", mesh2)
        cf1.set_all(False)
        cf2.set_all(False)
        m1 = round(cf1.size()*refinements)
        m2 = round(cf2.size()*refinements)
        mi1 = randint(0,cf1.size(),m1)
        mi2 = randint(0,cf2.size(),m2)
        for i in mi1:
            cf1[i] = True
        for i in mi2:
            cf2[i] = True
#        newmesh1 = adapt(mesh1, cf1)
        newmesh1 = refine(mesh1, cf1)
#        newmesh2 = adapt(mesh2, cf2)
        newmesh2 = refine(mesh2, cf2)
        mesh1 = newmesh1
        mesh2 = newmesh2
    return [(0.,0.),(1.,0.),(1.,1.),(0.,1.)], mesh1, mesh2

def get_vertices(corners, mesh1, mesh2):
#    EPS = min(mesh1.hmin(),mesh2.hmin())/1e3
    EPS = 1e-6
    vertices = copy(corners)
    vertices.extend([tuple(v) for v in mesh1.coordinates()])
    vertices.extend([tuple(v) for v in mesh2.coordinates()])
    vertices = sorted(vertices, key=itemgetter(0,1))
    vi = 0
    while True:
        if abs(vertices[vi][0]-vertices[vi+1][0])+abs(vertices[vi][1]-vertices[vi+1][1]) < EPS:
            p = vertices.pop(vi+1)
            print "popped", p, " identical to ", vertices[vi]
        else:
            vi += 1
        if vi == len(vertices)-1:
            break
    for c in corners:
        try:
            vertices.remove(c)
        except:
            pass
    corners.extend(vertices)
    return corners

def main():
    def round_trip_connect(start, end):
      result = []
      for i in range(start, end):
        result.append((i, i+1))
      result.append((end, start))
      return result

    corners, mesh1, mesh2 = generate_meshes(2, 1, 0.3)
    points = get_vertices(corners, mesh1, mesh2)
    print "points", np.array(points)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(corners)-1))

    mesh = triangle.build(info, allow_volume_steiner=False, allow_boundary_steiner=False, min_angle=60)

    if False:
        print "vertices:"
        for i, p in enumerate(mesh.points):
            print i, p
        print "point numbers in triangles:"
        for i, t in enumerate(mesh.elements):
            print i, t

    finemesh = Mesh()
    ME = MeshEditor()
    ME.open(finemesh,2,2)
    ME.init_vertices(len(mesh.points))
    ME.init_cells(len(mesh.elements))
    for i,v in enumerate(mesh.points):
        ME.add_vertex(i,v[0],v[1])
    for i,c in enumerate(mesh.elements):
        ME.add_cell(i,c[0],c[1],c[2])
    ME.close()

    triangle.write_gnuplot_mesh("triangles.dat", mesh)

    plot(mesh1)
    plot(mesh2)
    plot(finemesh)
    interactive()

if __name__ == "__main__":
    main()
