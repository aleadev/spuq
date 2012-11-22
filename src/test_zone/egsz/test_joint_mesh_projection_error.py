from dolfin import UnitSquare, plot, CellFunction, refine, Expression, Function, norm, FunctionSpace, interpolate
from spuq.fem.fenics.fenics_utils import create_joint_mesh
from random import random

def test1():
    # setup meshes
    P = 0.3
    ref1 = 4
    ref2 = 14
    mesh1 = UnitSquare(2, 2)
    mesh2 = UnitSquare(2, 2)
    # refinement loops
    for level in range(ref1):
        mesh1 = refine(mesh1)
    for level in range(ref2):
        # mark and refine
        markers = CellFunction("bool", mesh2)
        markers.set_all(False)
        # randomly refine mesh
        for i in range(mesh2.num_cells()):
            if random() <= P:
                markers[i] = True
        mesh2 = refine(mesh2, markers)

    # create joint meshes
    mesh1j, parents1 = create_joint_mesh([mesh2], mesh1)
    mesh2j, parents2 = create_joint_mesh([mesh1], mesh2)

    # evaluate errors  joint meshes
    ex1 = Expression("sin(2*A*x[0])*sin(2*A*x[1])", A=10)
    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    V1j = FunctionSpace(mesh1j, "CG", 1)
    V2j = FunctionSpace(mesh2j, "CG", 1)
    f1 = interpolate(ex1, V1)
    f2 = interpolate(ex1, V2)
    # interpolate on respective joint meshes
    f1j = interpolate(f1, V1j)
    f2j = interpolate(f2, V2j)
    f1j1 = interpolate(f1j, V1)
    f2j2 = interpolate(f2j, V2)
    # evaluate error with regard to original mesh
    e1 = Function(V1)
    e2 = Function(V2)
    e1.vector()[:] = f1.vector() - f1j1.vector()
    e2.vector()[:] = f2.vector() - f2j2.vector()
    print "error on V1:", norm(e1, "L2")
    print "error on V2:", norm(e2, "L2")

    plot(f1j, title="f1j")
    plot(f2j, title="f2j")
    plot(mesh1, title="mesh1")
    plot(mesh2, title="mesh2")
    plot(mesh1j, title="joint mesh from mesh1")
    plot(mesh2j, title="joint mesh from mesh2", interactive=True)

test1()
