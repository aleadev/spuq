from dolfin import *
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.math_utils.multiindex import Multiindex

def test1():    # deprecated
    mesh1 = UnitSquare(5, 5)
    mesh2 = refine(mesh1)
    V1 = FunctionSpace(mesh1, 'CG', 1)
    V2 = FunctionSpace(mesh2, 'CG', 1)
    ex1 = Expression('a*x[0]+b*x[1]', a=1, b=1)
    ex2 = Expression('sin(a*pi*x[0])+sin(b*pi*x[1])', a=1, b=1)
    for ex in (ex1, ex2):
        f2 = interpolate(ex, V2)
        f21 = interpolate(f2, V1)
        f212 = interpolate(f21, V2)
        print "L2", errornorm(f2, f21, 'L2'), errornorm(f2, f212, 'L2')
        print "H1", errornorm(f2, f21, 'H1'), errornorm(f2, f212, 'H1')
    
    mis = (Multiindex(), Multiindex([1]), Multiindex([0, 1]))
    mv = MultiVectorWithProjection()
    mv[mis[0]] = FEniCSVector(f2)
    mv[mis[1]] = FEniCSVector(Function(V1))
    mv[mis[2]] = FEniCSVector(Function(V2))
    f2p1 = mv.get_back_projection(mis[0], mis[1])._fefunc
    f2p2 = mv.get_back_projection(mis[0], mis[2])._fefunc
    
    print "L2", errornorm(f2, f2p1, 'L2'), errornorm(f2, f2p2, 'L2')
    print "H1", errornorm(f2, f2p1, 'H1'), errornorm(f2, f2p2, 'H1')
    
    plot(mv[mis[0]]._fefunc, title="f2")
    plot(f2p1, title="f2p1")
    plot(f2p2, title="f2p2")
    interactive()

def test2():
    mesh1 = UnitSquare(5, 5)
    mesh2 = refine(mesh1)
    Vc1 = FunctionSpace(mesh1, 'CG', 1)
    Vc2 = FunctionSpace(mesh1, 'CG', 2)
    Vf1 = FunctionSpace(mesh2, 'CG', 1)
    ex1 = Expression('a*x[0]+b*x[1]', a=1, b=1)
    ex2 = Expression('sin(a*pi*x[0])+sin(b*pi*x[1])', a=1, b=1)
    ex3 = Expression('sin(a*pi*x[0])+sin(b*pi*x[1])', a=4, b=4)
    for ex in (ex1, ex2, ex3):
        ff1 = interpolate(ex, Vf1)
        fc1 = interpolate(ff1, Vc1)
        fc21 = interpolate(fc1, Vc1)
        fc2 = interpolate(ff1, Vc2)
        print "L2", errornorm(fc2, fc21, 'L2')
        print "H1", errornorm(fc2, fc21, 'H1')

def test3():
    mesh1 = UnitSquare(5, 5)
    mesh2 = refine(refine(mesh1))
    Vc1 = FunctionSpace(mesh1, 'CG', 1)
    Vf1 = FunctionSpace(mesh2, 'CG', 1)
    ex1 = Expression('a*x[0]+b*x[1]', a=1, b=1)
    ex2 = Expression('sin(a*pi*x[0])+sin(b*pi*x[1])', a=1, b=1)
    ex3 = Expression('sin(a*pi*x[0])+sin(b*pi*x[1])', a=4, b=4)
    for ex in (ex1, ex2, ex3):
        fc1 = interpolate(ex, Vc1)
        fcf1 = interpolate(fc1, Vf1)
        ff1 = interpolate(ex, Vf1)
        print "--------------------------"
        print "L2", errornorm(fcf1, ff1, 'L2')
        print "H1", errornorm(fcf1, ff1, 'H1')

print "TEST 2"
test2()
print "TEST 3"
test3()
