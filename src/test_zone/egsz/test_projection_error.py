from __future__ import division

from dolfin import *
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.math_utils.multiindex import Multiindex

# FEniCS logging
from dolfin import (set_log_level, set_log_active, INFO, DEBUG, WARNING)
set_log_active(True)
set_log_level(WARNING)

ex1 = Expression('a*x[0]+b*x[1]', a=1, b=1)
ex2 = Expression('sin(2*a*pi*x[0])+sin(2*b*pi*x[1])', a=1, b=1)
ex3 = Expression('sin(2*a*pi*x[0])*cos(2*b*pi*x[1])', a=4, b=4)
ex4 = Expression('sin(2*a*pi*x[0])*cos(2*b*pi*x[1])', a=14, b=4)
EX = (ex1, ex2, ex3, ex4)

def test1():    # deprecated
    print "TEST 1"
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
    print "TEST 2"
    mesh1 = UnitSquare(5, 5)
    mesh2 = refine(mesh1)
    Vc1 = FunctionSpace(mesh1, 'CG', 1)
    Vc2 = FunctionSpace(mesh1, 'CG', 2)
    Vf1 = FunctionSpace(mesh2, 'CG', 1)
    for ex in EX:
        ff1 = interpolate(ex, Vf1)
        fc1 = interpolate(ff1, Vc1)
        fc21 = interpolate(fc1, Vc1)
        fc2 = interpolate(ff1, Vc2)
        print "L2", errornorm(fc2, fc21, 'L2')
        print "H1", errornorm(fc2, fc21, 'H1')

def test3():
    print "TEST 3"
    mesh1 = UnitSquare(5, 5)
    mesh2 = refine(refine(mesh1))
    Vc1 = FunctionSpace(mesh1, 'CG', 1)
    Vf1 = FunctionSpace(mesh2, 'CG', 1)
    for ex in EX:
        fc1 = interpolate(ex, Vc1)
        fcf1 = interpolate(fc1, Vf1)
        ff1 = interpolate(ex, Vf1)
        print "--------------------------"
        print "L2", errornorm(fcf1, ff1, 'L2')
        print "H1", errornorm(fcf1, ff1, 'H1')

def test4():
    print "TEST 4"
    mis = (Multiindex(), Multiindex([1]), Multiindex([0, 1]), Multiindex([1, 1]))
    mv = MultiVectorWithProjection()
    V1 = FEniCSBasis(FunctionSpace(UnitSquare(5, 5), 'CG', 1))
    V2, _, _ = V1.refine()
    V3, _, _ = V2.refine()
    V4 = V1.refine_maxh(1 / 30, uniform=True)
    print "mesh1", V1.mesh, "maxh,minh=", V1.maxh, V1.minh
    print "mesh2", V2.mesh, "maxh,minh=", V2.maxh, V2.minh
    print "mesh3", V3.mesh, "maxh,minh=", V3.maxh, V3.minh
    print "mesh4", V4.mesh, "maxh,minh=", V4.maxh, V4.minh
    F2 = FEniCSVector.from_basis(V2)
    F3 = FEniCSVector.from_basis(V3)
    F4 = FEniCSVector.from_basis(V4)
    mv[mis[0]] = FEniCSVector.from_basis(V1)
    mv[mis[1]] = F2
    mv[mis[2]] = F3
    mv[mis[3]] = F4
    for j, ex in enumerate(EX):
        print "ex[", j, "] =================="
        for degree in range(1, 4):
            print "\t=== degree ", degree, "==="
            F2.interpolate(ex)
            F3.interpolate(ex)
            F4.interpolate(ex)
            err1 = mv.get_projection_error_function(mis[1], mis[0], degree, refine_mesh=False)
            err2 = mv.get_projection_error_function(mis[2], mis[0], degree, refine_mesh=False)
            err3 = mv.get_projection_error_function(mis[3], mis[0], degree, refine_mesh=False)
            print "\t[NO DESTINATION MESH REFINEMENT]"
            print "\t\tV2 L2", norm(err1._fefunc, 'L2'), "H1", norm(err1._fefunc, 'H1')
            print "\t\tV3 L2", norm(err2._fefunc, 'L2'), "H1", norm(err2._fefunc, 'H1')
            print "\t\tV4 L2", norm(err3._fefunc, 'L2'), "H1", norm(err3._fefunc, 'H1')
            err1 = mv.get_projection_error_function(mis[1], mis[0], degree, refine_mesh=True)
            err2 = mv.get_projection_error_function(mis[2], mis[0], degree, refine_mesh=True)
            err3 = mv.get_projection_error_function(mis[3], mis[0], degree, refine_mesh=True)
            print "\t[WITH DESTINATION MESH REFINEMENT]"
            print "\t\tV2 L2", norm(err1._fefunc, 'L2'), "H1", norm(err1._fefunc, 'H1')
            print "\t\tV3 L2", norm(err2._fefunc, 'L2'), "H1", norm(err2._fefunc, 'H1')
            print "\t\tV4 L2", norm(err3._fefunc, 'L2'), "H1", norm(err3._fefunc, 'H1')

#test2()
#test3()
test4()
