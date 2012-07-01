from __future__ import division

from dolfin import *
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.fem.fenics.fenics_vector import FEniCSVector
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.math_utils.multiindex import Multiindex

# FEniCS logging
from dolfin import (set_log_level, set_log_active, INFO, DEBUG, WARNING, parameters)
from dolfin import (assemble, dx, ds, dS, solve, nabla_grad, inner, Function, FunctionSpace,
                        norm, TrialFunction, TestFunction, UnitSquare)
import ufl
set_log_active(False)
set_log_level(WARNING)

ex_degree = 3
element = FiniteElement('Lagrange', ufl.triangle, 1)
ex1a = Expression('a*x[0]+b*x[1]', a=1, b=1, element=element)
ex2a = Expression('A*sin(2*a*pi*x[0])+A*sin(2*b*pi*x[1])', A=10, a=1, b=1, element=element)
ex3a = Expression('A*sin(2*a*pi*x[0])*cos(2*b*pi*x[1])', A=10, a=4, b=4, element=element)
ex4a = Expression('A*sin(2*a*pi*x[0])*cos(2*b*pi*x[1])', A=10, a=14, b=4, element=element)
ex1b = Expression('a*x[0]+b*x[1]', a=1, b=1, degree=ex_degree, element=element)
ex2b = Expression('A*sin(2*a*pi*x[0])+A*sin(2*b*pi*x[1])', A=10, a=1, b=1, degree=ex_degree, element=element)
ex3b = Expression('A*sin(2*a*pi*x[0])*cos(2*b*pi*x[1])', A=10, a=4, b=4, degree=ex_degree, element=element)
ex4b = Expression('A*sin(2*a*pi*x[0])*cos(2*b*pi*x[1])', A=10, a=14, b=4, degree=ex_degree, element=element)
EX = (ex1a, ex1b, ex2a, ex2b, ex3a, ex3b, ex4a, ex4b)

quadrature_degree = 5

def testA():
    print "TEST A"
    quadrature_degree_old = parameters["form_compiler"]["quadrature_degree"]
      
    # setup problem
    mesh1 = UnitSquare(5, 5)
    V1 = FunctionSpace(mesh1, 'CG', 1)
    mesh2 = UnitSquare(10, 10)
    V2 = FunctionSpace(mesh2, 'CG', 1)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    f = Constant(1.0)

    # define boundary conditions
#    bc = DirichletBC(V, 0.0, "near(x[0], 0.0) || near(x[0], 1.0)")
    def u0_boundary(x, on_boundary):
        return (x[0] <= DOLFIN_EPS - 1.0 or (x[0] >= 0.0 - DOLFIN_EPS and x[1] < 1.0 - DOLFIN_EPS)) and on_boundary
    #    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
    u0 = Constant(0.0)
    bc1 = DirichletBC(V1, u0, u0_boundary)
    bc2 = DirichletBC(V2, u0, u0_boundary)

    # solution vector 
    u1_h = Function(V1)
    u2_h = Function(V2)

    # iterate expressions
    for j, ex in enumerate(EX):
        for qdegree in range(quadrature_degree):
            if qdegree == 0:
                qdegree = -1
            parameters["form_compiler"]["quadrature_degree"] = qdegree

            # forms
#            b1 = f * v1 * dx
#            b2 = f * v2 * dx
#            b1 = ex * v1 * dx
#            b2 = ex * v2 * dx
            b1 = inner(nabla_grad(ex), nabla_grad(ex)) * v1 * dx
            b2 = inner(nabla_grad(ex), nabla_grad(ex)) * v2 * dx

            a1 = inner(nabla_grad(u1), nabla_grad(v1)) * dx
            a2 = inner(nabla_grad(u2), nabla_grad(v2)) * dx
#            a1 = ex * inner(nabla_grad(u1), nabla_grad(v1)) * dx
#            a2 = ex * inner(nabla_grad(u2), nabla_grad(v2)) * dx
#            a1 = inner(nabla_grad(ex), nabla_grad(ex)) * inner(nabla_grad(u1), nabla_grad(v1)) * dx
#            a2 = inner(nabla_grad(ex), nabla_grad(ex)) * inner(nabla_grad(u2), nabla_grad(v2)) * dx
        
            # compute solution
            solve(a1 == b1, u1_h, bc1)
            solve(a2 == b2, u2_h, bc2)
        
            print "[V1] norms for quad=", qdegree, "ex", j, ":", norm(u1_h, 'L2'), norm(u1_h, 'H1')
            print "[V2] norms for quad=", qdegree, "ex", j, ":", norm(u2_h, 'L2'), norm(u2_h, 'H1')
        print "-------------------------------------------------------------------------"
    parameters["form_compiler"]["quadrature_degree"] = quadrature_degree_old

testA()
