from dolfin import *
import numpy as np
import scipy.linalg as la

np.set_printoptions(suppress=True, linewidth=1000, precision=3, edgeitems=20)

def remove_boundary_entries(A, bc):
    from dolfin.cpp import _set_matrix_single_item
    dofs = bc.get_boundary_values().keys()
    for i in dofs:
        _set_matrix_single_item(A, i, i, 0.0)



N = 4
mesh = UnitSquare(N, N)
mesh = UnitInterval(N)
V = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = 100*inner(nabla_grad(u), nabla_grad(v)) * dx
f = Constant(1.0000)
L = f * v * dx


def u0_boundary(x, on_boundary):
    return on_boundary
u0 = Expression("10*(x[0]-0.5)")
bc = DirichletBC(V, u0, u0_boundary)

dofs = bc.get_boundary_values().keys()
vals = bc.get_boundary_values().values()
print dofs

A0 = assemble(a).array()
b0 = assemble(L).array()
g0 = b0 * 0
g0[dofs] = vals

n = np.size(A0,1)
I = np.eye(n)
I_I = I+0
I_I[dofs,dofs]=0
I_B=I-I_I

AAA = assemble_system(a, L, bc)[0].array()
D_B=AAA*I_B
#print D_B

b2 = np.dot(I_I,b0) + np.dot(D_B,g0)-np.dot(I_I,np.dot(A0,g0))
print b2
print
A2 = np.dot(I_I,np.dot(A0,I_I)) + D_B

A, b = assemble_system(a, L, bc)
A=A.array()
b=b.array()

print np.array([b0, b, b2]).T
print

print la.norm(A-A2), la.norm(b-b2)
print

