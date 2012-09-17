from dolfin import *
import numpy as np
import scipy.linalg as la

np.set_printoptions(linewidth=10000)

N = 3
mesh = UnitSquare(N, N)
V = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = 100*inner(nabla_grad(u), nabla_grad(v)) * dx
f = Constant(10000)
L = f * v * dx


def u0_boundary(x, on_boundary):
    return on_boundary
u0 = Expression("x[0]")
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
print D_B

b2 = np.dot(I_I,b0) + np.dot(D_B,g0)-np.dot(I_I,np.dot(A0,g0))
print b2
A2 = np.dot(I_I,np.dot(A0,I_I)) + D_B


A, b = assemble_system(a, L, bc)
A=A.array()
b=b.array()

print np.array([b, b0, b2]).T

print la.norm(A-A2), la.norm(b-b2)
print A2


if False:
    print (A+0.001).astype("int")
    print (10*b).astype("int")
    print la.norm(A), la.norm(A)**2
    print



def remove_boundary_entries(A, bc):
    from dolfin.cpp import _set_matrix_single_item
    dofs = bc.get_boundary_values().keys()
    for i in dofs:
        _set_matrix_single_item(A, i, i, 0.0)
        


