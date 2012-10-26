from dolfin import *
import numpy as np
import scipy.linalg as la

np.set_printoptions(suppress=True, linewidth=1000, precision=3, edgeitems=20)

def remove_boundary_entries(A, bc):
    from dolfin.cpp import _set_matrix_single_item
    dofs = bc.get_boundary_values().keys()
    values = np.zeros(1, dtype=np.float_)
    rows = np.array([0], dtype=np.uintc)
    cols = np.array([0], dtype=np.uintc)
    A.apply("insert")
    for d in dofs:
        rows[0] = d
        cols[0] = d
        A.set(values, rows, cols)
    A.apply("insert")


def prepare_poisson():
    N = 4
    mesh = UnitSquare(N, N)
    #mesh = UnitInterval(N)
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = 100 * inner(nabla_grad(u), nabla_grad(v)) * dx
    f = Constant(1.0000)
    L = f * v * dx
    
    def u0_boundary(x, on_boundary):
        return on_boundary
    u0 = Expression("10*(x[0]-0.5)")
    bc = DirichletBC(V, u0, u0_boundary)

    return a, L, (bc,)


def prepare_elasticity():
    N = 4
    mesh = UnitSquare(N, N)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    
    # Sub domain for clamp at left end
    def left(x, on_boundary):
        return x[0] <= 0.0 and on_boundary
    
    # Sub domain for rotation at right end
    def right(x, on_boundary):
        return x[0] >= 1.0 and on_boundary
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0.0, 0.0))
    
    E = 1e3
    nu = 0.4
    
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    print "mu=%s, lambda=%s" % (mu, lmbda)
    sigma = lambda v: 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(v.cell().d)
    
    a = inner(sigma(u), grad(v)) * dx
    L = inner(f, v) * dx
    
    # Set up boundary condition at left side
    bcl = DirichletBC(V, Constant((0, 0)), left)
    
    # Set up boundary condition at right side
    bcr = DirichletBC(V, Constant((1, 1)), right)
    
    # Set up boundary conditions
    bcs = [bcl, bcr]

    return a, L, bcs


#a, L, bcs = prepare_poisson()
a, L, bcs = prepare_elasticity()

A0 = assemble(a).array()
b0 = assemble(L).array()

for bc in bcs:
    dofs = bc.get_boundary_values().keys()
    vals = bc.get_boundary_values().values()
    print dofs
    
    g0 = b0 * 0
    g0[dofs] = vals
    
    n = np.size(A0, 1)
    I = np.eye(n)
    I_I = I + 0
    I_I[dofs, dofs] = 0
    I_B = I - I_I
    
    AAA = assemble_system(a, L, bc)[0].array()
    D_B = AAA * I_B
    #print D_B
    
    b2 = np.dot(I_I, b0) + np.dot(D_B, g0) - np.dot(I_I, np.dot(A0, g0))
    print b2
    print
    A2 = np.dot(I_I, np.dot(A0, I_I)) + D_B
    
    A, b = assemble_system(a, L, bc)
    A = A.array()
    b = b.array()
    
    print np.array([b0, b, b2]).T
    print
    
    print la.norm(A - A2), la.norm(b - b2)
    print
