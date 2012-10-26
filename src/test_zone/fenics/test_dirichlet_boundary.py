from dolfin import *
import numpy as np
import scipy.linalg as la

np.set_printoptions(suppress=True, linewidth=1000, precision=3, edgeitems=20)


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
    N = 1
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
    bcr = DirichletBC(V, Constant((1, 2)), right)
    
    # Set up boundary conditions
    bcs = [bcl, bcr]

    return a, L, bcs


#a, L, bcs = prepare_poisson()
a, L, bcs = prepare_elasticity()


def assem1():
    A, b = assemble_system(a, L, bcs)
    A = A.array()
    b = b.array()
    return A, b

def assem2():
    A = assemble(a).array()
    b = assemble(L).array()

    for bc in bcs:
        dofs = bc.get_boundary_values().keys()
        vals = bc.get_boundary_values().values()
        print dofs
    
        g0 = b * 0
        g0[dofs] = vals
        
        n = np.size(A, 1)
        I = np.eye(n)
        I_I = I + 0
        I_I[dofs, dofs] = 0
        I_B = I - I_I
    
        AAA = assemble_system(a, L, bc)[0].array()
        D_B = AAA * I_B
    
        b = np.dot(I_I, b) + np.dot(D_B, g0) - np.dot(I_I, np.dot(A, g0))
        A = np.dot(I_I, np.dot(A, I_I)) + D_B
    return A, b

def assem3():
    A = assemble(a).array()
    b = assemble(L).array()

    dofs = sum([bc.get_boundary_values().keys() for bc in bcs], [])
    vals = sum([bc.get_boundary_values().values() for bc in bcs], [])
    print dofs

    g0 = b * 0
    g0[dofs] = vals
    
    n = np.size(A, 1)
    I = np.eye(n)
    I_I = I + 0
    I_I[dofs, dofs] = 0
    I_B = I - I_I
    
    AAA = assemble_system(a, L, bcs)[0].array()
    D_B = AAA * I_B

    b = np.dot(I_I, b) + np.dot(D_B, g0) - np.dot(I_I, np.dot(A, g0))
    A = np.dot(I_I, np.dot(A, I_I)) + D_B
    return A, b

    
A1, b1 = assem1()
A2, b2 = assem2()
A3, b3 = assem3()

print la.norm(A1 - A2), la.norm(b1 - b2)
print la.norm(A1 - A3), la.norm(b1 - b3)
print
