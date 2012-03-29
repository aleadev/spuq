import numpy as np
from spuq.utils.testing import *


try:
    import dolfin
    from dolfin import (UnitSquare, FunctionSpace, Expression, interpolate, dx,
                        inner, nabla_grad, TrialFunction, TestFunction,
                        assemble, Constant, DirichletBC, Mesh, PETScMatrix,
                        SLEPcEigenSolver, Function, solve)
    from spuq.fem.fenics.fenics_basis import FEniCSBasis
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
    HAVE_FENICS = True
    # Test for PETSc and SLEPc
    HAVE_SLEPC = dolfin.has_linear_algebra_backend("PETSc") and dolfin.has_slepc()
    TestFunction = no_test(TestFunction)
except:
    HAVE_FENICS = False


def homogeneous_bc(V):
    # define boundary conditions
    def u0_boundary(x, on_boundary):
        return on_boundary
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, u0_boundary)
    return bc

def assemble_lhs(coeff, V):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # setup problem, assemble and apply boundary conditions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
    A = assemble(a)

    # apply boundary conditions
    bc = homogeneous_bc(V)
    bc.apply(A)
    return A

def assemble_gramian(basis):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # get FEniCS function space
    V = basis._fefs

    # setup problem, assemble and apply boundary conditions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u * v * dx
    G = assemble(a)
    return G

def assemble_rhs(coeff, V):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # setup problem, assemble and apply boundary conditions
    v = dolfin.TestFunction(V)
    a = coeff * v * dx
    b = dolfin.assemble(a)

    # apply boundary conditions
    bc = homogeneous_bc(V)
    bc.apply(b)
    return b

def evaluate_evp(basis):
    """Evaluate EVP"""
    assert HAVE_SLEPC
    # get FEniCS function space
    V = basis._fefs

    # Define basis and bilinear form
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(nabla_grad(u), nabla_grad(v)) * dx

    # Assemble stiffness form
    A = PETScMatrix()
    assemble(a, tensor=A)

    # Create eigensolver
    eigensolver = SLEPcEigenSolver(A)

    # Compute all eigenvalues of A x = \lambda x
    print "Computing eigenvalues..."
    eigensolver.solve()
    return eigensolver

@skip_if(not HAVE_FENICS)
def teXXXst_fenics_vector():
#    quad_degree = 13
#    dolfin.parameters["form_compiler"]["quadrature_degree"] = quad_degree
    pi = 3.14159265358979323
    k1, k2 = 2, 3
    EV = pi * pi * (k1 * k1 + k2 * k2)
    N = 11
    degree = 1
    mesh = UnitSquare(N, N)
    fs = FunctionSpace(mesh, "CG", degree)
    ex = Expression("A*sin(k1*pi*x[0])*sin(k2*pi*x[1])", k1=k1, k2=k2, A=1.0)

    x = FEniCSVector(interpolate(ex, fs))
#    print "x.coeff", x.coeffs.array()

    ex.A = EV
    b_ex = assemble_rhs(ex, fs)
    bexg = interpolate(ex, fs)

#    print b_ex.array()
#    print b_ex.array() / (2 * pi * pi * x.coeffs.array())

    Afe = assemble_lhs(Expression('1'), fs)

    # apply discrete operator on (interpolated) x
    A = FEniCSOperator(Afe, x.basis)
    b = A * x

    # evaluate solution for eigenfunction rhs
    if False:
        b_num = Function(fs)
        solve(A, b_num.vector(), b_ex)
        bnv = A * b_num.vector()
        b3 = Function(fs, bnv / EV)

    np.set_printoptions(threshold='nan', suppress=True)
    print b.coeffs.array()
    print np.abs((b_ex.array() - b.coeffs.array()) / np.max(b_ex.array()))
    print np.max(np.abs((b_ex.array() - b.coeffs.array()) / np.max(b_ex.array())))
    #print b_ex.array() / (M * interpolate(ex1, fs).vector()).array()

#    #assert_array_almost_equal(b.coeffs, b_ex.coeffs)


    b2 = Function(fs, b_ex.copy())
    bg = Function(fs, b_ex.copy())
    b2g = Function(fs, b_ex.copy())
    G = assemble_gramian(x.basis)
    dolfin.solve(G, bg.vector(), b.coeffs)
    dolfin.solve(G, b2g.vector(), b2.vector())


#    # compute eigenpairs numerically
#    eigensolver = evaluate_evp(FEniCSBasis(fs))
#    # Extract largest (first) eigenpair
#    r, c, rx, cx = eigensolver.get_eigenpair(0)    
#    print "Largest eigenvalue: ", r    
#    # Initialize function and assign eigenvector
#    ef0 = Function(fs)
#    ef0.vector()[:] = rx

    if False:
        # export
        out_b = dolfin.File(__name__ + "_b.pvd", "compressed")
        out_b << b._fefunc
        out_b_ex = dolfin.File(__name__ + "_b_ex.pvd", "compressed")
        out_b_ex << b2
        out_b_num = dolfin.File(__name__ + "_b_num.pvd", "compressed")
        out_b_num << b_num


    #dolfin.plot(x._fefunc, title="interpolant x", rescale=False, axes=True, legend=True)
    dolfin.plot(bg, title="b", rescale=False, axes=True, legend=True)
    dolfin.plot(b2g, title="b_ex (ass/G)", rescale=False, axes=True, legend=True)
    dolfin.plot(bexg, title="b_ex (dir)", rescale=False, axes=True, legend=True)
    #dolfin.plot(b_num, title="b_num", rescale=False, axes=True, legend=True)
#    dolfin.plot(b3, title="M*b_num", rescale=False, axes=True, legend=True)
    #dolfin.plot(ef0, title="ef0", rescale=False, axes=True, legend=True)
    print dolfin.errornorm(u=b._fefunc, uh=b2) #, norm_type, degree, mesh)
    dolfin.interactive()

def sample_problem():
    k1, k2 = 2, 3
    EV = np.pi * np.pi * (k1 * k1 + k2 * k2)
    N = 21
    degree = 2
    mesh = UnitSquare(N, N)
    fs = FunctionSpace(mesh, "CG", degree)
    ex = Expression("amp*sin(k1*pi*x[0])*sin(k2*pi*x[1])", k1=k1, k2=k2, amp=1.0)
    return fs, ex, EV

@skip_if(not HAVE_FENICS)
def test_fenics_operator():
    fs, ex, EV = sample_problem()

    basis = FEniCSBasis(fs)
    x_fe = interpolate(ex, fs)
    A_fe = assemble_lhs(Expression('1'), fs)

    A = FEniCSOperator(A_fe, basis)
    x = FEniCSVector(x_fe)
    ex.amp = EV
    b_ex_fe = assemble_rhs(ex, fs)

    # apply discrete operator on (interpolated) x
    b = A * x
    b_fe = b.coeffs

    assert_array_almost_equal(b_fe, b_ex_fe, decimal=2)


@skip_if(not HAVE_FENICS)
def test_fenics_solve_operator():
    fs, ex, EV = sample_problem()

    basis = FEniCSBasis(fs)
    x_ex_fe = interpolate(ex, fs)
    A_fe = assemble_lhs(Expression('1'), fs)

    Ainv = FEniCSSolveOperator(A_fe, basis)
    ex.amp = EV
    b_fe = assemble_rhs(ex, fs)
    b = FEniCSVector(Function(fs, b_fe))

    # apply discrete operator on (interpolated) x
    x = Ainv * b
    x_fe = x.coeffs

    assert_array_almost_equal(x_fe, x_ex_fe.vector(), decimal=2)



test_main()
