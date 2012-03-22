from spuq.utils.testing import *


try:
    import dolfin
    from dolfin import (UnitSquare, FunctionSpace, Expression, interpolate, dx, inner, nabla_grad, TrialFunction, TestFunction,
                            assemble, Constant, DirichletBC, Mesh, PETScMatrix, SLEPcEigenSolver, Function, solve)
    from spuq.fem.fenics.fenics_basis import FEniCSBasis
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
    HAVE_FENICS = True
    # Test for PETSc and SLEPc
    HAVE_SLEPC = dolfin.has_linear_algebra_backend("PETSc") and dolfin.has_slepc()
except:
    HAVE_FENICS = False


def homogeneous_bc(V):
    # define boundary conditions
    def u0_boundary(x, on_boundary):
        return on_boundary
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, u0_boundary)
    return bc

def assemble_lhs(coeff, basis):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # get FEniCS function space
    V = basis._fefs

    # setup problem, assemble and apply boundary conditions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(coeff * nabla_grad(u), nabla_grad(v)) * dx
    A = assemble(a)

    # apply boundary conditions
    bc = homogeneous_bc(V)
    bc.apply(A)
    return A

def assemble_rhs(coeff, basis):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # get FEniCS function space
    V = basis._fefs

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
def test_fenics_vector():
    k1, k2 = 1, 1
    pi = 3.14159265358979323
    mesh = UnitSquare(3, 3)
    fs = FunctionSpace(mesh, "CG", 2)
    ex = Expression("A*sin(k1*pi*x[0])*sin(k2*pi*x[1])", degree=3, k1=k1, k2=k2, A=1.0)
    x = FEniCSVector(interpolate(ex, fs))
    print "x.coeff", x.coeffs.array()

    ex.A = pi * pi * (k1 * k1 + k2 * k2)
    b_ex = assemble_rhs(ex, x.basis)
    print b_ex.array()
    print b_ex.array() / (2 * pi * pi * x.coeffs.array())

    a = Expression('1')
    M = assemble_lhs(a, x.basis)

    # apply discrete operator on (interpolated) x
    A = FEniCSOperator(M, x.basis)
    b = A * x
    
    # evaluate solution for eigenfunction rhs
    b_num = Function(fs)
    solve(M, b_num.vector(), b_ex)
    

    print b.coeffs.array() / x.coeffs.array()
    print b_ex.array() / b.coeffs.array()
    #print b_ex.array() / (M * interpolate(ex1, fs).vector()).array()

#    #assert_array_almost_equal(b.coeffs, b_ex.coeffs)
    b2 = Function(fs, b_ex)

#    # compute eigenpairs numerically
#    eigensolver = evaluate_evp(FEniCSBasis(fs))
#    # Extract largest (first) eigenpair
#    r, c, rx, cx = eigensolver.get_eigenpair(0)    
#    print "Largest eigenvalue: ", r    
#    # Initialize function and assign eigenvector
#    ef0 = Function(fs)
#    ef0.vector()[:] = rx

    #dolfin.plot(x._fefunc, title="x")
    dolfin.plot(b._fefunc, title="b", rescale=False, axes=True)
    dolfin.plot(b2, title="b_ex", rescale=False, axes=True)
    dolfin.plot(b_num, title="b_num", rescale=False, axes=True)
#    dolfin.plot(ef0, title="ef0", rescale=False, axes=True)
    print dolfin.errornorm(u=b._fefunc, uh=b2) #, norm_type, degree, mesh)
    dolfin.interactive()


test_main()
