from spuq.utils.testing import *


try:
    import dolfin
    from dolfin import UnitSquare, FunctionSpace, Expression, interpolate
    from spuq.fem.fenics.fenics_basis import FEniCSBasis
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_operator import FEniCSOperator, FEniCSSolveOperator
    HAVE_FENICS = True
except:
    HAVE_FENICS = False

def homogeneous_bc(V):
    # define boundary conditions
    def u0_boundary(x, on_boundary):
        return on_boundary
    u0 = dolfin.Constant(0.0)
    bc = dolfin.DirichletBC(V, u0, u0_boundary)
    return bc

def assemble_lhs(coeff, basis):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # get FEniCS function space
    V = basis._fefs

    # setup problem, assemble and apply boundary conditions
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    a = dolfin.inner(coeff * dolfin.nabla_grad(u), dolfin.nabla_grad(v)) * dolfin.dx
    A = dolfin.assemble(a)

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
    a = coeff * v * dolfin.dx
    b = dolfin.assemble(a)

    # apply boundary conditions
    bc = homogeneous_bc(V)
    bc.apply(b)
    return b

@skip_if(not HAVE_FENICS)
def test_fenics_vector():
    k1, k2 = 1, 1
    mesh = UnitSquare(15, 15)
    fs = FunctionSpace(mesh, "CG", 2)
    ex1 = Expression("A*sin(k1*pi*x[0])*sin(k2*pi*x[1])", k1=k1, k2=k2, A=1.0)
    x = FEniCSVector(interpolate(ex1, fs))
    print x.coeffs.array()

    pi = 3.14159265358979323
    ex2 = Expression("A*sin(k1*pi*x[0])*sin(k2*pi*x[1])", k1=k1, k2=k2, A=pi * pi * (k1 * k1 + k2 * k2))
    b_ex = assemble_rhs(ex2, x.basis)
    print b_ex.array()
    print b_ex.array() / (2 * pi * pi * x.coeffs.array())

    a = Expression('1')
    M = assemble_lhs(a, x.basis)

    A = FEniCSOperator(M, x.basis)
    b = A * x
    print b.coeffs.array() / x.coeffs.array()
    print b_ex.array() / b.coeffs.array()
    #print b_ex.array() / (M * interpolate(ex1, fs).vector()).array()

    #assert_array_almost_equal(b.coeffs, b_ex.coeffs)
    b2 = b.copy()
    print b
    print b2

    b.coeffs = b_ex

    #dolfin.plot(x._fefunc, title="x")
    dolfin.plot(b._fefunc, title="b", rescale=False, axes=True)
    dolfin.plot(b2._fefunc, title="b_ex", rescale=False, axes=True)
    print dolfin.errornorm(u=b._fefunc, uh=b2._fefunc) #, norm_type, degree, mesh)
    dolfin.interactive()


test_main()
