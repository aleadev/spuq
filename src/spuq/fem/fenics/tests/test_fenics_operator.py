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


def assemble_lhs(coeff, basis):
    """Assemble the discrete problem (i.e. the stiffness matrix)."""
    # get FEniCS function space
    V = basis._fefs

    # define boundary conditions
    def u0_boundary(x, on_boundary):
        return on_boundary
    u0 = dolfin.Constant(0.0)
    bc = dolfin.DirichletBC(V, u0, u0_boundary)

    # setup problem, assemble and apply boundary conditions
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    a = dolfin.inner(coeff * dolfin.nabla_grad(u), dolfin.nabla_grad(v)) * dolfin.dx
    A = dolfin.assemble(a)
    bc.apply(A)
    return A


@skip_if(not HAVE_FENICS)
def test_fenics_vector():
    k1, k2 = 1, 1
    mesh = UnitSquare(62, 62)
    fs = FunctionSpace(mesh, "CG", 1)
    ex = Expression("A*sin(k1*pi*x[0])*sin(k2*pi*x[1])", k1=k1, k2=k2, A=1.0)
    x = FEniCSVector(interpolate(ex, fs))
    print x.coeffs.array()

    pi = 3.14159265358979323
    ex = Expression("A*sin(k1*pi*x[0])*sin(k2*pi*x[1])", k1=k1, k2=k2, A= -pi * pi * (k1 * k1 + k2 * k2))
    b_ex = FEniCSVector(interpolate(ex, fs))
    print b_ex.coeffs.array()
    print b_ex.coeffs.array() / (-2 * pi * pi * x.coeffs.array())

    a = Expression('1')
    M = assemble_lhs(a, x.basis)

    A = FEniCSOperator(M, x.basis)
    b = A * x
    print b.coeffs.array() / x.coeffs.array()
    print b_ex.coeffs.array() / b.coeffs.array()

    #assert_array_almost_equal(b.coeffs, b_ex.coeffs)

    #dolfin.plot(x._fefunc, title="x")
    dolfin.plot(b._fefunc, title="b")
    dolfin.plot((-b_ex)._fefunc, title="b_ex")
    dolfin.interactive()


test_main()
