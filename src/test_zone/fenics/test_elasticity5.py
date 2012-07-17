# source: https://answers.launchpad.net/dolfin/+question/200555
import numpy as np
import dolfin as df

def main():
    L = 10.0
    H = 10.0

    mesh = df.UnitSquare(10,10,'left')
    mesh.coordinates()[:,0] *= L
    mesh.coordinates()[:,1] *= H

    U = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
    U_x, U_y = U.split()
    u = df.TrialFunction(U)
    v = df.TestFunction(U)

    E = 2.0E11
    nu = 0.3

    lmbda = nu*E/((1.0 + nu)*(1.0 - 2.0*nu))
    mu = E/(2.0*(1.0 + nu))

    # Elastic Modulus
    C_numpy = np.array([[lmbda + 2.0*mu, lmbda, 0.0],
                        [lmbda, lmbda + 2.0*mu, 0.0],
                        [0.0, 0.0, mu ]])
    C = df.as_matrix(C_numpy)

    from dolfin import dot, dx, grad, inner, ds

    def eps(u):
        """ Returns a vector of strains of size (3,1) in the Voigt notation
        layout {eps_xx, eps_yy, gamma_xy} where gamma_xy = 2*eps_xy"""
        return df.as_vector([u[i].dx(i) for i in range(2)] +
                            [u[i].dx(j) + u[j].dx(i) for (i,j) in [(0,1)]])

    a = inner(eps(v), C*eps(u))*dx
    A = a

    # Dirichlet BC
    class LeftBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and np.abs(x[0]) < tol

    class RightBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and np.abs(x[0] - self.L) < tol

    class BottomBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and np.abs(x[1]) < tol

    left_boundary = LeftBoundary()
    right_boundary = RightBoundary()
    right_boundary.L = L
    bottom_boundary = BottomBoundary()

    zero = df.Constant(0.0)
    bc_left_Ux = df.DirichletBC(U_x, zero, left_boundary)
    bc_bottom_Uy = df.DirichletBC(U_y, zero, bottom_boundary)
    bcs = [bc_left_Ux, bc_bottom_Uy]

    # Neumann BCs
    t = df.Constant(10000.0)
    boundary_parts = df.EdgeFunction("uint", mesh, 1)
    right_boundary.mark(boundary_parts, 0)
    l = inner(t,v[0])*ds(0)

    u_h = df.Function(U)
    problem = df.LinearVariationalProblem(A, l, u_h, bcs=bcs)
    solver = df.LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "direct"
    solver.solve()

    u_x, u_y = u_h.split()

    stress = df.project(C*eps(u_h), df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3))

    df.plot(u_x)
    df.plot(u_y)
    df.plot(stress[0])
    df.interactive()

if __name__ == "__main__":
    main()
