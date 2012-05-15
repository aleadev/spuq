from __future__ import division
import os


try:
    from dolfin import (Function, FunctionSpace, Constant, refine,
                        solve, plot, interactive, project, errornorm)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_basis import FEniCSBasis
except Exception, e:
    import traceback

    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)



# setup initial multivector
def setup_vec(mesh):
    fs = FunctionSpace(mesh, "CG", 1)
    vec = FEniCSVector(Function(fs))
    return vec


# create reference mesh and function space
def get_proj_basis(mesh0, mesh_refinements=None, maxh=None, degree=1):
    if not(mesh_refinements is None):
        mesh = mesh0
        for _ in range(mesh_refinements):
            mesh = refine(mesh)
        V = FunctionSpace(mesh, "CG", degree)
        return FEniCSBasis(V)
    else:
        assert not(maxh is None)
        B = FEniCSBasis(FunctionSpace(mesh0, "CG", degree))
        return B.refine_maxh(maxh, True)


def get_projected_sol(w, mu, proj_basis):
    return w.project(w[mu], proj_basis)


def compute_parametric_sample_solution(RV_samples, coeff_field, w, proj_basis):
    Lambda = w.active_indices()
    sample_map, _ = coeff_field.sample_realization(Lambda, RV_samples)

    # sum up (stochastic) solution vector on reference function space wrt samples
    Lambda = w.active_indices()

    sample_sol = sum(get_projected_sol(w, mu, proj_basis) * sample_map[mu] for mu in Lambda)
    return sample_sol


def compute_direct_sample_solution_old(RV_samples, coeff_field, A, f, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a += a_m

    A = FEMPoisson.assemble_lhs(a, proj_basis)
    b = FEMPoisson.assemble_rhs(f, proj_basis)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), a


def compute_direct_sample_solution(RV_samples, coeff_field, A, f, maxm, proj_basis):
    try:
        A = coeff_field.A
        A_m = coeff_field.A_m
    except AttributeError:
        a = coeff_field.mean_func
        A = FEMPoisson.assemble_lhs(a, proj_basis)
        A_m = [None] * maxm
        coeff_field.A = A
        coeff_field.A_m = A_m

    for m in range(maxm):
        if A_m[m] is None:
            a_m = coeff_field[m][0]
            A_m[m] = FEMPoisson.assemble_lhs(a_m, proj_basis)
        A += RV_samples[m] * A_m[m]

    b = FEMPoisson.assemble_rhs(f, proj_basis)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), coeff_field.mean_func
