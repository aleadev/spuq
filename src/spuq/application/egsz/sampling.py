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
def setup_vector(mesh, degree=1):
    fs = FunctionSpace(mesh, "CG", degree)
    vec = FEniCSVector(Function(fs))
    return vec


# create reference mesh and function space
def get_projection_basis(mesh0, mesh_refinements=None, maxh=None, degree=1):
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


def get_projected_solution(w, mu, proj_basis):
    # TODO: obfuscated method call since project is not obvious in the interface of MultiVector!
    return w.project(w[mu], proj_basis)


def prepare_w_projections(w, proj_basis):
    return {mu:get_projected_solution(w, mu, proj_basis) for mu in w.active_indices()}


def compute_parametric_sample_solution(RV_samples, coeff_field, w, proj_basis, proj_cache=None):
    Lambda = w.active_indices()
    sample_map, _ = coeff_field.sample_realization(Lambda, RV_samples)
    # sum up (stochastic) solution vector on reference function space wrt samples
    Lambda = w.active_indices()
    if proj_cache is None:
        sample_sol = sum(get_projected_solution(w, mu, proj_basis) * sample_map[mu] for mu in Lambda)
    else:
        sample_sol = sum(proj_cache[mu] * sample_map[mu] for mu in Lambda)
    return sample_sol


def compute_direct_sample_solution(RV_samples, coeff_field, A, f, maxm, proj_basis):
    try:
        A0 = coeff_field.A
        A_m = coeff_field.A_m
    except AttributeError:
        a = coeff_field.mean_func
        A0 = FEMPoisson.assemble_lhs(a, proj_basis, withBC=False)
        A_m = [None] * maxm
        coeff_field.A = A0
        coeff_field.A_m = A_m

    A = A0.copy()
    for m in range(maxm):
        if A_m[m] is None:
            a_m = coeff_field[m][0]
            A_m[m] = FEMPoisson.assemble_lhs(a_m, proj_basis, withBC=False)
        A += RV_samples[m] * A_m[m]

    b = FEMPoisson.assemble_rhs(f, proj_basis, withBC=False)
    A, b = FEMPoisson.apply_dirichlet_bc(proj_basis, A, b)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X))


def compute_direct_sample_solution_old(RV_samples, coeff_field, A, f, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m

    A = FEMPoisson.assemble_lhs(a, proj_basis)
    b = FEMPoisson.assemble_rhs(f, proj_basis)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), a


def get_coeff_realisation(RV_samples, coeff_field, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", m, RV_samples[m] 
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m
    return FEniCSVector(project(a, proj_basis._fefs))
