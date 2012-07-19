from __future__ import division
import os


try:
    from dolfin import (Function, VectorFunctionSpace, FunctionSpace, Constant, refine,
                        solve, plot, interactive, project, errornorm)
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_basis import FEniCSBasis
except Exception, e:
    import traceback

    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)



# setup initial multivector
def setup_vector(mesh, pde, degree=1):
#    fs = FunctionSpace(mesh, "CG", degree)
    fs = pde.function_space(mesh, degree=degree)
    vec = FEniCSVector(Function(fs))
    return vec


# create reference mesh and function space
def get_projection_basis(mesh0, mesh_refinements=None, maxh=None, degree=1, sub_spaces=None, family=None):
    if family is None:
        family = 'CG'
    if not(mesh_refinements is None):
        mesh = mesh0
        for _ in range(mesh_refinements):
            mesh = refine(mesh)
        if sub_spaces is None or sub_spaces == 0:
            V = FunctionSpace(mesh, family, degree)
        else:
            V = VectorFunctionSpace(mesh, family, degree)
            assert V.num_sub_spaces() == sub_spaces
        return FEniCSBasis(V)
    else:
        assert not(maxh is None)
        if sub_spaces is None or sub_spaces == 0:
            V = FunctionSpace(mesh0, family, degree)
        else:
            V = VectorFunctionSpace(mesh0, family, degree)
            assert V.num_sub_spaces() == sub_spaces
        B = FEniCSBasis(V)
        return B.refine_maxh(maxh, True)


def get_projected_solution(w, mu, proj_basis):
    # TODO: obfuscated method call since project is not obvious in the interface of MultiVector!
#    print "sampling.get_projected_solution"
#    print w[mu].num_sub_spaces
#    print proj_basis.num_sub_spaces
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


def compute_direct_sample_solution(pde, RV_samples, coeff_field, A, f, maxm, proj_basis, Dirichlet_boundary=lambda x, on_boundary: on_boundary):
    try:
        A0 = coeff_field.A
        A_m = coeff_field.A_m
    except AttributeError:
        a = coeff_field.mean_func
        A0 = pde.assemble_lhs(a, proj_basis, withBC=False)
        A_m = [None] * maxm
        coeff_field.A = A0
        coeff_field.A_m = A_m

    A = A0.copy()
    for m in range(maxm):
        if A_m[m] is None:
            a_m = coeff_field[m][0]
            A_m[m] = pde.assemble_lhs(a_m, proj_basis, withBC=False)
        A += RV_samples[m] * A_m[m]

    b = pde.assemble_rhs(f, proj_basis, withBC=False)
    A, b = pde.apply_dirichlet_bc(proj_basis, A, b, Dirichlet_boundary=Dirichlet_boundary)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X))


def compute_direct_sample_solution_old(pde, RV_samples, coeff_field, A, f, maxm, proj_basis, Dirichlet_boundary=lambda x, on_boundary: on_boundary):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m

    A = pde.assemble_lhs(a, proj_basis, Dirichlet_boundary=Dirichlet_boundary)
    b = pde.assemble_rhs(f, proj_basis, Dirichlet_boundary=Dirichlet_boundary)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), a


def get_coeff_realisation(RV_samples, coeff_field, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m
    return FEniCSVector(project(a, proj_basis._fefs))
