from __future__ import division
import os
import logging

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

# module logger
logger = logging.getLogger(__name__)

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


def compute_parametric_sample_solution(RV_samples, coeff_field, w, proj_basis, cache=None):
    Lambda = w.active_indices()
    sample_map, _ = coeff_field.sample_realization(Lambda, RV_samples)
    # sum up (stochastic) solution vector on reference function space wrt samples
    
    if cache is None:
        sample_sol = sum(get_projected_solution(w, mu, proj_basis) * sample_map[mu] for mu in Lambda)
    else:
        try:
            projected_sol = cache.projected_sol
        except AttributeError:
            projected_sol = {mu: get_projected_solution(w, mu, proj_basis) for mu in Lambda}
            cache.projected_sol = projected_sol
        sample_sol = sum(projected_sol[mu] * sample_map[mu] for mu in Lambda)
    return sample_sol


def compute_direct_sample_solution(pde, RV_samples, coeff_field, A, maxm, proj_basis, cache=None):
    try:
        A0 = cache.A
        A_m = cache.A_m
        b = cache.b
    except AttributeError:
        a = coeff_field.mean_func
        A0 = pde.assemble_lhs(a, proj_basis, withBC=False)
        b = pde.assemble_rhs(proj_basis, withBC=False)
        A_m = [None] * maxm
        cache.A = A0
        cache.A_m = A_m
        cache.b = b

    from spuq.utils.timing import timing
    with timing(msg="direct AM", logfunc=logger.info):
        A = A0.copy()
        for m in range(maxm):
            if A_m[m] is None:
                a_m = coeff_field[m][0]
                A_m[m] = pde.assemble_lhs(a_m, proj_basis, withBC=False)
            A += RV_samples[m] * A_m[m]

    with timing(msg="direct BC", logfunc=logger.info):
        A, b = pde.apply_dirichlet_bc(proj_basis, A, b)
    with timing(msg="direct Solve", logfunc=logger.info):
        X = 0 * b
        solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X))


def compute_direct_sample_solution_old(pde, RV_samples, coeff_field, A, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m

    A = pde.assemble_lhs(a, proj_basis)
    b = pde.assemble_rhs(proj_basis)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), a


def get_coeff_realisation(RV_samples, coeff_field, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m
    return FEniCSVector(project(a, proj_basis._fefs))
