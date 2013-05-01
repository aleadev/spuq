from __future__ import division
import os
import logging
from spuq.utils.timing import timing

from spuq.utils.timing import timing

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
def get_projection_basis(mesh0, mesh_refinements=None, maxh=None, degree=1, sub_spaces=None, family='CG'):
    if mesh_refinements is not None:
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
        assert maxh is not None
        if sub_spaces is None or sub_spaces == 0:
            V = FunctionSpace(mesh0, family, degree)
        else:
            V = VectorFunctionSpace(mesh0, family, degree)
            assert V.num_sub_spaces() == sub_spaces
        B = FEniCSBasis(V)
        return B.refine_maxh(maxh, True)


def get_projected_solution(w, mu, proj_basis):
    # TODO: obfuscated method call since project is not obvious in the interface of MultiVector! This should be separated more clearly!
#    print "sampling.get_projected_solution"
#    print w[mu].num_sub_spaces
#    print proj_basis.num_sub_spaces
    with timing(msg="get_projected_solution (%s --- %i)" % (str(mu), w[mu].dim), logfunc=logger.debug):
        w_proj = w.project(w[mu], proj_basis)
    return w_proj


def prepare_w_projections(w, proj_basis):
    return {mu:get_projected_solution(w, mu, proj_basis) for mu in w.active_indices()}


def compute_parametric_sample_solution(RV_samples, coeff_field, w, proj_basis, cache=None):
    with timing(msg="parametric_sample_sol", logfunc=logger.info):
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


def compute_solution_variance(coeff_field, w, proj_basis):
    Lambda = w.active_indices()
    # sum up (stochastic) solution vector on reference function space wrt samples

    sample_sol = 0
    for mu in Lambda:
        if mu.order == 0:
            continue
        w_mu = get_projected_solution(w, mu, proj_basis)
        f = w_mu._fefunc
        #print "1>" + str(list(f.vector().array()))
        f.vector()[:] = (f.vector().array()) ** 2
        #print "2>" + str(list(f.vector().array()))
        #f2 = project(f*f, w_mu.basis._fefs)
        #sample_sol += FEniCSVector(f2)
        sample_sol += FEniCSVector(f)
    return sample_sol


def compute_direct_sample_solution(pde, RV_samples, coeff_field, A, maxm, proj_basis, cache=None):
    try:
        A0 = cache.A
        A_m = cache.A_m
        b = cache.b
        print "CACHE USED"
    except AttributeError:
        with timing(msg="direct_sample_sol: compute A_0, b", logfunc=logger.info):
            a = coeff_field.mean_func
            A0 = pde.assemble_lhs(basis=proj_basis, coeff=a, withDirichletBC=False)
            b = pde.assemble_rhs(basis=proj_basis, coeff=a, withDirichletBC=False)
            A_m = [None] * maxm
            print "CACHE NOT USED"
        if cache is not None:
            cache.A = A0
            cache.A_m = A_m
            cache.b = b

    with timing(msg="direct_sample_sol: compute A_m", logfunc=logger.info):
        A = A0.copy()
        for m in range(maxm):
            if A_m[m] is None:
                a_m = coeff_field[m][0]
                A_m[m] = pde.assemble_lhs(basis=proj_basis, coeff=a_m, withDirichletBC=False)
            A += RV_samples[m] * A_m[m]

    with timing(msg="direct_sample_sol: apply BCs", logfunc=logger.info):
        A, b = pde.apply_dirichlet_bc(proj_basis._fefs, A, b)

    with timing(msg="direct_sample_sol: solve linear system", logfunc=logger.info):
        X = 0 * b
        logger.info("compute_direct_sample_solution with %i dofs" % b.size())
        solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X))


def compute_direct_sample_solution_old(pde, RV_samples, coeff_field, A, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m

    A = pde.assemble_lhs(basis=proj_basis, coeff=a)
    b = pde.assemble_rhs(basis=proj_basis, coeff=a)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), a


def get_coeff_realisation(RV_samples, coeff_field, maxm, proj_basis):
    a = coeff_field.mean_func
    for m in range(maxm):
        a_m = RV_samples[m] * coeff_field[m][0]
        a = a + a_m
    return FEniCSVector(project(a, proj_basis._fefs))
