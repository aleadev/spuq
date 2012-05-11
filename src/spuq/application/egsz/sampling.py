from __future__ import division
import os


try:
    from dolfin import (Function, FunctionSpace, Constant, UnitSquare, refine,
                        solve, plot, interactive, project, errornorm)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.application.egsz.adaptive_solver import AdaptiveSolver
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
def get_proj_basis(mesh0, num_mesh_refinements):
    mesh = refine(mesh0)
    for i in range(num_mesh_refinements):
        mesh = refine(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    return FEniCSBasis(V)

def get_projected_sol(w, mu, proj_basis):
    return w.project(w[mu], proj_basis)

def compute_parametric_sample_solution(RV_samples, coeff_field, w, proj_basis):
    Lambda = w.active_indices()
    sample_map, _ = coeff_field.sample_realization(Lambda, RV_samples)

    # sum up (stochastic) solution vector on reference function space wrt samples
    Lambda = w.active_indices()

    sample_sol = sum(get_projected_sol(w, mu, proj_basis) * sample_map[mu] for mu in Lambda)
    return sample_sol

def compute_direct_sample_solution(RV_samples, coeff_field, A, maxm, proj_basis):
    # sum up coefficient field sample
    a = coeff_field.mean_func
    for m in range(maxm):
        if m == 10:
            continue
        print m, RV_samples[m], coeff_field[m][0].cppcode, coeff_field[m][0].A, coeff_field[m][0].m, coeff_field[m][0].n
        a_m = RV_samples[m] * coeff_field[m][0]
        a += a_m

    A = FEMPoisson.assemble_lhs(a, proj_basis)
    b = FEMPoisson.assemble_rhs(Constant("1.0"), proj_basis)
    X = 0 * b
    solve(A, X, b)
    return FEniCSVector(Function(proj_basis._fefs, X)), a

