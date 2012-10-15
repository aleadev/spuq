from __future__ import division
import logging
import os
import functools
import numpy as np
from math import sqrt

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.adaptive_solver import setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator, ASSEMBLY_TYPE
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.application.egsz.adaptive_solver import prepare_rhs, pcg_solve
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.math_utils.multiindex import Multiindex
from spuq.utils.plot.plotter import Plotter
from spuq.application.egsz.egsz_utils import setup_logging
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
    from spuq.application.egsz.fem_discretisation import FEMNavierLame
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

LOG_LEVEL = logging.INFO
logger = setup_logging(LOG_LEVEL)

# ------------------------------------------------------------

# ============================================================
# PART A: Simulation Options
# ============================================================

# domain
domain = 'square'

# initial mesh elements
#initial_mesh_N = 3
initial_mesh_N = 10

# decay exponent
decay_exp = 2

# FEM degree
degree = 1

# multioperator assembly type
assembly_type = ASSEMBLY_TYPE.MU #JOINT_GLOBAL #JOINT_MU

# plotting flag
PLOT_SOLUTION = True


# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = list(Multiindex.createCompleteOrderSet(2, 1))
#mis = list(Multiindex.createCompleteOrderSet(0, 1))
mis = [mis[0], mis[2]]
#mis = [mis[0]]

# setup meshes
mesh0, boundaries, dim = SampleDomain.setupDomain(domain, initial_mesh_N=initial_mesh_N)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# define coefficient field
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials")
gamma = 0.9
coeff_field = SampleProblem.setupCF(coeff_types[1], decayexp=decay_exp, gamma=gamma, freqscale=1, freqskip=0, rvtype="uniform", scale=1e5)
a0 = coeff_field.mean_func

# setup boundary conditions
Dirichlet_boundary = None
uD = None
Neumann_boundary = None
g = None
# ========== Navier-Lame ===========
# define source term
f = Constant((0.0, 0.0))
# define Dirichlet bc
Dirichlet_boundary = (boundaries['left'], boundaries['right'])
uD = (Constant((0.0, 0.0)), Constant((0.3, 0.0)))
#    Dirichlet_boundary = (boundaries['left'], boundaries['right'])
#    uD = (Constant((0.0, 0.0)), Constant((1.0, 1.0)))
# homogeneous Neumann does not have to be set explicitly
Neumann_boundary = None # (boundaries['right'])
g = None #Constant((0.0, 10.0))
# create pde instance
pde = FEMNavierLame(mu=1e4, lmbda0=a0,
                    dirichlet_boundary=Dirichlet_boundary, uD=uD,
                    neumann_boundary=Neumann_boundary, g=g,
                    f=f)

# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs, assembly_type=assembly_type)

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=degree))
logger.info("active indices of w after initialisation: %s", w.active_indices())


# ============================================================
# PART C: Assemble and Solve
# ============================================================

# pcg solver
pcg_eps = 1e-6
pcg_maxiter = 100

b = prepare_rhs(A, w, coeff_field, pde)

P = PreconditioningOperator(coeff_field.mean_func, pde.assemble_solve_operator)
w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)

logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)

#if True:
#    for i, mesh in enumerate(meshes):
#        plot(mesh, title="mesh %i" % i, mode="displacement")
#    interactive()

# plot sample solution
if PLOT_SOLUTION:
    # get random field sample and evaluate solution (direct and parametric)
    RV_samples = coeff_field.sample_rvs()
    ref_maxm = w.max_order
    mu0 = Multiindex()
    sub_spaces = w[mu0].basis.num_sub_spaces
    degree = w[mu0].basis.degree
    maxh = w[mu0].basis.minh
    projection_basis = get_projection_basis(mesh0, maxh=maxh, degree=degree, sub_spaces=sub_spaces)
    sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
    sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis)
    sol_variance = compute_solution_variance(coeff_field, w, projection_basis)
#        # debug---
#        if not True:        
#            for mu in w.active_indices():
#                for i, wi in enumerate(w_history):
#                    if i == len(w_history) - 1 or True:
#                        plot(wi[mu]._fefunc, title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
##                        plot(wi[mu]._fefunc.function_space().mesh(), title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
#                interactive()
#        # ---debug
    mesh_param = sample_sol_param._fefunc.function_space().mesh()
    mesh_direct = sample_sol_direct._fefunc.function_space().mesh()
    wireframe = True
    viz_p = plot(sample_sol_param._fefunc, title="parametric solution", mode="displacement", mesh=mesh_param, wireframe=wireframe)#, rescale=False)
    viz_d = plot(sample_sol_direct._fefunc, title="direct solution", mode="displacement", mesh=mesh_direct, wireframe=wireframe)#, rescale=False)
    
    for mu in w.active_indices():
        viz_p = plot(w[mu]._fefunc, title="parametric solution: " + str(mu), mode="displacement", mesh=mesh_param, wireframe=wireframe)
    interactive()



#import math
#from spuq.linalg.vector import inner
#def norm(v):
#    return math.sqrt(inner(v, v))
#
#def traceit(frame, event, arg):
#    filename = frame.f_code.co_filename
#    funcname = frame.f_code.co_name
#    lineno = frame.f_lineno
#
#    if event == "line" and lineno == 33 and funcname == "pcg":
#        locals = frame.f_locals
#        locals["norm"] = norm
#        globals = frame.f_globals
#        print "Iter %s -> %s, %s, %s" % eval(
#            "(i-1, norm(rho[i-1]), norm(s[i-1]), norm(v[i-1]))",
#            globals, locals)
#
#    return traceit
#
#import sys
#print sys.settrace(traceit)
