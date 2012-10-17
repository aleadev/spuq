from __future__ import division
import logging
import os
import functools
from math import sqrt

from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, ASSEMBLY_TYPE
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
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

# setup logging
LOG_LEVEL = logging.INFO
logger = setup_logging(LOG_LEVEL)

# determine path of this module
path = os.path.dirname(__file__)

# ============================================================
# PART A: Simulation Options
# ============================================================

# set problem (0:Poisson, 1:Navier-Lame)
domaintype = 0
domains = ('square', 'lshape', 'cooks')
domain = domains[domaintype]

# decay exponent
decay_exp = 2

# refinements
max_refinements = 0

# polynomial degree of FEM approximation
degree = 1

# multioperator assembly type
assembly_type = ASSEMBLY_TYPE.MU #JOINT_GLOBAL #JOINT_MU

# flag for final solution export
SAVE_SOLUTION = ''
#SAVE_SOLUTION = os.path.join(os.path.dirname(__file__), "results/demo-residual-A2")

# plotting flag
PLOT_SOLUTION = True

# flags for residual, projection, new mi refinement 
REFINEMENT = {"RES":True, "PROJ":True, "MI":False}
UNIFORM_REFINEMENT = True

# initial mesh elements
initial_mesh_N = 10

# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 1)]

# debug---
#mis = [Multiindex(), ]
mis = [mis[0], mis[2]]
# ---debug

# setup domain and meshes
mesh0, boundaries, dim = SampleDomain.setupDomain(domain, initial_mesh_N=initial_mesh_N)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.4, 0.3))
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# define coefficient field
# NOTE: for proper treatment of corner points, see elasticity_residual_estimator
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
A = MultiOperator(coeff_field, pde.assemble_operator, assembly_type=assembly_type)

#setup multivector
w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=degree))
logger.info("active indices of w after initialisation: %s", w.active_indices())


# ============================================================
# PART C: Adaptive Algorithm
# ============================================================

# -------------------------------------------------------------
# -------------- ADAPTIVE ALGORITHM OPTIONS -------------------
# -------------------------------------------------------------
# error constants
cQ = 1.0
ceta = 1.0
# marking parameters
theta_eta = 0.5         # residual marking bulk parameter
theta_zeta = 0.1        # projection marking threshold factor
min_zeta = 1e-10        # minimal projection error considered
maxh = 1 / 10           # maximal mesh width for projection maximum norm evaluation
newmi_add_maxm = 10     # maximal search length for new new multiindices (to be added to max order of solution)
theta_delta = 0.95       # number new multiindex activation bound
max_Lambda_frac = 1 / 10 # fraction of |Lambda| for max number of new multiindices
# residual error evaluation
quadrature_degree = 3
# projection error evaluation
projection_degree_increase = 2
refine_projection_mesh = 2
# pcg solver
pcg_eps = 1e-6
pcg_maxiter = 100
error_eps = 1e-4

w_history = []

# refinement loop
# ===============
w0 = w
w, sim_stats = AdaptiveSolver(A, coeff_field, pde, mis, w0, mesh0, degree, gamma=gamma, cQ=cQ, ceta=ceta,
                    # marking parameters
                    theta_eta=theta_eta, theta_zeta=theta_zeta, min_zeta=min_zeta,
                    maxh=maxh, newmi_add_maxm=newmi_add_maxm, theta_delta=theta_delta,
                    max_Lambda_frac=max_Lambda_frac,
                    # residual error evaluation
                    quadrature_degree=quadrature_degree,
                    # projection error evaluation
                    projection_degree_increase=projection_degree_increase, refine_projection_mesh=refine_projection_mesh,
                    # pcg solver
                    pcg_eps=pcg_eps, pcg_maxiter=pcg_maxiter,
                    # adaptive algorithm threshold
                    error_eps=error_eps,
                    # refinements
                    max_refinements=max_refinements, do_refinement=REFINEMENT, do_uniform_refinement=UNIFORM_REFINEMENT,
                    w_history=w_history)

from operator import itemgetter
active_mi = [(mu, w[mu]._fefunc.function_space().mesh().num_cells()) for mu in w.active_indices()]
active_mi = sorted(active_mi, key=itemgetter(1), reverse=True)
logger.info("==== FINAL MESHES ====")
for mu in active_mi:
    logger.info("--- %s has %s cells", mu[0], mu[1])
print "ACTIVE MI:", active_mi
print

# memory usage info
import resource
logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")


if len(sim_stats) > 1:
    try:
        from matplotlib.pyplot import figure, show, legend
        x = [s["DOFS"] for s in sim_stats]
        L2 = [s["L2"] for s in sim_stats]
        H1 = [s["H1"] for s in sim_stats]
        errest = [sqrt(s["EST"]) for s in sim_stats]
        reserr = [s["RES"] for s in sim_stats]
        projerr = [s["PROJ"] for s in sim_stats]
        mi = [s["MI"] for s in sim_stats]
        num_mi = [len(m) for m in mi]
        print "errest", errest
        
        # figure 1
        # --------
        fig1 = figure()
        fig1.suptitle("residual estimator")
        ax = fig1.add_subplot(111)
        if REFINEMENT["MI"]:
            ax.loglog(x, num_mi, '--y+', label='active mi')
        ax.loglog(x, errest, '-g<', label='error estimator')
        ax.loglog(x, reserr, '-.cx', label='residual part')
        ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
        legend(loc='upper right')
        show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        logger.info("skipped plotting since matplotlib is not available...")

# plot sample solution
if PLOT_SOLUTION:
    # get random field sample and evaluate solution (direct and parametric)
    RV_samples = coeff_field.sample_rvs()
    ref_maxm = w_history[-1].max_order
    sub_spaces = w[Multiindex()].basis.num_sub_spaces
    degree = w[Multiindex()].basis.degree
    maxh = w[Multiindex()].basis.minh
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
