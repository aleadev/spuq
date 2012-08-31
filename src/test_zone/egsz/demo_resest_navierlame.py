from __future__ import division
import logging
import os
import functools
from math import sqrt

from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.application.egsz.fem_discretisation import FEMNavierLame
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

def setup_logging(level):
    # log level and format configuration
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=__file__[:-2] + 'log', level=LOG_LEVEL,
                        format=log_format)
    
    # FEniCS logging
    from dolfin import (set_log_level, set_log_active, INFO, DEBUG, WARNING)
    set_log_active(True)
    set_log_level(WARNING)
    fenics_logger = logging.getLogger("FFC")
    fenics_logger.setLevel(logging.WARNING)
    fenics_logger = logging.getLogger("UFL")
    fenics_logger.setLevel(logging.WARNING)
    
    # module logger
    logger = logging.getLogger(__name__)
    logging.getLogger("spuq.application.egsz.multi_operator").disabled = True
    #logging.getLogger("spuq.application.egsz.marking").setLevel(logging.INFO)
    # add console logging output
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)
    logging.getLogger("spuq").addHandler(ch)
    return logger

# setup logging
LOG_LEVEL = logging.INFO
logger = setup_logging(LOG_LEVEL)

# determine path of this module
path = os.path.dirname(__file__)

# ============================================================
# PART A: Simulation Options
# ============================================================

# set problem (0:Poisson, 1:Navier-Lame)
pdetype = 1
domaintype = 0
domains = ('square', 'lshape', 'cooks')
domain = domains[domaintype]

# decay exponent
decay_exp = 2

# refinements
max_refinements = 1

# polynomial degree of FEM approximation
degree = 1

# flag for residual graph plotting
PLOT_RESIDUAL = True

# flag for final mesh plotting
PLOT_MESHES = False

# flag for (sample) solution plotting
PLOT_SOLUTION = True

# flag for final solution export
#SAVE_SOLUTION = ''
SAVE_SOLUTION = os.path.join(os.path.dirname(__file__), "results/demo-residual-A2")

# flags for residual, projection, new mi refinement 
REFINEMENT = {"RES":True, "PROJ":True, "MI":False}
UNIFORM_REFINEMENT = True

# initial mesh elements
initial_mesh_N = 10

# MC error sampling
MC_RUNS = 1
MC_N = 1
MC_HMAX = 1 / 10

# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 1)]

# debug---
#mis = [Multiindex(),]
# ---debug

# setup domain and meshes
mesh0, boundaries = SampleDomain.setupDomain(domain, initial_mesh_N=initial_mesh_N)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.4, 0.3))
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# ---debug
#from spuq.application.egsz.multi_vector import MultiVectorWithProjection
#if SAVE_SOLUTION != "":
#    w.pickle(SAVE_SOLUTION)
#u = MultiVectorWithProjection.from_pickle(SAVE_SOLUTION, FEniCSVector)
#import sys
#sys.exit()
# ---debug

# define coefficient field
# NOTE: for proper treatment of corner points, see elasticity_residual_estimator
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials")
gamma = 0.9
coeff_field = SampleProblem.setupCF(coeff_types[1], decayexp=decay_exp, gamma=gamma, freqscale=1, freqskip=20, rvtype="uniform", scale=100000)
a0 = coeff_field.mean_func

# setup boundary conditions
Dirichlet_boundary = None
uD = None
Neumann_boundary = None
g = None
if pdetype == 1:
    # ========== Navier-Lame ===========
    # define source term
    f = Constant((0.0, 0.0))
    # define Dirichlet bc
    Dirichlet_boundary = (boundaries['left'], boundaries['right'])
    uD = (Constant((0.0, 0.0)), Constant((-0.3, 0.0)))
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
else:
    assert pdetype == 0
    # ========== Poisson ===========
    # define source term
    #f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
    f = Constant(1.0)
    # define Dirichlet bc
    # 4 Dirichlet
    # Dirichlet_boundary = (boundaries['left'], boundaries['right'], boundaries['top'], boundaries['bottom'])
    # uD = (Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0))
    # 2 Dirichlet
    Dirichlet_boundary = (boundaries['left'], boundaries['right'])
    uD = (Constant(0.0), Constant(0.0))
#    # 1 Dirichlet
#    Dirichlet_boundary = (boundaries['left'])
#    uD = (Constant(0.0))
#    # homogeneous Neumann does not have to be set explicitly
#    Neumann_boundary = None
#    g = None
    # create pde instance
    pde = FEMPoisson(a0=a0, dirichlet_boundary=Dirichlet_boundary, uD=uD,
                     neumann_boundary=Neumann_boundary, g=g,
                     f=f)

# define multioperator
A = MultiOperator(coeff_field, pde.assemble_operator)

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
pcg_eps = 1e-4
pcg_maxiter = 100
error_eps = 1e-4

if MC_RUNS > 0:
    w_history = []
else:
    w_history = None

# NOTE: for Cook's membrane, the mesh refinement gets stuck for some reason...
if domaintype == 2:
    maxh = 0.0
    MC_HMAX = 0

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


