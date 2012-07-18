from __future__ import division
import logging
import os
import functools

from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet

try:
    from dolfin import (Function, FunctionSpace, Constant, UnitSquare, compile_subdomains,
                        Mesh, interactive, project, errornorm, DOLFIN_EPS)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.application.egsz.adaptive_solver import AdaptiveSolver
#    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.application.egsz.sampling import (get_projection_basis, compute_direct_sample_solution,
                                                compute_parametric_sample_solution, setup_vector, get_coeff_realisation)
except Exception, e:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# program parameters
PLOT_SOLUTION = False
MC_PLOT = True
MC_RUNS = 1
MC_N = 1
MC_HMAX = 1 / 20
MC_DEGREE = 1
NUM_REFINE = 2


# log level and format configuration
LOG_LEVEL = logging.DEBUG
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

# determine path of this module
path = os.path.dirname(__file__)
lshape_xml = os.path.join(path, 'lshape.xml')

# ------------------------------------------------------------


# ============================================================
# PART A: Problem Setup
# ============================================================

# problem discretisation
pde = FEMPoisson

# polynomial degree of FEM approximation
degree = 1

# flags for residual, projection, new mi refinement 
refinement = {"RES": True, "PROJ": True, "MI": True}
uniform_refinement = False

# define source term
f = pde.f(0)

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 1)]

# setup meshes
lshape = True

if lshape: 
    mesh0 = Mesh(lshape_xml)
    maxx, minx, maxy, miny = 1, -1, 1, -1
else:
    mesh0 = UnitSquare(5, 5)
    maxx, minx, maxy, miny = 1, 0, 1, 0
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.4, 0.3))
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)

# setup boundary parts
top, bottom, left, right = compile_subdomains([  'near(x[1], maxy) && on_boundary',
                                                 'near(x[1], miny) && on_boundary',
                                                 'near(x[0], minx) && on_boundary',
                                                 'near(x[0], maxx) && on_boundary'])
top.maxy = maxy
bottom.miny = miny
left.minx = minx
right.maxx = maxx

#meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=2, randref=(0.7, 0.8))
meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)
w0 = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=degree))

logger.info("active indices of w after initialisation: %s", w0.active_indices())

# define coefficient field
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials")
gamma = 0.9
coeff_field = SampleProblem.setupCF(coeff_types[1], decayexp=2, gamma=gamma, freqscale=1, freqskip=10, rvtype="uniform")

# define Dirichlet and Neumann boundaries
#Dirichlet_boundary = lambda x, on_boundary: on_boundary and (x[0] <= DOLFIN_EPS or x[0] >= 1.0 - DOLFIN_EPS)# or x[1] >= 1.0 - DOLFIN_EPS)
Dirichlet_boundary = (left, top)
# homogeneous Neumann does not have to be set explicitly
Neumann_boundary = None
g = None

# define multioperator and rhs
A = MultiOperator(coeff_field, functools.partial(pde.assemble_operator, Dirichlet_boundary=Dirichlet_boundary))
rhs = functools.partial(pde.assemble_rhs, f=f, g=g, Neumann_boundary=Neumann_boundary)


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

w, sim_stats = AdaptiveSolver(A, coeff_field, pde, rhs, f, mis, w0, mesh0,
    gamma=gamma,
    do_refinement=refinement,
    do_uniform_refinement=uniform_refinement,
    max_refinements=NUM_REFINE,
    pcg_eps=1e-1)

logger.debug("active indices of w after solution: %s", w.active_indices())


# ============================================================
# PART C: Evaluation of Deterministic Solution and Comparison
# ============================================================

def run_mc(w, err, pde):
    import time
    from dolfin import norm
    
    # create reference mesh and function space
    projection_basis = get_projection_basis(mesh0, maxh=min(w[Multiindex()].basis.minh / 4, MC_HMAX))
    logger.debug("hmin of mi[0] = %s, reference mesh = (%s, %s)", w[Multiindex()].basis.minh, projection_basis.minh, projection_basis.maxh)

    # get realization of coefficient field
    err_L2, err_H1 = 0, 0
    for i in range(MC_N):
        logger.info("---- MC Iteration %i/%i ----", i + 1 , MC_N)
        RV_samples = coeff_field.sample_rvs()
        logger.debug("-- RV_samples: %s", [RV_samples[j] for j in range(w.max_order)])
        t1 = time.time()
        sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
        t2 = time.time()
        sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, f, 2 * w.max_order, projection_basis, Dirichlet_boundary=Dirichlet_boundary)
        t3 = time.time()
        cerr_L2 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "L2")
        cerr_H1 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "H1")
        logger.debug("-- current error L2 = %s    H1 = %s", cerr_L2, cerr_H1)
        err_L2 += 1.0 / MC_N * cerr_L2
        err_H1 += 1.0 / MC_N * cerr_H1
        
        if i + 1 == MC_N:
            # error function
            errf = sample_sol_param - sample_sol_direct
            
            # deterministic part
            sample_sol_direct_a0 = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, f, 0, projection_basis)
            L2_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "L2")
            H1_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "H1")
            logger.info("-- DETERMINISTIC error L2 = %s    H1 = %s", L2_a0, H1_a0)

            # stochastic part
            sample_sol_direct_am = sample_sol_direct - sample_sol_direct_a0
#            L2_am = errornorm(sample_sol_param._fefunc, sample_sol_direct_am._fefunc, "L2")
#            H1_am = errornorm(sample_sol_param._fefunc, sample_sol_direct_am._fefunc, "H1")
            logger.info("-- STOCHASTIC norm L2 = %s    H1 = %s", sample_sol_direct_am.norm("L2"), sample_sol_direct_am.norm("H1"))
            if MC_PLOT:
                sample_sol_param.plot(title="param")
                sample_sol_direct.plot(title="direct")
                errf.plot(title="|param-direct| error")
                sample_sol_direct_am.plot(title="direct stochastic part")
                fc = get_coeff_realisation(RV_samples, coeff_field, w.max_order, projection_basis)
                fc.plot(title="coeff", interactive=True)
            
        t4 = time.time()
        logger.info("TIMING: param: %s, direct %s, error %s", t2 - t1, t3 - t2, t4 - t3)

    logger.info("MC Error: L2: %s, H1: %s", err_L2, err_H1)
    err.append((err_L2, err_H1))

# iterate MC
err = []
for _ in range(MC_RUNS):
    run_mc(w, err, pde)

#print "evaluated errors (L2,H1):", err
L2err = sum([e[0] for e in err]) / len(err)
H1err = sum([e[1] for e in err]) / len(err)
print "average ERRORS: L2 = ", L2err, "   H1 = ", H1err
