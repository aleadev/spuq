from __future__ import division
import logging
import os

from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet

try:
    from dolfin import (Function, FunctionSpace, Constant, UnitSquare,
                        interactive, project, errornorm)
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
MC_RUNS = 2
MC_N = 3
MC_HMAX = 1 / 10
MC_DEGREE = 1


# log level and format configuration
LOG_LEVEL = logging.INFO
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


# flags for residual, projection, new mi refinement 
refinement = {"RES": True, "PROJ": True, "MI": True}
uniform_refinement = False

# define source term
f = Constant("1.0")

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 1)]

# setup meshes
#mesh0 = refine(Mesh(lshape_xml))
mesh0 = UnitSquare(5, 5)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine": 3})

w0 = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vector)

logger.info("active indices of w after initialisation: %s", w0.active_indices())

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=2, amp=2, rvtype="uniform")

# define multioperator
A = MultiOperator(coeff_field, FEMPoisson.assemble_operator)


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

w, sim_stats = AdaptiveSolver(A, coeff_field, f, mis, w0, mesh0,
    do_refinement=refinement,
    do_uniform_refinement=uniform_refinement,
    max_refinements=2 * 0,
    pcg_eps=1e-4)

logger.debug("active indices of w after solution: %s", w.active_indices())


# ============================================================
# PART C: Evaluation of Deterministic Solution and Comparison
# ============================================================

# create reference mesh and function space
proj_basis = get_projection_basis(mesh0, maxh=1 / 10, degree=MC_DEGREE)

def run_mc(w, err):
#    import time
    
    # create reference mesh and function space
    projection_basis = get_projection_basis(mesh0, maxh=1 / 10)

    # get realization of coefficient field
    err_L2, err_H1 = 0, 0
    for i in range(MC_N):
        logger.info("MC Iteration %i/%i", i , MC_N)
        RV_samples = coeff_field.sample_rvs()
        logger.debug("RV_samples: %s", [RV_samples[j] for j in range(w.max_order)])
#        t1 = time.time()
        sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, proj_basis)
#        t2 = time.time()
        sample_sol_direct = compute_direct_sample_solution(RV_samples, coeff_field, A, f, w.max_order, projection_basis)
#        t3 = time.time()
        cerr_L2 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "L2")
        cerr_H1 = errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "H1")
        err_L2 += 1.0 / MC_N * cerr_L2
        err_H1 += 1.0 / MC_N * cerr_H1
        
        if i == 0:
            sample_sol_direct_a0 = compute_direct_sample_solution(RV_samples, coeff_field, A, f, 0, projection_basis)
            L2_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "L2")
            H1_a0 = errornorm(sample_sol_param._fefunc, sample_sol_direct_a0._fefunc, "H1")
            print "deterministic error L2 = ", L2_a0, "   H1 = ", H1_a0
            fc = get_coeff_realisation(RV_samples, coeff_field, w.max_order, projection_basis)
            fc.plot(interactive=True)
            
#        t4 = time.time()
#        logger.info("TIMING: param: %s, direct %s, error %s", t2 - t1, t3 - t2, t4 - t3)

    logger.info("MC Error: L2: %s, H1: %s", err_L2, err_H1)
    err.append((err_L2, err_H1))

# iterate MC
err = []
for _ in range(MC_RUNS):
    run_mc(w, err)

#print "evaluated errors (L2,H1):", err
L2err = sum([e[0] for e in err]) / len(err)
H1err = sum([e[1] for e in err]) / len(err)
print "average ERRORS: L2 = ", L2err, "   H1 = ", H1err
