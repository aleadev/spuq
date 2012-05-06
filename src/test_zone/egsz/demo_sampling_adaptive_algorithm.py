from __future__ import division
import logging
import os

from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet

try:
    from dolfin import (Function, FunctionSpace, Constant, UnitSquare, refine,
                        solve, plot, interactive, project, errornorm)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.application.egsz.adaptive_solver import AdaptiveSolver
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except Exception, e:
    import traceback

    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# setup logging
# log level
LOG_LEVEL = logging.INFO
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=__file__[:-2] + 'log', level=LOG_LEVEL,
    format=log_format)
fenics_logger = logging.getLogger("FFC")
fenics_logger.setLevel(logging.WARNING)
fenics_logger = logging.getLogger("UFL")
fenics_logger.setLevel(logging.WARNING)
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

# utility functions 

# setup initial multivector
def setup_vec(mesh):
    fs = FunctionSpace(mesh, "CG", 1)
    vec = FEniCSVector(Function(fs))
    return vec


# ============================================================
# PART A: Problem Setup
# ============================================================

# flag for residual graph plotting
PLOT_RESIDUAL = True

# flag for final solution plotting
PLOT_MESHES = False

# flags for residual, projection, new mi refinement 
refinement = {"RES": True, "PROJ": True, "MI": False}
uniform_refinement = False

# define source term and diffusion coefficient
#f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
f = Constant("1.0")

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 2)]

# setup meshes
#mesh0 = refine(Mesh(lshape_xml))
mesh0 = UnitSquare(5, 5)
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine": 3})

w0 = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)

logger.info("active indices of w after initialisation: %s", w0.active_indices())

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=2, amp=0.40, rvtype="uniform")

# define multioperator
A = MultiOperator(coeff_field, FEMPoisson.assemble_operator)


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

w, info = AdaptiveSolver(A, coeff_field, f, mis, w0, mesh0,
    do_refinement=refinement,
    do_uniform_refinement=uniform_refinement,
    max_refinements=0,
    pcg_eps=1e-4)

#coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=0, amp=10, rvtype="uniform")


# ============================================================
# PART C: Evaluation of Deterministic Solution and Comparison
# ============================================================

# dbg
print "w:", w

Delta = w.active_indices()
maxm = w.max_order

# get realization of coefficient field
sample_map, RV_samples = coeff_field.sample_realization(Delta)

# dbg
print "RV_samples:", RV_samples
print "sample_map:", sample_map

# create reference mesh and function space
cf_mesh_refinements = 2
mesh = refine(mesh0)
for i in range(cf_mesh_refinements):
    mesh = refine(mesh)
fs = FunctionSpace(mesh, "CG", 1)

# ============== STOCHASTIC SOLUTION ==============

# sum up (stochastic) solution vector on reference function space wrt samples
sample_sol_param = None
vec = FEniCSVector(Function(fs))
for mu in Delta:
    sol = w.project(w[mu], vec) * sample_map[mu]
    if sample_sol_param is None:
        sample_sol_param = sol
    else:
        sample_sol_param += sol

# store stochastic part of solution
sample_sol_stochastic = sample_sol_param - w.project(w[Multiindex()], vec) * sample_map[Multiindex()]


# ============== DETERMINISTIC SOLUTION ==============

# sum up coefficient field sample
a0 = coeff_field.mean_func
a = a0
for m in range(maxm):
    if m == 10:
        continue
    print m, RV_samples[m], coeff_field[m][0].cppcode, coeff_field[m][0].A, coeff_field[m][0].m, coeff_field[m][0].n
    a_m = RV_samples[m] * coeff_field[m][0]
    a += a_m

A = FEMPoisson.assemble_lhs(a, vec.basis)
b = FEMPoisson.assemble_rhs(Constant("1.0"), vec.basis)
X = 0 * b
solve(A, X, b)
sample_sol_direct = FEniCSVector(Function(vec.basis._fefs, X))

# evaluate errors
print "ERRORS: L2 =", errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "L2"), \
            "  H1 =", errornorm(sample_sol_param._fefunc, sample_sol_direct._fefunc, "H1")
sample_sol_err = sample_sol_param - sample_sol_direct
sample_sol_err.coeffs = sample_sol_err.coeffs
sample_sol_err.coeffs.abs()

# plotting
a_func = project(a, fs)
a_vec = FEniCSVector(a_func)
a_vec.plot(interactive=False, title="coefficient field")
sample_sol_param.plot(interactive=False, title="parametric solution")
sample_sol_stochastic.plot(interactive=False, title="stochastic part of solution")
sample_sol_direct.plot(interactive=False, title="direct solution")
sample_sol_err.plot(interactive=True, title="error")
