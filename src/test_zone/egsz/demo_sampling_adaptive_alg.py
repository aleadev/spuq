from __future__ import division
from functools import partial
from math import sqrt
import logging
import os
from spuq.application.egsz.multi_vector import MultiVectorWithProjection

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.linalg.vector import inner

try:
    from dolfin import (Function, FunctionSpace, Constant, Mesh, cells,
                        UnitSquare, refine, plot, interactive, interpolate)
    from spuq.application.egsz.marking import Marking
    from spuq.application.egsz.residual_estimator import ResidualEstimator
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.application.egsz.adaptive_solver import adaptive_solver
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_utils import error_norm
except:
    print "FEniCS has to be available"
    os.sys.exit(1)

from spuq.application.egsz.multi_vector import MultiVectorWithProjection

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
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine": 0})

w0 = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)

logger.info("active indices of w after initialisation: %s", w0.active_indices())

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square", {"exp": 4})

# define multioperator
A = MultiOperator(coeff_field, FEMPoisson.assemble_operator)


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

(w, info) = adaptive_solver(A, coeff_field, f, mis, w0, mesh0,
    do_refinement=refinement,
    do_uniform_refinement=uniform_refinement,
    max_refinements=1
)


# ============================================================
# PART C: Plotting and Export of Data
# ============================================================

print w

Delta = w.active_indices()
maxm = w.max_order() + 1
RV_samples = [0, ]
coeff_field.
for m in range(1, maxm):
    RV_samples.append(coeff_field[m][1].sample(1))

sample_map = {}

def prod(l):
    p = 1
    for f in l:
        if p is None:
            p = f
        else:
            p *= f
    return p

for mu in Delta:
    sample_map[mu] = prod(coeff_field[m + 1][1].orth_polys[mu[m]](RV_samples[m + 1]) for m in range(len(mu)))

print RV_samples
print sample_map

from dolfin import refine

mesh = refine(mesh0)
for i in range(5):
    mesh = refine(mesh)
fs = FunctionSpace(mesh, "CG", 1)

sample_sol = None
vec = FEniCSVector(Function(fs))
for mu in Delta:
    sol = w.project(w[mu], vec) * float(sample_map[mu])
    if sample_sol is None:
        sample_sol = sol
    else:
        sample_sol += sol

sample_sol.plot(interactive=True)

a0 = coeff_field[0][0]
a = a0
for m in range(1, maxm):
    a_m = coeff_field[m][0] * float(RV_samples[m])
    a += a_m
