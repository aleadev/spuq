from __future__ import division
from functools import partial
import logging
import os

from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet

try:
    from dolfin import (Function, FunctionSpace, Constant, Mesh,
                        refine, plot, interactive, solve)
    from spuq.application.egsz.marking import Marking
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    print "FEniCS has to be available"
    os.sys.exit(1)

PLOT_MESHES = True

# setup logging
logging.basicConfig(filename=__file__[:-2] + 'log', level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fenics_logger = logging.getLogger("FFC")
fenics_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# determine path of this module
path = os.path.dirname(__file__)
lshape_xml = os.path.join(path, 'lshape.xml')

# ============================================================
# PART A: Problem Setup
# ============================================================

# define source term and diffusion coefficient
#f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
f = Constant("1.0")
diffcoeff = Constant("1.0")

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 2)]

# setup meshes 
mesh0 = refine(Mesh(lshape_xml))
#mesh0 = UnitSquare(3, 3)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":10, "random":(0.4, 0.3)})
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":1})

# setup initial multivector
def setup_vec(mesh=mesh0, with_solve=True):
    fs = FunctionSpace(mesh, "CG", 1)
    vec = FEniCSVector(Function(fs))
    if with_solve:
        eval_poisson(vec)
    return vec

def eval_poisson(vec=None):
    if vec == None:
        vec = setup_vec(with_solve=False)
    fem_A = FEMPoisson.assemble_lhs(diffcoeff, vec.basis)
    fem_b = FEMPoisson.assemble_rhs(f, vec.basis)
    solve(fem_A, vec.coeffs, fem_b)
    return vec

zero_vec = partial(setup_vec, with_solve=False)
w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), zero_vec)
logger.info("active indices of after initialisation: %s", w.active_indices())

#if PLOT_MESHES:
#    for mu, vec in w.iteritems():
#        plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
#        plot(vec._fefunc)
#    interactive()

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square")


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

# refinement loop
# ===============
theta_eta = 0.3
theta_zeta = 0.8
min_zeta = 1e-10
maxh = 1 / 10
theta_delta = 0.8
refinements = 5

for refinement in range(refinements):
    logger.info("*****************************")
    logger.info("REFINEMENT LOOP iteration %i", refinement + 1)
    logger.info("*****************************")

    # evaluate residual and projection error estimates
    # ================================================
    mesh_markers_R, mesh_markers_P, new_multiindices = Marking.mark(w, coeff_field, f, theta_eta, theta_zeta, theta_delta, min_zeta, maxh)
    mesh_markers = mesh_markers_R.copy()
    mesh_markers.update(mesh_markers_P)
    Marking.refine(w, mesh_markers, new_multiindices.keys(), eval_poisson)

# show refined meshes
if PLOT_MESHES:
    for mu, vec in w.iteritems():
        plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
        plot(vec._fefunc)
    interactive()
