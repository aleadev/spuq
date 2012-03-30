from __future__ import division
from functools import partial
import logging
import os

from spuq.application.egsz.pcg import pcg
from spuq.application.egsz.multi_operator import MultiOperator, PreconditioningOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.linalg.vector import inner
try:
    from dolfin import (Function, FunctionSpace, Constant, Mesh,
                        UnitSquare, refine, plot, interactive, solve, interpolate)
    from spuq.application.egsz.marking import Marking
    from spuq.application.egsz.residual_estimator import ResidualEstimator
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.fem.fenics.fenics_utils import error_norm
except:
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# setup logging
# log level
LOG_LEVEL = logging.DEBUG
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
PLOT_MESHES = True

# flags for residual, projection, new mi refinement 
REFINEMENT = (True, True, False)

# define source term and diffusion coefficient
#f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
f = Constant("1.0")

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 1)]

# setup meshes 
#mesh0 = refine(Mesh(lshape_xml))
mesh0 = UnitSquare(5, 5)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":10, "random":(0.4, 0.3)})
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":0})

## debug---
#meshes[0] = refine(meshes[0])
## ---debug

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)
logger.info("active indices of after initialisation: %s", w.active_indices())

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square")
a0, _ = coeff_field[0]

# define multioperator
A = MultiOperator(coeff_field, FEMPoisson.assemble_operator)


# ============================================================
# PART B: Adaptive Algorithm
# ============================================================

# refinement loop
# ===============
# error constants
gamma = 0.9
cQ = 1.0
ceta = 1.0
# marking parameters
theta_eta = 0.8
theta_zeta = 0.8
min_zeta = 1e-5
maxh = 1 / 10
theta_delta = 0.1
# solver
pcg_eps = 1e-3
pcg_maxiter = 100
error_eps = 1e-2
max_refinements = 10

R = list()
for refinement in range(max_refinements):
    logger.info("************* REFINEMENT LOOP iteration %i *************", refinement + 1)

    # pcg solve
    # ---------
    b = 0 * w
    b0 = b[Multiindex()]
    b0.coeffs = FEMPoisson.assemble_rhs(f, b0.basis)
    P = PreconditioningOperator(a0, FEMPoisson.assemble_solve_operator)
    w, zeta, numit = pcg(A, b, P, w0=w, eps=pcg_eps, maxiter=pcg_maxiter)
    logger.info("PCG finished with zeta=%f after %i iterations", zeta, numit)
    b2 = A * w
#    residual = inner(b2 - b, b2 - b)    see errornorm for why this might be unstable numerically
    L2error = error_norm(b, b2, "L2")
    H1error = error_norm(b, b2, "H1")
    dofs = sum([b[mu]._fefunc.function_space().dim() for mu in b.keys()])
    R.append(list((L2error, H1error, dofs)))
    logger.info("Residual = %s (L2) %s (H1) with %s dofs", L2error, H1error, dofs)

    # error evaluation
    # ----------------
    xi, resind, projind = ResidualEstimator.evaluateError(w, coeff_field, f, zeta, gamma, ceta, cQ, 1 / 10)
    logger.info("Estimator Error = %s", xi)
    R[-1].append(xi)
    if xi <= error_eps:
        logger.info("error reached requested accuracy, xi=%f", xi)
        break

    # marking
    # -------
    mesh_markers_R, mesh_markers_P, new_multiindices = \
                    Marking.mark(resind, projind, w, coeff_field, theta_eta, theta_zeta, theta_delta, min_zeta, maxh)
    logger.info("MARKING will be carried out with %s cells", sum([len(cell_ids) for cell_ids in mesh_markers_R.itervalues()])
                                        + sum([len(cell_ids) for cell_ids in mesh_markers_P.itervalues()]) + len(new_multiindices))
    if REFINEMENT[0]:
        mesh_markers = mesh_markers_R.copy()
    else:
        mesh_markers = {}
        logger.info("SKIP residual refinement")
    if REFINEMENT[1]:
        mesh_markers.update(mesh_markers_P)
    else:
        logger.info("SKIP projection refinement")
    if not REFINEMENT[2] or refinement == max_refinements:
        new_multiindices = {}
        logger.info("SKIP new multiindex refinement")
    Marking.refine(w, mesh_markers, new_multiindices.keys(), partial(setup_vec, mesh=mesh0))
logger.info("ENDED refinement loop at refinement %i with %i dofs", refinement, R[-1][2])
logger.info("Residuals: %s", R)


# ============================================================
# PART C: Plotting and Export of Data
# ============================================================

# plot residuals
if PLOT_RESIDUAL and len(R) > 1:
    try:
        from matplotlib.pyplot import figure, show, legend
        x = [r[2] for r in R]
        L2 = [r[0] for r in R]
        H1 = [r[1] for r in R]
        errest = [r[3] for r in R]
        fig = figure()
        ax = fig.add_subplot(111)
        ax.loglog(x, errest, '-g<', label='error estimator')
        ax.loglog(x, H1, '-b^', label='H1 residual')
        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend()
        show()
    except:
        logger.info("skipped plotting since matplotlib is not available...")

# plot final meshes
if PLOT_MESHES:
    for mu, vec in w.iteritems():
        plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
        vec.plot(title=str(mu), interactive=False)
    interactive()
