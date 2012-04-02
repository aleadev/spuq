from __future__ import division
from functools import partial
from math import sqrt
import logging
import os

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
PLOT_MESHES = False

# flags for residual, projection, new mi refinement 
REFINEMENT = {"RES":True, "PROJ":True, "MI":True}
UNIFORM_REFINEMENT = True

# define source term and diffusion coefficient
#f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
f = Constant("1.0")

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 2)]

# debug---
#mis = (Multiindex(),)
# ---debug

# setup meshes 
#mesh0 = refine(Mesh(lshape_xml))
mesh0 = UnitSquare(5, 5)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":10, "random":(0.4, 0.3)})
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":0})

# debug---
#meshes[0] = refine(meshes[0])
# ---debug

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)
logger.info("active indices of w after initialisation: %s", w.active_indices())

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square")
#coeff_field = SampleProblem.setupCF("linear")
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
theta_eta = 0.6         # residual marking
theta_zeta = 0.8        # projection marking
min_zeta = 1e-5         # minimal projection error considered
maxh = 1 / 10           # maximal mesh width for projection maximum norm evaluation
maxm = 10               # maximal search length for new new multiindices
theta_delta = 0.1       # number new multiindex activation bound
# pcg solver
pcg_eps = 1e-3
pcg_maxiter = 100
error_eps = 1e-2
# refinements
max_refinements = 5
# data collection
sim_info = {}
R = list()              # residual, estimator and dof progress
# refinement loop
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
    L2error = error_norm(b, b2, "L2")
    H1error = error_norm(b, b2, "H1")
    dofs = sum([b[mu]._fefunc.function_space().dim() for mu in b.keys()])
    R.append({"L2":L2error, "H1":H1error, "DOFS":dofs})
    logger.info("Residual = %s (L2) %s (H1) with %s dofs", L2error, H1error, dofs)

    # error evaluation
    # ----------------
    xi, resind, projind = ResidualEstimator.evaluateError(w, coeff_field, f, zeta, gamma, ceta, cQ, 1 / 10)
    reserr = sqrt(sum([sum(resind[mu].coeffs ** 2) for mu in resind.keys()]))          # TODO: sqrt?
    projerr = sqrt(sum([sum(projind[mu].coeffs ** 2) for mu in projind.keys()]))
    logger.info("Estimator Error = %s while residual error is %s and projection error is %s", xi, reserr, projerr)
    sim_info[refinement] = ([(mu, vec.basis.dim) for mu, vec in w.iteritems()], R[-1])
    R[-1]["EST"] = xi
    R[-1]["RES"] = reserr
    R[-1]["PROJ"] = projerr
    R[-1]["MI"] = len(sim_info[refinement][0])
    if xi <= error_eps:
        logger.info("error reached requested accuracy, xi=%f", xi)
        break
    
    # marking
    # -------
    if refinement < max_refinements - 1:
        if not UNIFORM_REFINEMENT:
            mesh_markers_R, mesh_markers_P, new_multiindices = \
                            Marking.mark(resind, projind, w, coeff_field, theta_eta, theta_zeta, theta_delta, min_zeta, maxh, maxm)
            logger.info("MARKING will be carried out with %s cells and %s new multiindices", sum([len(cell_ids) for cell_ids in mesh_markers_R.itervalues()])
                                                + sum([len(cell_ids) for cell_ids in mesh_markers_P.itervalues()]), len(new_multiindices))
            if REFINEMENT["RES"]:
                mesh_markers = mesh_markers_R.copy()
            else:
                mesh_markers = {}
                logger.info("SKIP residual refinement")
            if REFINEMENT["PROJ"]:
                for mu, cells in mesh_markers_P.iteritems():
                    if len(cells) > 0:
                        mesh_markers[mu] = mesh_markers[mu].union(cells)
            else:
                logger.info("SKIP projection refinement")
            if not REFINEMENT["MI"] or refinement == max_refinements:
                new_multiindices = {}
                logger.info("SKIP new multiindex refinement")
        else:
            logger.info("UNIFORM REFINEMENT active")
            mesh_markers = {}            
            for mu, vec in w.iteritems():
                mesh_markers[mu] = list([c.index() for c in cells(vec._fefunc.function_space().mesh())])
#            # debug---
#            mu = Multiindex()
#            mesh_markers[mu] = list([c.index() for c in cells(w[mu]._fefunc.function_space().mesh())])
#            # ---debug
            new_multiindices = {}
        Marking.refine(w, mesh_markers, new_multiindices.keys(), partial(setup_vec, mesh=mesh0))
logger.info("ENDED refinement loop at refinement %i with %i dofs and %i active multiindices",
                                refinement, sim_info[refinement][1]["DOFS"], len(sim_info[refinement][0]))
logger.info("Residuals: %s", R)
logger.info("Simulation run data: %s", sim_info)

# ============================================================
# PART C: Plotting and Export of Data
# ============================================================

# plot residuals
if PLOT_RESIDUAL and len(R) > 1:
    try:
        from matplotlib.pyplot import figure, show, legend
        x = [r["DOFS"] for r in R]
        L2 = [r["L2"] for r in R]
        H1 = [r["H1"] for r in R]
        errest = [r["EST"] for r in R]
        reserr = [r["RES"] for r in R]
        projerr = [r["PROJ"] for r in R]
        num_mi = [r["MI"] for r in R]
        fig = figure()
        ax = fig.add_subplot(111)
        if REFINEMENT["MI"]:
            ax.loglog(x, num_mi, '--y+', label='active mi')
        ax.loglog(x, errest, '-g<', label='error estimator')
        ax.loglog(x, reserr, '-.cx', label='residual part')
        ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
        ax.loglog(x, H1, '-b^', label='H1 residual')
        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend(loc='lower right')
        show()
    except:
        logger.info("skipped plotting since matplotlib is not available...")

# plot final meshes
if PLOT_MESHES:
    for mu, vec in w.iteritems():
        plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
        vec.plot(title=str(mu), interactive=False)
    interactive()
