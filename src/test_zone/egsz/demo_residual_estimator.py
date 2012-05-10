from __future__ import division
import logging
import os

from spuq.application.egsz.adaptive_solver import AdaptiveSolver
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
try:
    from dolfin import (Function, FunctionSpace, Constant, UnitSquare, plot, interactive)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
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
REFINEMENT = {"RES":True, "PROJ":True, "MI":False}
UNIFORM_REFINEMENT = False

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
mesh0 = UnitSquare(4, 4)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":10, "random":(0.4, 0.3)})
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":0})

# debug---
#meshes[0] = refine(meshes[0])
# ---debug

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)
logger.info("active indices of w after initialisation: %s", w.active_indices())

# define coefficient field
coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=4)
#coeff_field = SampleProblem.setupCF("constant", decayexp=4)
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
theta_eta = 0.6         # residual marking bulk parameter
theta_zeta = 0.4        # projection marking threshold factor
min_zeta = 1e-15        # minimal projection error considered
maxh = 1 / 10           # maximal mesh width for projection maximum norm evaluation
maxm = 10               # maximal search length for new new multiindices
theta_delta = 0.1       # number new multiindex activation bound
# pcg solver
pcg_eps = 2e-6
pcg_maxiter = 100
error_eps = 1e-2
# refinements
max_refinements = 10

w0 = w
w, sim_stats = AdaptiveSolver(A, coeff_field, f, mis, w0, mesh0, gamma=gamma, cQ=cQ, ceta=ceta,
                    # marking parameters
                    theta_eta=theta_eta, theta_zeta=theta_zeta, min_zeta=min_zeta, maxh=maxh, maxm=maxm, theta_delta=theta_delta,
                    # pcg solver
                    pcg_eps=pcg_eps, pcg_maxiter=pcg_maxiter,
                    # adaptive algorithm threshold
                    error_eps=error_eps,
                    # refinements
                    max_refinements=max_refinements, do_refinement=REFINEMENT, do_uniform_refinement=UNIFORM_REFINEMENT)


# ============================================================
# PART C: Plotting and Export of Data
# ============================================================

# plot residuals
if PLOT_RESIDUAL and len(sim_stats) > 1:
    try:
        from matplotlib.pyplot import figure, show, legend
        x = [s["DOFS"] for s in sim_stats]
        L2 = [s["L2"] for s in sim_stats]
        H1 = [s["H1"] for s in sim_stats]
        errest = [s["EST"] for s in sim_stats]
        reserr = [s["RES"] for s in sim_stats]
        projerr = [s["PROJ"] for s in sim_stats]
        mi = [s["MI"] for s in sim_stats]
        num_mi = [len(m) for m in mi]
        # figure 1
        # --------
        fig = figure()
        ax = fig.add_subplot(111)
#        if REFINEMENT["MI"]:
#            ax.loglog(x, num_mi, '--y+', label='active mi')
#        ax.loglog(x, errest, '-g<', label='error estimator')
#        ax.loglog(x, reserr, '-.cx', label='residual part')
#        ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
        ax.loglog(x, H1, '-b^', label='H1 residual')
        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend(loc='upper right')
        # figure 2
        # --------
        fig2 = figure()
        ax = fig2.add_subplot(111)
        if REFINEMENT["MI"]:
            ax.loglog(x, num_mi, '--y+', label='active mi')
        ax.loglog(x, errest, '-g<', label='error estimator')
        ax.loglog(x, reserr, '-.cx', label='residual part')
        ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
#        ax.loglog(x, H1, '-b^', label='H1 residual')
#        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend(loc='upper right')
        show()
    except:
        import traceback
        print traceback.format_exc()
        logger.info("skipped plotting since matplotlib is not available...")

# plot final meshes
if PLOT_MESHES:
    for mu, vec in w.iteritems():
        plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
        vec.plot(title=str(mu), interactive=False)
    interactive()
