from __future__ import division
import logging
import os
from math import sqrt

from spuq.application.egsz.adaptive_solver import AdaptiveSolver
from spuq.application.egsz.multi_operator import MultiOperator
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Constant, UnitSquare, plot, interactive, set_log_level, set_log_active)
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------

# setup logging
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

# flag for final solution export
SAVE_SOLUTION = '' #"results/demo-residual"

# flags for residual, projection, new mi refinement 
REFINEMENT = {"RES":True, "PROJ":True, "MI":True}
UNIFORM_REFINEMENT = False

# define source term
#f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
f = Constant("1.0")

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(2, 1)]

# debug---
#mis = (Multiindex(),)
# ---debug

# setup meshes 
#mesh0 = refine(Mesh(lshape_xml))
mesh0 = UnitSquare(4, 4)
#meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":10, "random":(0.4, 0.3)})
meshes = SampleProblem.setupMeshes(mesh0, len(mis), {"refine":0})

# debug---
#from dolfin import refine
#meshes[1] = refine(meshes[1])
# ---debug

w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), setup_vec)
logger.info("active indices of w after initialisation: %s", w.active_indices())

# ---debug
#from spuq.application.egsz.multi_vector import MultiVectorWithProjection
#if SAVE_SOLUTION != "":
#    w.pickle(SAVE_SOLUTION)
#u = MultiVectorWithProjection.from_pickle(SAVE_SOLUTION, FEniCSVector)
#import sys
#sys.exit()
# ---debug

# define coefficient field
#coeff_field = SampleProblem.setupCF("EF-square-cos", decayexp=1)
coeff_field = SampleProblem.setupCF("constant", decayexp=2)
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
theta_eta = 0.3         # residual marking bulk parameter
theta_zeta = 0.3        # projection marking threshold factor
min_zeta = 1e-15        # minimal projection error considered
maxh = 1 / 10           # maximal mesh width for projection maximum norm evaluation
maxm = 10               # maximal search length for new new multiindices
theta_delta = 0.9       # number new multiindex activation bound
max_Lambda_frac = 1 / 10 # fraction of |Lambda| for max number of new multiindices 
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
                    max_Lambda_frac=max_Lambda_frac,
                    # pcg solver
                    pcg_eps=pcg_eps, pcg_maxiter=pcg_maxiter,
                    # adaptive algorithm threshold
                    error_eps=error_eps,
                    # refinements
                    max_refinements=max_refinements, do_refinement=REFINEMENT, do_uniform_refinement=UNIFORM_REFINEMENT)


# ============================================================
# PART C: Plotting and Export of Data
# ============================================================

from operator import itemgetter
active_mi = [(mu, w[mu]._fefunc.function_space().mesh().num_cells()) for mu in w.active_indices()]
active_mi = sorted(active_mi, key=itemgetter(1), reverse=True)
logger.info("==== FINAL MESHES ====")
for mu in active_mi:
    logger.info("--- %s has %s cells", mu[0], mu[1])
print "ACTIVE MI:", active_mi

# save solution
if SAVE_SOLUTION != "":
    # save solution (also creates directory if not existing)
    w.pickle(SAVE_SOLUTION)
    # save simulation data
    import pickle
    with open(os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'), 'wb') as f:
        pickle.dump(sim_stats, f)

# plot residuals
if PLOT_RESIDUAL and len(sim_stats) > 1:
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
        # figure 1
        # --------
        fig = figure()
        fig.suptitle("error")
        ax = fig.add_subplot(111)
#        if REFINEMENT["MI"]:
#            ax.loglog(x, num_mi, '--y+', label='active mi')
#        ax.loglog(x, errest, '-g<', label='error estimator')
#        ax.loglog(x, reserr, '-.cx', label='residual part')
#        ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
        ax.loglog(x, H1, '-b^', label='H1 residual')
        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend(loc='upper right')
        if SAVE_SOLUTION != "":
            fig.savefig(os.path.join(SAVE_SOLUTION, 'RES.png'))
        # figure 2
        # --------
        fig2 = figure()
        fig2.suptitle("residual estimator")
        ax = fig2.add_subplot(111)
        if REFINEMENT["MI"]:
            ax.loglog(x, num_mi, '--y+', label='active mi')
        ax.loglog(x, errest, '-g<', label='error estimator')
        ax.loglog(x, reserr, '-.cx', label='residual part')
        ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
#        ax.loglog(x, H1, '-b^', label='H1 residual')
#        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend(loc='upper right')
        if SAVE_SOLUTION != "":
            fig2.savefig(os.path.join(SAVE_SOLUTION, 'EST.png'))
        show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        logger.info("skipped plotting since matplotlib is not available...")

# plot final meshes
if PLOT_MESHES:
    USE_MAYAVI = Plotter.hasMayavi() 
    for mu, vec in w.iteritems():
        if USE_MAYAVI:
            # mesh
#            Plotter.figure(bgcolor=(1, 1, 1))
#            mesh = vec.basis.mesh
#            Plotter.plotMesh(mesh.coordinates(), mesh.cells(), representation='mesh')
#            Plotter.axes()
#            Plotter.labels()
#            Plotter.title(str(mu))
            # function
            Plotter.figure(bgcolor=(1, 1, 1))
            mesh = vec.basis.mesh
            Plotter.plotMesh(mesh.coordinates(), mesh.cells(), vec.coeffs)
            Plotter.axes()
            Plotter.labels()
            Plotter.title(str(mu))
        else:
            plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
            vec.plot(title=str(mu), interactive=False)
    if USE_MAYAVI:
        Plotter.show(stop=True)
        Plotter.close(allfig=True)
    else:
        interactive()
