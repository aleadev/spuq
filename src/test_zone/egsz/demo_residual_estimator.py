from __future__ import division
import logging
import os
import functools
from math import sqrt
from collections import defaultdict

from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, ASSEMBLY_TYPE
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
pdetype = 0
domaintype = 0
domains = ('square', 'lshape', 'cooks')
domain = domains[domaintype]

# decay exponent
decay_exp = 4

# refinements
max_refinements = 3

# polynomial degree of FEM approximation
degree = 1

# multioperator assembly type
assembly_type = ASSEMBLY_TYPE.MU #JOINT_GLOBAL #JOINT_MU

# flag for residual graph plotting
PLOT_RESIDUAL = True

# flag for final mesh plotting
PLOT_MESHES = True

# flag for (sample) solution plotting
PLOT_SOLUTION = True

# flag for final solution export
SAVE_SOLUTION = ''
#SAVE_SOLUTION = os.path.join(os.path.dirname(__file__), "results/demo-residual-A2-neumann")

# flags for residual, projection, new mi refinement 
REFINEMENT = {"RES":True, "PROJ":True, "MI":True}
UNIFORM_REFINEMENT = False

# initial mesh elements
initial_mesh_N = 10

# MC error sampling
MC_RUNS = 0
MC_N = 3
MC_HMAX = 3 / 10

# ============================================================
# PART B: Problem Setup
# ============================================================

# define initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(3, 1)]

# debug---
#mis = [mis[0]]
#mis = [Multiindex(), ]
#mis = [Multiindex(), Multiindex([1])]
# ---debug

# setup domain and meshes
mesh0, boundaries, dim = SampleDomain.setupDomain(domain, initial_mesh_N=initial_mesh_N)
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
coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
gamma = 0.9
if pdetype == 0:
    coeff_field = SampleProblem.setupCF(coeff_types[1], decayexp=decay_exp, gamma=gamma, freqscale=1, freqskip=0, rvtype="uniform")
else:
    coeff_field = SampleProblem.setupCF(coeff_types[1], decayexp=decay_exp, gamma=gamma, freqscale=1, freqskip=0, rvtype="uniform", scale=1e5)
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
    uD = (Constant((0.0, 0.0)), Constant((0.3, 0.0)))
#    Dirichlet_boundary = (boundaries['left'], boundaries['right'])
#    uD = (Constant((0.0, 0.0)), Constant((1.0, 1.0)))
    # homogeneous Neumann does not have to be set explicitly
    Neumann_boundary = None # (boundaries['right'])
    g = None #Constant((0.0, 10.0))
    # create pde instance
    pde = FEMNavierLame(mu=1e2, lmbda0=a0,
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
#    Dirichlet_boundary = (boundaries['left'], boundaries['right'], boundaries['top'], boundaries['bottom'])
#    uD = (Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0))
    # 2 Dirichlet
    Dirichlet_boundary = (boundaries['left'], boundaries['right'])
    uD = (Constant(0.0), Constant(3.0))
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
A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs, assembly_type=assembly_type)

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
theta_eta = 0.4         # residual marking bulk parameter
theta_zeta = 0.01        # projection marking threshold factor
min_zeta = 1e-10        # minimal projection error considered
maxh = 1 / 10           # maximal mesh width for projection maximum norm evaluation
newmi_add_maxm = 10     # maximal search length for new new multiindices (to be added to max order of solution)
theta_delta = 0.95      # number new multiindex activation bound
max_Lambda_frac = 1 / 10 # fraction of |Lambda| for max number of new multiindices
# residual error evaluation
quadrature_degree = 2
# projection error evaluation
projection_degree_increase = 2
refine_projection_mesh = 2
# pcg solver
pcg_eps = 1e-6
pcg_maxiter = 100
error_eps = 1e-5

if MC_RUNS > 0 or True:
    w_history = []
else:
    w_history = None



def traceit(frame, event, arg):
    filename = frame.f_code.co_filename
    funcname = frame.f_code.co_name
    lineno = frame.f_lineno

    if event == "return" and funcname == "pcg":
        w = arg[0]
        plot(w[Multiindex()]._fefunc, title="Foo", interactive=False, wireframe=True)
        #plot( w[Multiindex()]._fefunc, title="Foo", mode="displacement", interactive=False, wireframe=True)


    return traceit

import sys
print sys.settrace(traceit)



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


# ============================================================
# PART D: Export of Solution
# ============================================================
# NOTE: save at this point since MC tends to run out of memory
if SAVE_SOLUTION != "":
    # save solution (also creates directory if not existing)
    w.pickle(SAVE_SOLUTION)
    # save simulation data
    import pickle
    with open(os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'), 'wb') as fout:
        pickle.dump(sim_stats, fout)


# ============================================================
# PART E: MC Error Sampling
# ============================================================
if MC_RUNS > 0:
    ref_maxm = w_history[-1].max_order
    for i, w in enumerate(w_history):
        if i == 0:
            continue
        logger.info("MC error sampling for w[%i] (of %i)", i, len(w_history))
        # memory usage info
        logger.info("\n======================================\nMEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) + "\n======================================\n")
        L2err, H1err, L2err_a0, H1err_a0 = sample_error_mc(w, pde, A, coeff_field, mesh0, ref_maxm, MC_RUNS, MC_N, MC_HMAX)
        sim_stats[i - 1]["MC-L2ERR"] = L2err
        sim_stats[i - 1]["MC-H1ERR"] = H1err
        sim_stats[i - 1]["MC-L2ERR_a0"] = L2err_a0
        sim_stats[i - 1]["MC-H1ERR_a0"] = H1err_a0


# ============================================================
# PART F: Export Updated Data and Plotting
# ============================================================
# save updated data
if SAVE_SOLUTION != "":
    # save updated statistics
    import pickle
    with open(os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'), 'wb') as fout:
        pickle.dump(sim_stats, fout)

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
        _reserrmu = [s["RES-mu"] for s in sim_stats]
        _projerrmu = [s["PROJ-mu"] for s in sim_stats]
        if MC_RUNS > 0:
            mcL2 = [s["MC-L2ERR"] for s in sim_stats]
            mcH1 = [s["MC-H1ERR"] for s in sim_stats]
            mcL2_a0 = [s["MC-L2ERR_a0"] for s in sim_stats]
            mcH1_a0 = [s["MC-H1ERR_a0"] for s in sim_stats]
            effest = [est / err for est, err in zip(errest, mcH1)]
        mi = [s["MI"] for s in sim_stats]
        num_mi = [len(m) for m in mi]
        reserrmu = defaultdict(list)
        for rem in _reserrmu:
            for mu, v in rem:
                reserrmu[mu].append(v)
        print "errest", errest
        if MC_RUNS > 0:
            print "mcH1", mcH1
            print "efficiency", [est / err for est, err in zip(errest, mcH1)]
            
        # figure 1
        # --------
#        fig = figure()
#        fig.suptitle("error")
#        ax = fig.add_subplot(111)
#        ax.loglog(x, H1, '-g<', label='H1 residual')
#        ax.loglog(x, L2, '-c+', label='L2 residual')
#        ax.loglog(x, mcH1, '-b^', label='MC H1 error')
#        ax.loglog(x, mcL2, '-ro', label='MC L2 error')
#        ax.loglog(x, mcH1_a0, '-.b^', label='MC H1 error a0')
#        ax.loglog(x, mcL2_a0, '-.ro', label='MC L2 error a0')
#        legend(loc='upper right')
#        if SAVE_SOLUTION != "":
#            fig.savefig(os.path.join(SAVE_SOLUTION, 'RES.png'))

        # --------
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
        if MC_RUNS > 0:
            ax.loglog(x, mcH1, '-b^', label='MC H1 error')
            ax.loglog(x, mcL2, '-ro', label='MC L2 error')
#        ax.loglog(x, H1, '-b^', label='H1 residual')
#        ax.loglog(x, L2, '-ro', label='L2 residual')
        legend(loc='upper right')
        if SAVE_SOLUTION != "":
            fig2.savefig(os.path.join(SAVE_SOLUTION, 'EST.png'))
            fig2.savefig(os.path.join(SAVE_SOLUTION, 'EST.eps'))

        # --------
        # figure 3
        # --------
        fig3 = figure()
        fig3.suptitle("efficiency residual estimator")
        ax = fig3.add_subplot(111)
        ax.loglog(x, errest, '-g<', label='error estimator')
        if MC_RUNS > 0:
            ax.loglog(x, mcH1, '-b^', label='MC H1 error')
            ax.loglog(x, effest, '-ro', label='efficiency')        
        legend(loc='upper right')
        if SAVE_SOLUTION != "":
            fig3.savefig(os.path.join(SAVE_SOLUTION, 'ESTEFF.png'))
            fig3.savefig(os.path.join(SAVE_SOLUTION, 'ESTEFF.eps'))

        # --------
        # figure 4
        # --------
        fig4 = figure()
        fig4.suptitle("residual contributions")
        ax = fig4.add_subplot(111)
        for mu, v in reserrmu.iteritems():
            ms = str(mu)
            ms = ms[ms.find('=') + 1:-1]
            ax.loglog(x[-len(v):], v, '-g<', label=ms)
        legend(loc='upper right')
#        if SAVE_SOLUTION != "":
#            fig4.savefig(os.path.join(SAVE_SOLUTION, 'RESCONTRIB.png'))
#            fig4.savefig(os.path.join(SAVE_SOLUTION, 'RESCONTRIB.eps'))
        
        show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        logger.info("skipped plotting since matplotlib is not available...")

# plot final meshes
if PLOT_MESHES:
    USE_MAYAVI = Plotter.hasMayavi() and False
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
            viz_mesh = plot(vec.basis.mesh, title="mesh " + str(mu), interactive=False, axes=True)
            if SAVE_SOLUTION != '':
                viz_mesh.write_png(SAVE_SOLUTION + '/mesh' + str(mu) + '.png')
                viz_mesh.write_ps(SAVE_SOLUTION + '/mesh' + str(mu), format='pdf')
#            vec.plot(title=str(mu), interactive=False)
    if USE_MAYAVI:
        Plotter.show(stop=True)
        Plotter.close(allfig=True)
    else:
        interactive()

# plot sample solution
if PLOT_SOLUTION:
    # get random field sample and evaluate solution (direct and parametric)
    RV_samples = coeff_field.sample_rvs()
    ref_maxm = w_history[-1].max_order
    sub_spaces = w[Multiindex()].basis.num_sub_spaces
    degree = w[Multiindex()].basis.degree
    maxh = min(w[Multiindex()].basis.minh / 4, MC_HMAX)
    maxh = w[Multiindex()].basis.minh
    projection_basis = get_projection_basis(mesh0, maxh=maxh, degree=degree, sub_spaces=sub_spaces)
    sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
    sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis)
    sol_variance = compute_solution_variance(coeff_field, w, projection_basis)

    # plot
    print sub_spaces
    if sub_spaces == 0:
        viz_p = plot(sample_sol_param._fefunc, title="parametric solution", axes=True)
        viz_d = plot(sample_sol_direct._fefunc, title="direct solution", axes=True)
        if ref_maxm > 0:
            viz_v = plot(sol_variance._fefunc, title="solution variance", axes=True)

        # debug---
        if not True:        
            for mu in w.active_indices():
                for i, wi in enumerate(w_history):
                    if i == len(w_history) - 1 or True:
                        plot(wi[mu]._fefunc, title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
#                        plot(wi[mu]._fefunc.function_space().mesh(), title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
                interactive()
        # ---debug
        
        for mu in w.active_indices():
            plot(w[mu]._fefunc, title="parametric solution " + str(mu), axes=True)
    else:
        mesh_param = sample_sol_param._fefunc.function_space().mesh()
        mesh_direct = sample_sol_direct._fefunc.function_space().mesh()
        wireframe = True
        viz_p = plot(sample_sol_param._fefunc, title="parametric solution", mode="displacement", mesh=mesh_param, wireframe=wireframe)#, rescale=False)
        viz_d = plot(sample_sol_direct._fefunc, title="direct solution", mode="displacement", mesh=mesh_direct, wireframe=wireframe)#, rescale=False)
        
        for mu in w.active_indices():
            viz_p = plot(w[mu]._fefunc, title="parametric solution: " + str(mu), mode="displacement", mesh=mesh_param, wireframe=wireframe)
    interactive()

if SAVE_SOLUTION != '':
    logger.info("exported solution to " + SAVE_SOLUTION)
