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


def run_SFEM(opts, conf):
#    option_defs = (("SFEM",
#                     {"problem_type":1,
#                         "domain":0,
#                         "boundary_type":1,
#                         "assemble_type":0,
#                         "FEM_degree":1,
#                         "decay_exp":1,
#                         "coeff_type":1,
#                         "coeff_scale":2,
#                         "freq_scale":2,
#                         "freq_skip":1,
#                         "gamma":2}),
#                   ("SFEM adaptive algorithm",
#                     {"iterations":1,
#                         "uniform_refinement":3,
#                         "initial_Lambda":1,
#                         "refine_residual":3,
#                         "refine_projection":3,
#                         "refine_Lambda":3,
#                         "cQ":2,
#                         "ceta":2,
#                         "theta_eta":2,
#                         "theta_zeta":2,
#                         "min_zeta":2,
#                         "maxh":2,
#                         "newmi_add_maxm":1,
#                         "theta_delta":2,
#                         "max_Lambda_frac":2,
#                         "quadrature_degree":1,
#                         "projection_degree_increase":1,
#                         "refine_projection_mesh":1,
#                         "pcg_eps":2,
#                         "pcg_maxiter":1,
#                         "error_eps":2}))

    # propagate config values
    for sec in conf.keys():
        if sec == "LOGGING":
            continue
        secconf = conf[sec]
        for key, val in secconf.iteritems():
            print "CONF_" + key + "= secconf['" + key + "']"
            exec "CONF_" + key + "= secconf['" + key + "']"

    # setup logging
    exec "LOG_LEVEL = logging." + conf["LOGGING"]["level"]
    logger = setup_logging(LOG_LEVEL)
    
    # determine path of this module
    path = os.path.dirname(__file__)
    
    # ============================================================
    # PART A: Simulation Options
    # ============================================================
    
    # flag for final solution export
    if not opts.saveData:
        SAVE_SOLUTION = ''
    else:
        SAVE_SOLUTION = os.path.join(opts.basedir, "SFEM-results")
    
    # flags for residual, projection, new mi refinement 
    REFINEMENT = {"RES":CONF_refine_residual, "PROJ":CONF_refine_project, "MI":CONF_refine_Lambda}
    
    # initial mesh elements
    initial_mesh_N = 10
    
    # ============================================================
    # PART B: Problem Setup
    # ============================================================
    
    # define initial multiindices
    mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(CONF_initial_Lambda, 1)]
    
    # setup domain and meshes
    mesh0, boundaries, dim = SampleDomain.setupDomain(CONF_domain, initial_mesh_N=initial_mesh_N)
    #meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.4, 0.3))
    meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)
    
    # define coefficient field
    # NOTE: for proper treatment of corner points, see elasticity_residual_estimator
    coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
    coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                        freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale)
    a0 = coeff_field.mean_func


    # TODO: define bc in sample problem class!
    
    
    # setup boundary conditions
    Dirichlet_boundary = None
    uD = None
    Neumann_boundary = None
    g = None
    if CONF_problem_type == 1:
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
        assert CONF_problem_type == 0
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
    
    # NOTE: for Cook's membrane, the mesh refinement gets stuck for some reason...
    if domaintype == 2:
        maxh = 0.0
        MC_HMAX = 0
    
    # refinement loop
    # ===============
    w0 = w
    w, sim_stats = AdaptiveSolver(A, coeff_field, pde, mis, w0, mesh0, degree, gamma=CONF_gamma, cQ=CONF_cQ, ceta=CONF_ceta,
                        # marking parameters
                        theta_eta=CONF_theta_eta, theta_zeta=CONF_theta_zeta, min_zeta=CONF_min_zeta,
                        maxh=CONF_maxh, newmi_add_maxm=CONF_newmi_add_maxm, theta_delta=CONF_theta_delta,
                        max_Lambda_frac=CONF_max_Lambda_frac,
                        # residual error evaluation
                        quadrature_degree=CONF_quadrature_degree,
                        # projection error evaluation
                        projection_degree_increase=CONF_projection_degree_increase, refine_projection_mesh=CONF_refine_projection_mesh,
                        # pcg solver
                        pcg_eps=CONF_pcg_eps, pcg_maxiter=CONF_pcg_maxiter,
                        # adaptive algorithm threshold
                        error_eps=CONF_error_eps,
                        # refinements
                        max_refinements=CONF_max_refinements, do_refinement=REFINEMENT, do_uniform_refinement=CONF_uniform_refinement,
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
    # PART E: Plotting
    # ============================================================
    
    # plot residuals
    if opts.plotEstimator and len(sim_stats) > 1:
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
            mi = [s["MI"] for s in sim_stats]
            num_mi = [len(m) for m in mi]
            reserrmu = defaultdict(list)
            for rem in _reserrmu:
                for mu, v in rem:
                    reserrmu[mu].append(v)
            print "errest", errest
                
            # --------
            # figure 1
            # --------
            fig1 = figure()
            fig1.suptitle("residual estimator")
            ax = fig1.add_subplot(111)
            if REFINEMENT["MI"]:
                ax.loglog(x, num_mi, '--y+', label='active mi')
            ax.loglog(x, errest, '-g<', label='error estimator')
            ax.loglog(x, reserr, '-.cx', label='residual part')
            ax.loglog(x[1:], projerr[1:], '-.m>', label='projection part')
            legend(loc='upper right')
    
            # --------
            # figure 2
            # --------
            fig1 = figure()
            fig1.suptitle("efficiency residual estimator")
            ax = fig1.add_subplot(111)
            ax.loglog(x, errest, '-g<', label='error estimator')
            legend(loc='upper right')
    
            # --------
            # figure 3
            # --------
            fig3 = figure()
            fig3.suptitle("residual contributions")
            ax = fig3.add_subplot(111)
            for mu, v in reserrmu.iteritems():
                ms = str(mu)
                ms = ms[ms.find('=') + 1:-1]
                ax.loglog(x[-len(v):], v, '-g<', label=ms)
            legend(loc='upper right')
            
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
    if opts.plotSolution:
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
