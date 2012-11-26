from __future__ import division
import os
import functools
import logging
from math import sqrt
from collections import defaultdict

from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
from spuq.application.egsz.multi_operator import MultiOperator, ASSEMBLY_TYPE
from spuq.application.egsz.sample_problems import SampleProblem
from spuq.application.egsz.sample_domains import SampleDomain
from spuq.application.egsz.mc_error_sampling import sample_error_mc
from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
from spuq.application.egsz.sampling import get_projection_basis
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.math_utils.multiindex import Multiindex
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.plot.plotter import Plotter
try:
    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
                        plot, interactive, set_log_level, set_log_active)
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    from spuq.application.egsz.egsz_utils import setup_logging, stats_plotter
except:
    import traceback
    print traceback.format_exc()
    print "FEniCS has to be available"
    os.sys.exit(1)

# ------------------------------------------------------------
logger = logging.getLogger(__name__)

def run_SFEM(opts, conf):
    # propagate config values
    for sec in conf.keys():
        if sec == "LOGGING":
            continue
        secconf = conf[sec]
        for key, val in secconf.iteritems():
            print "CONF_" + key + "= secconf['" + key + "'] =", secconf[key]
            exec "CONF_" + key + "= secconf['" + key + "']"

    # setup logging
    print "LOG_LEVEL = logging." + conf["LOGGING"]["level"]
    exec "LOG_LEVEL = logging." + conf["LOGGING"]["level"]
    setup_logging(LOG_LEVEL, logfile=CONF_experiment_name + "_SFEM")
    
    # determine path of this module
    path = os.path.dirname(__file__)
    
    # ============================================================
    # PART A: Simulation Options
    # ============================================================
    
    # flags for residual, projection, new mi refinement 
    REFINEMENT = {"RES":CONF_refine_residual, "PROJ":CONF_refine_projection, "MI":CONF_refine_Lambda}

    
    # ============================================================
    # PART B: Problem Setup
    # ============================================================
    
    # define initial multiindices
    mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(CONF_initial_Lambda, 1)]
    
    # setup domain and meshes
    mesh0, boundaries, dim = SampleDomain.setupDomain(CONF_domain, initial_mesh_N=CONF_initial_mesh_N)
    #meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=10, randref=(0.4, 0.3))
    meshes = SampleProblem.setupMeshes(mesh0, len(mis), num_refine=0)
    
    # define coefficient field
    # NOTE: for proper treatment of corner points, see elasticity_residual_estimator
    coeff_types = ("EF-square-cos", "EF-square-sin", "monomials", "constant")
    from itertools import count
    if CONF_mu is not None:
        muparam = (CONF_mu, (0 for _ in count()))
    else:
        muparam = None 
    coeff_field = SampleProblem.setupCF(coeff_types[CONF_coeff_type], decayexp=CONF_decay_exp, gamma=CONF_gamma,
                                    freqscale=CONF_freq_scale, freqskip=CONF_freq_skip, rvtype="uniform", scale=CONF_coeff_scale, secondparam=muparam)

    # setup boundary conditions and pde
    pde, Dirichlet_boundary, uD, Neumann_boundary, g, f = SampleProblem.setupPDE(CONF_boundary_type, CONF_domain, CONF_problem_type, boundaries, coeff_field)

    # define multioperator
    A = MultiOperator(coeff_field, pde.assemble_operator, pde.assemble_operator_inner_dofs, assembly_type=eval("ASSEMBLY_TYPE." + CONF_assembly_type))

    # setup initial solution multivector
    w = SampleProblem.setupMultiVector(dict([(mu, m) for mu, m in zip(mis, meshes)]), functools.partial(setup_vector, pde=pde, degree=CONF_FEM_degree))
    logger.info("active indices of w after initialisation: %s", w.active_indices())

    sim_stats = None
    w_history = []
    if opts.continueSFEM:
        try:
            logger.info("CONTINUIING EXPERIMENT: loading previous data of %s...", CONF_experiment_name)
            import pickle
            LOAD_SOLUTION = os.path.join(opts.basedir, CONF_experiment_name)
            logger.info("loading solutions from %s" % os.path.join(LOAD_SOLUTION, 'SFEM-SOLUTIONS.pkl'))
            # load solutions
            with open(os.path.join(LOAD_SOLUTION, 'SFEM-SOLUTIONS.pkl'), 'rb') as fin:
                w_history = pickle.load(fin)
            # convert to MultiVectorWithProjection
            for i, mv in enumerate(w_history):
                w_history[i] = MultiVectorWithProjection(cache_active=True, multivector=w_history[i])
            # load simulation data
            logger.info("loading statistics from %s" % os.path.join(LOAD_SOLUTION, 'SIM-STATS.pkl'))
            with open(os.path.join(LOAD_SOLUTION, 'SIM-STATS.pkl'), 'rb') as fin:
                sim_stats = pickle.load(fin)
            logger.info("active indices of w after initialisation: %s", w_history[-1].active_indices())
            w0 = w_history[-1]
        except:
            logger.warn("FAILED LOADING EXPERIMENT %s --- STARTING NEW DATA", CONF_experiment_name)
            w0 = w    
    else:
        w0 = w

    
    # ============================================================
    # PART C: Adaptive Algorithm
    # ============================================================
    
    # refinement loop
    # ===============
    w, sim_stats = AdaptiveSolver(A, coeff_field, pde, mis, w0, mesh0, CONF_FEM_degree, gamma=CONF_gamma, cQ=CONF_cQ, ceta=CONF_ceta,
                        # marking parameters
                        theta_eta=CONF_theta_eta, theta_zeta=CONF_theta_zeta, min_zeta=CONF_min_zeta,
                        maxh=CONF_maxh, newmi_add_maxm=CONF_newmi_add_maxm, theta_delta=CONF_theta_delta,
                        marking_strategy=CONF_marking_strategy, max_Lambda_frac=CONF_max_Lambda_frac,
                        # residual error evaluation
                        quadrature_degree=CONF_quadrature_degree,
                        # projection error evaluation
                        projection_degree_increase=CONF_projection_degree_increase, refine_projection_mesh=CONF_refine_projection_mesh,
                        # pcg solver
                        pcg_eps=CONF_pcg_eps, pcg_maxiter=CONF_pcg_maxiter,
                        # adaptive algorithm threshold
                        error_eps=CONF_error_eps,
                        # refinements
                        max_refinements=CONF_iterations, do_refinement=REFINEMENT, do_uniform_refinement=CONF_uniform_refinement,
                        w_history=w_history,
                        sim_stats=sim_stats)
    
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
    # PART D: Export of Solutions and Simulation Data
    # ============================================================
        
    # flag for final solution export
    if opts.saveData:
        import pickle
        SAVE_SOLUTION = os.path.join(opts.basedir, CONF_experiment_name)
        try:
            os.makedirs(SAVE_SOLUTION)
        except:
            pass
        logger.info("saving solutions into %s" % os.path.join(SAVE_SOLUTION, 'SFEM-SOLUTIONS.pkl'))
        # save solutions
        with open(os.path.join(SAVE_SOLUTION, 'SFEM-SOLUTIONS.pkl'), 'wb') as fout:
            pickle.dump(w_history, fout)
        # save simulation data
        logger.info("saving statistics into %s" % os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'))
        with open(os.path.join(SAVE_SOLUTION, 'SIM-STATS.pkl'), 'wb') as fout:
            pickle.dump(sim_stats, fout)
    
    
    # ============================================================
    # PART E: Plotting
    # ============================================================
    
    # plot residuals
    if opts.plotEstimator and len(sim_stats) > 1:
        try:
            from matplotlib.pyplot import figure, show, legend
            X = [s["DOFS"] for s in sim_stats]
            print "DOFS", X
            L2 = [s["L2"] for s in sim_stats]
            H1 = [s["H1"] for s in sim_stats]
            errest = [sqrt(s["EST"]) for s in sim_stats]
            res_part = [s["RES-PART"] for s in sim_stats]
            proj_part = [s["PROJ-PART"] for s in sim_stats]
            pcg_part = [s["PCG-PART"] for s in sim_stats]
            _reserrmu = [s["RES-mu"] for s in sim_stats]
            _projerrmu = [s["PROJ-mu"] for s in sim_stats]
            proj_max_zeta = [s["PROJ-MAX-ZETA"] for s in sim_stats]
            proj_max_inactive_zeta = [s["PROJ-MAX-INACTIVE-ZETA"] for s in sim_stats]
            try:
                proj_inactive_zeta = sorted([v for v in sim_stats[-2]["PROJ-INACTIVE-ZETA"].values()], reverse=True)
            except:
                proj_inactive_zeta = None
            mi = [s["MI"] for s in sim_stats]
            num_mi = [len(m) for m in mi]
            time_pcg = [s["TIME-PCG"] for s in sim_stats]
            time_estimator = [s["TIME-ESTIMATOR"] for s in sim_stats]
            time_inactive_mi = [s["TIME-INACTIVE-MI"] for s in sim_stats]
            time_marking = [s["TIME-MARKING"] for s in sim_stats]
            reserrmu = defaultdict(list)
            for rem in _reserrmu:
                for mu, v in rem:
                    reserrmu[mu].append(v)
            projerrmu = defaultdict(list)
            for pem in _projerrmu:
                for mu, v in pem:
                    projerrmu[mu].append(v)
            print "errest", errest
    
            # --------
            # figure 2
            # --------
            fig2 = figure()
            fig2.suptitle("error estimator")
            ax = fig2.add_subplot(111)
            ax.loglog(X, errest, '-g<', label='error estimator')
            legend(loc='upper right')
    
            # --------
            # figure 3a
            # --------
            if opts.plotEstimatorAll:
                max_mu_plotting = 7
                fig3 = figure()
                fig3.suptitle("residual contributions")
                ax = fig3.add_subplot(111)
                for i, muv in enumerate(reserrmu.iteritems()):
                    mu, v = muv
                    if i < max_mu_plotting:
                        mu, v = muv
                        ms = str(mu)
                        ms = ms[ms.find('=') + 1:-1]
                        ax.loglog(X[-len(v):], v, '-g<', label=ms)
                legend(loc='upper right')
    
            # --------
            # figure 3b
            # --------
            if opts.plotEstimatorAll:
                fig3b = figure()
                fig3b.suptitle("projection contributions")
                ax = fig3b.add_subplot(111)
                for i, muv in enumerate(projerrmu.iteritems()):
                    mu, v = muv
                    if max(v) > 1e-10 and i < max_mu_plotting:
                        ms = str(mu)
                        ms = ms[ms.find('=') + 1:-1]
                        ax.loglog(X[-len(v):], v, '-g<', label=ms)
                legend(loc='upper right')
    
            # --------
            # figure 4
            # --------
            if opts.plotEstimatorAll:
                fig4 = figure()
                fig4.suptitle("projection $\zeta$")
                ax = fig4.add_subplot(111)
                ax.loglog(X[1:], proj_max_zeta[1:], '-g<', label='max active $\zeta$')
                ax.loglog(X[1:], proj_max_inactive_zeta[1:], '-b^', label='max inactive $\zeta$')
                legend(loc='upper right')
    
            # --------
            # figure 5
            # --------
            fig5 = figure()
            fig5.suptitle("timings")
            ax = fig5.add_subplot(111)
            ax.loglog(X, time_pcg, '-g<', label='pcg')
            ax.loglog(X, time_estimator, '-b^', label='estimator')
            ax.loglog(X, time_inactive_mi, '-c+', label='inactive_mi')
            ax.loglog(X, time_marking, '-ro', label='marking')
            legend(loc='upper right')
                
            # --------
            # figure 6
            # --------
            if opts.plotEstimatorAll:
                fig6 = figure()
                fig6.suptitle("projection error")
                ax = fig6.add_subplot(111)
                ax.loglog(X[1:], proj_part[1:], '-.m>', label='projection part')
                legend(loc='upper right')
                
            # --------
            # figure 7
            # --------
            if opts.plotEstimatorAll and proj_inactive_zeta is not None:
                fig7 = figure()
                fig7.suptitle("inactive multiindex $\zeta$")
                ax = fig7.add_subplot(111)
                ax.loglog(range(len(proj_inactive_zeta)), proj_inactive_zeta, '-.m>', label='inactive $\zeta$')
                legend(loc='lower right')
                
            # --------
            # figure 1
            # --------
            fig1 = figure()
            fig1.suptitle("residual estimator")
            ax = fig1.add_subplot(111)
            if REFINEMENT["MI"]:
                ax.loglog(X, num_mi, '--y+', label='active mi')
            ax.loglog(X, errest, '-g<', label='error estimator')
            ax.loglog(X, res_part, '-.cx', label='residual part')
            ax.loglog(X[1:], proj_part[1:], '-.m>', label='projection part')
            ax.loglog(X, pcg_part, '-.b>', label='pcg part')
            legend(loc='upper right')
            
            show()  # this invalidates the figure instances...
        except:
            import traceback
            print traceback.format_exc()
            logger.info("skipped plotting since matplotlib is not available...")
    
    # plot final meshes
    if opts.plotMesh:
        USE_MAYAVI = Plotter.hasMayavi() and False
        w = w_history[-1]
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
                viz_mesh = plot(vec.basis.mesh, title="mesh " + str(mu), interactive=False)
#                if SAVE_SOLUTION != '':
#                    viz_mesh.write_png(SAVE_SOLUTION + '/mesh' + str(mu) + '.png')
#                    viz_mesh.write_ps(SAVE_SOLUTION + '/mesh' + str(mu), format='pdf')
    #            vec.plot(title=str(mu), interactive=False)
        if USE_MAYAVI:
            Plotter.show(stop=True)
            Plotter.close(allfig=True)
        else:
            interactive()
    
    # plot sample solution
    if opts.plotSolution:
        w = w_history[-1]
        # get random field sample and evaluate solution (direct and parametric)
        RV_samples = coeff_field.sample_rvs()
        ref_maxm = w_history[-1].max_order
        sub_spaces = w[Multiindex()].basis.num_sub_spaces
        degree = w[Multiindex()].basis.degree
        maxh = min(w[Multiindex()].basis.minh / 4, CONF_max_h)
        maxh = w[Multiindex()].basis.minh
        projection_basis = get_projection_basis(mesh0, maxh=maxh, degree=degree, sub_spaces=sub_spaces)
        sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
        sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis)
        sol_variance = compute_solution_variance(coeff_field, w, projection_basis)
    
        # plot
        print sub_spaces
        if sub_spaces == 0:
            viz_p = plot(sample_sol_param._fefunc, title="parametric solution")
            viz_d = plot(sample_sol_direct._fefunc, title="direct solution")
            if ref_maxm > 0:
                viz_v = plot(sol_variance._fefunc, title="solution variance")
    
            # debug---
            if not True:        
                for mu in w.active_indices():
                    for i, wi in enumerate(w_history):
                        if i == len(w_history) - 1 or True:
                            plot(wi[mu]._fefunc, title="parametric solution " + str(mu) + " iteration " + str(i))
    #                        plot(wi[mu]._fefunc.function_space().mesh(), title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
                    interactive()
            # ---debug
            
#            for mu in w.active_indices():
#                plot(w[mu]._fefunc, title="parametric solution " + str(mu))
        else:
            mesh_param = sample_sol_param._fefunc.function_space().mesh()
            mesh_direct = sample_sol_direct._fefunc.function_space().mesh()
            wireframe = True
            viz_p = plot(sample_sol_param._fefunc, title="parametric solution", mode="displacement", mesh=mesh_param, wireframe=wireframe)#, rescale=False)
            viz_d = plot(sample_sol_direct._fefunc, title="direct solution", mode="displacement", mesh=mesh_direct, wireframe=wireframe)#, rescale=False)
            
#            for mu in w.active_indices():
#                viz_p = plot(w[mu]._fefunc, title="parametric solution: " + str(mu), mode="displacement", mesh=mesh_param, wireframe=wireframe)
        interactive()
