from __future__ import division

import optparse
import numpy as np
import os
from math import sqrt
from collections import defaultdict
import matplotlib.pyplot as plt

#from spuq.application.egsz.adaptive_solver import AdaptiveSolver, setup_vector
#from spuq.application.egsz.multi_operator import MultiOperator, ASSEMBLY_TYPE
#from spuq.application.egsz.sample_problems import SampleProblem
#from spuq.application.egsz.sample_domains import SampleDomain
#from spuq.application.egsz.mc_error_sampling import sample_error_mc
#from spuq.application.egsz.sampling import compute_parametric_sample_solution, compute_direct_sample_solution, compute_solution_variance
#from spuq.application.egsz.sampling import get_projection_basis
#from spuq.math_utils.multiindex import Multiindex
#from spuq.math_utils.multiindex_set import MultiindexSet
#from spuq.utils.plot.plotter import Plotter
#try:
#    from dolfin import (Function, FunctionSpace, Mesh, Constant, UnitSquare, compile_subdomains,
#                        plot, interactive, set_log_level, set_log_active)
#    from spuq.fem.fenics.fenics_vector import FEniCSVector
#    from spuq.application.egsz.egsz_utils import setup_logging, stats_plotter
#except:
#    import traceback
#    print traceback.format_exc()
#    print "FEniCS has to be available"
#    os.sys.exit(1)

# ------------------------------------------------------------

# ==================
# A Parse Arguments
# ==================
usage = "%prog [options] experiment_directory"
optparser = optparse.OptionParser(usage=usage)

optparser.add_option('--withFigures', '--with-figures',
                     action='store_true', default=False, dest='withFigures',
                     help='export several figures')

optparser.add_option('--showFigures', '--show-figures',
                     action='store_true', default=False, dest='showFigures',
                     help='show several figures')

optparser.add_option('--withMesh', '--with-mesh',
                     action='store_true', default=False, dest='withMesh',
                     help='export meshes')

optparser.add_option('--withSolution', '--with-solution',
                     action='store_true', default=False, dest='withSolution',
                     help='export sample parametric and direct solution')

optparser.add_option('--withMI', '--with-mi',
                     action='store_true', default=False, dest='withMI',
                     help='export multiindex statistics')

options, args = optparser.parse_args()
if len(args) < 1:
    optparser.error('No experiment directory specified (use -h/--help for help)')
elif len(args) > 2:
    optparser.error('Too many arguments (use -h/--help for help)')
else:
    options.experiment_dir = args[0]
    if len(args) > 1:
        options.iteration_level = args[1]
    else:
        options.iteration_level = -1

# ==================
# B Import Solution
# ==================
import pickle
print "loading solutions from %s" % os.path.join(options.experiment_dir, 'SFEM-SOLUTIONS.pkl')
# load solutions
with open(os.path.join(options.experiment_dir, 'SFEM-SOLUTIONS.pkl'), 'rb') as fin:
    w_history = pickle.load(fin)
# load simulation data
print "loading statistics from %s" % os.path.join(options.experiment_dir, 'SIM-STATS.pkl')
with open(os.path.join(options.experiment_dir, 'SIM-STATS.pkl'), 'rb') as fin:
    sim_stats = pickle.load(fin)
print "sim_stats has %s iterations" % len(sim_stats)


# ==================
# C Generate Figures
# ==================
if options.withFigures and len(sim_stats) > 1:
    try:
        # prepare data
        x = [s["DOFS"] for s in sim_stats]
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
        for rem in _projerrmu:
            for mu, v in rem:
                projerrmu[mu].append(v)
        print "ERROR ESTIMATOR", errest
            
        # --------
        # figure 1
        # --------
        fig1 = plt.figure()
        fig1.suptitle("residual estimator")
        ax = fig1.add_subplot(111)
        ax.loglog(x, num_mi, '--y+', label='active mi', linewidth=1.5)
        ax.loglog(x, errest, '-g<', label='estimator', linewidth=1.5)
        ax.loglog(x, res_part, '-.cx', label='residual', linewidth=1.5)
        ax.loglog(x[1:], proj_part[1:], '-.m>', label='projection', linewidth=1.5)
        ax.loglog(x, pcg_part, '-.b>', label='pcg', linewidth=1.5)
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error (number active multiindices)")
            
#        t = np.arange(1, num_mi[-1] * 2, 1)
#        ax2 = ax.twinx()
#        ax.plot(x, num_mi, '--y+', label='active mi', linewidth=1.5)
#        ax2.set_ylabel('number active multiindices')
##        for tl in ax2.get_yticklabels():
##            tl.set_color('b')

        plt.axhline(y=0)
        plt.axvline(x=0)
        ax.grid(True)
        leg = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05))
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        fig1.savefig(os.path.join(options.experiment_dir, 'fig1-estimator-all.pdf'))
        fig1.savefig(os.path.join(options.experiment_dir, 'fig1-estimator-all.png'))
    
        # --------
        # figure 2
        # --------
        fig2 = plt.figure()
        fig2.suptitle("efficiency residual estimator")
        ax = fig2.add_subplot(111)
        ax.loglog(x, errest, '-g<', label='error estimator')
        # TODO: MC ERROR and EFFICIENCY
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error (efficiency)")
        leg = plt.legend(loc='upper right')
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        ax.grid(True)
        fig2.savefig(os.path.join(options.experiment_dir, 'fig2-estimator.pdf'))
        fig2.savefig(os.path.join(options.experiment_dir, 'fig2-estimator.png'))
    
        # --------
        # figure 3
        # --------
        fig3 = plt.figure()
        fig3.suptitle("residual contributions of multiindices")
        ax = fig3.add_subplot(111)
        for mu, v in reserrmu.iteritems():
            ms = str(mu)
            ms = ms[ms.find('=') + 1:-1]
            ax.loglog(x[-len(v):], v, '-g<', label=ms)
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error")
        leg = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        ax.grid(True)
        fig3.savefig(os.path.join(options.experiment_dir, 'fig3-mi-residual.pdf'))
        fig3.savefig(os.path.join(options.experiment_dir, 'fig3-mi-residual.png'))
    
        # --------
        # figure 3b
        # --------
        fig3b = plt.figure()
        fig3b.suptitle("projection contributions")
        ax = fig3b.add_subplot(111)
        for mu, v in projerrmu.iteritems():
            ms = str(mu)
            ms = ms[ms.find('=') + 1:-1]
            try:
                ax.loglog(x[-len(v):], v, '-g<', label=ms)
            except:
                print "projection data for", mu, "is faulty... skipping..."
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error")
        leg = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        ax.grid(True)
        fig3b.savefig(os.path.join(options.experiment_dir, 'fig3b-mi-projection.pdf'))
        fig3b.savefig(os.path.join(options.experiment_dir, 'fig3b-mi-projection.png'))

        # --------
        # figure 4
        # --------
        fig4 = plt.figure()
        fig4.suptitle("projection $\zeta$")
        ax = fig4.add_subplot(111)
        ax.loglog(x[1:], proj_max_zeta[1:], '-g<', label='max zeta')
        ax.loglog(x[1:], proj_max_inactive_zeta[1:], '-b^', label='max inactive zeta')
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error")
        leg = plt.legend(loc='upper right')
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        ax.grid(True)
        fig4.savefig(os.path.join(options.experiment_dir, 'fig4-projection-zeta.pdf'))
        fig4.savefig(os.path.join(options.experiment_dir, 'fig4-projection-zeta.png'))
    
        # --------
        # figure 5
        # --------
        fig5 = plt.figure()
        fig5.suptitle("timings")
#        ax = fig5.add_subplot(111, aspect='equal')
        ax = fig5.add_subplot(111)
        ax.loglog(x, time_pcg, '-g<', label='pcg')
        ax.loglog(x, time_estimator, '-b^', label='estimator')
        ax.loglog(x, time_inactive_mi, '-c+', label='inactive_mi')
        ax.loglog(x, time_marking, '-ro', label='marking')
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("time in msec")
        leg = plt.legend(loc='lower right')
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        ax.grid(True)
        fig5.savefig(os.path.join(options.experiment_dir, 'fig5-timings.pdf'))
        fig5.savefig(os.path.join(options.experiment_dir, 'fig5-timings.png'))
        
        # --------
        # figure 6
        # --------
        fig6 = plt.figure()
        fig6.suptitle("projection error")
        ax = fig6.add_subplot(111)
        ax.loglog(x[1:], proj_part[1:], '-.m>', label='projection part')
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error")
        leg = plt.legend(loc='upper right')
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')    # the legend text fontsize
        ax.grid(True)
        fig6.savefig(os.path.join(options.experiment_dir, 'fig6-projection-error.pdf'))
        fig6.savefig(os.path.join(options.experiment_dir, 'fig6-projection-error.png'))
       
        if options.showFigures:
            plt.show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        print "skipped plotting since matplotlib is not available..."


# ==================
# D Generate Meshes
# ==================
if options.withMesh:
    from matplotlib import collections
    print "generating meshes for iteration", options.iteration_level
    w = w_history[options.iteration_level]
    for mu in w.active_indices():
        print "\t", mu
        itnr = options.iteration_level if options.iteration_level > 0 else len(w_history) + options.iteration_level
        mustr = str(mu).replace(' ', '')
        mustr = mustr[mustr.find("[") + 1: mustr.find("]")]
        fig1 = plt.figure()
        fig1.suptitle("mesh [%s] (iteration %i)" % (mustr, itnr))
        ax = fig1.add_subplot(111, aspect='equal')
        plt.axis('off')
        mesh = w[mu].mesh
        verts = mesh.coordinates()
        cells = mesh.cells()
        xlist, ylist = [], []
        for c in cells:
            for i in c:
                xlist.append(verts[i][0])
                ylist.append(verts[i][1])
            xlist.append(None)
            ylist.append(None)
        plt.fill(xlist, ylist, facecolor='none', alpha=1, edgecolor='b')
        fig1.savefig(os.path.join(options.experiment_dir, 'mesh%i-%s.pdf' % (itnr, mustr)))
        fig1.savefig(os.path.join(options.experiment_dir, 'mesh%i-%s.png' % (itnr, mustr)))


# ==================
# E Generate MI DATA
# ==================
print "generating multiindex data for iteration", options.iteration_level
w = w_history[options.iteration_level]
print w.dim


# ==========================
# F Generate SAMPLE SOLUTION
# ==========================
#    # plot sample solution
#    if opts.plotSolution:
#        w = w_history[-1]
#        # get random field sample and evaluate solution (direct and parametric)
#        RV_samples = coeff_field.sample_rvs()
#        ref_maxm = w_history[-1].max_order
#        sub_spaces = w[Multiindex()].basis.num_sub_spaces
#        degree = w[Multiindex()].basis.degree
#        maxh = min(w[Multiindex()].basis.minh / 4, CONF_max_h)
#        maxh = w[Multiindex()].basis.minh
#        projection_basis = get_projection_basis(mesh0, maxh=maxh, degree=degree, sub_spaces=sub_spaces)
#        sample_sol_param = compute_parametric_sample_solution(RV_samples, coeff_field, w, projection_basis)
#        sample_sol_direct = compute_direct_sample_solution(pde, RV_samples, coeff_field, A, ref_maxm, projection_basis)
#        sol_variance = compute_solution_variance(coeff_field, w, projection_basis)
#    
#        # plot
#        print sub_spaces
#        if sub_spaces == 0:
#            viz_p = plot(sample_sol_param._fefunc, title="parametric solution")
#            viz_d = plot(sample_sol_direct._fefunc, title="direct solution")
#            if ref_maxm > 0:
#                viz_v = plot(sol_variance._fefunc, title="solution variance")
#    
#            # debug---
#            if not True:        
#                for mu in w.active_indices():
#                    for i, wi in enumerate(w_history):
#                        if i == len(w_history) - 1 or True:
#                            plot(wi[mu]._fefunc, title="parametric solution " + str(mu) + " iteration " + str(i))
#    #                        plot(wi[mu]._fefunc.function_space().mesh(), title="parametric solution " + str(mu) + " iteration " + str(i), axes=True)
#                    interactive()
#            # ---debug
#            
##            for mu in w.active_indices():
##                plot(w[mu]._fefunc, title="parametric solution " + str(mu))
#        else:
#            mesh_param = sample_sol_param._fefunc.function_space().mesh()
#            mesh_direct = sample_sol_direct._fefunc.function_space().mesh()
#            wireframe = True
#            viz_p = plot(sample_sol_param._fefunc, title="parametric solution", mode="displacement", mesh=mesh_param, wireframe=wireframe)#, rescale=False)
#            viz_d = plot(sample_sol_direct._fefunc, title="direct solution", mode="displacement", mesh=mesh_direct, wireframe=wireframe)#, rescale=False)
#            
##            for mu in w.active_indices():
##                viz_p = plot(w[mu]._fefunc, title="parametric solution: " + str(mu), mode="displacement", mesh=mesh_param, wireframe=wireframe)
#        interactive()
