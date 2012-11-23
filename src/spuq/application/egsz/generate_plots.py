from __future__ import division

import optparse
import numpy as np
import os
from math import sqrt
from collections import defaultdict

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
        from matplotlib.pyplot import figure, show, legend, xlabel, ylabel
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
        print "errest", errest
            
        # --------
        # figure 1
        # --------
        fig1 = figure()
        fig1.suptitle("residual estimator")
        ax = fig1.add_subplot(111)
        ax.loglog(x, num_mi, '--y+', label='active mi', linewidth=1.5)
        ax.loglog(x, errest, '-g<', label='estimator', linewidth=1.5)
        ax.loglog(x, res_part, '-.cx', label='residual', linewidth=1.5)
        ax.loglog(x[1:], proj_part[1:], '-.m>', label='projection', linewidth=1.5)
        ax.loglog(x, pcg_part, '-.b>', label='pcg', linewidth=1.5)
        xlabel("overall degrees of freedom")
        ylabel("energy error (number active multiindices)")
            
#        t = np.arange(1, num_mi[-1] * 2, 1)
#        ax2 = ax.twinx()
#        ax.plot(x, num_mi, '--y+', label='active mi', linewidth=1.5)
#        ax2.set_ylabel('number active multiindices')
##        for tl in ax2.get_yticklabels():
##            tl.set_color('b')

        legend(loc='upper right')
        fig1.savefig(os.path.join(options.experiment_dir, 'fig1-estimator.pdf'))
        fig1.savefig(os.path.join(options.experiment_dir, 'fig1-estimator.png'))
    
        # --------
        # figure 2
        # --------
        fig1 = figure()
        fig1.suptitle("efficiency residual estimator")
        ax = fig1.add_subplot(111)
        ax.loglog(x, errest, '-g<', label='error estimator')
        # TODO: MC ERROR and EFFICIENCY
        xlabel("overall degrees of freedom")
        ylabel("energy error (efficiency)")
        legend(loc='upper right')
    
        # --------
        # figure 3
        # --------
        fig3 = figure()
        fig3.suptitle("residual contributions of multiindices")
        ax = fig3.add_subplot(111)
        for mu, v in reserrmu.iteritems():
            ms = str(mu)
            ms = ms[ms.find('=') + 1:-1]
            ax.loglog(x[-len(v):], v, '-g<', label=ms)
        xlabel("overall degrees of freedom")
        ylabel("energy error")
        legend(loc='upper right')
    
        # --------
        # figure 4
        # --------
        fig4 = figure()
        fig4.suptitle("projection zetas")
        ax = fig4.add_subplot(111)
        ax.loglog(x[1:], proj_max_zeta[1:], '-g<', label='max zeta')
        ax.loglog(x[1:], proj_max_inactive_zeta[1:], '-b^', label='max inactive zeta')
        xlabel("overall degrees of freedom")
        ylabel("energy error")
        legend(loc='upper right')
    
        # --------
        # figure 5
        # --------
        fig5 = figure()
        fig5.suptitle("timings")
        ax = fig5.add_subplot(111, aspect='equal')
        ax.loglog(x, time_pcg, '-g<', label='pcg')
        ax.loglog(x, time_estimator, '-b^', label='estimator')
        ax.loglog(x, time_inactive_mi, '-c+', label='inactive_mi')
        ax.loglog(x, time_marking, '-ro', label='marking')
        xlabel("overall degrees of freedom")
        ylabel("time in msec")
        legend(loc='upper right')
                 
        # --------
        # figure 6
        # --------
        fig6 = figure()
        fig6.suptitle("projection error")
        ax = fig6.add_subplot(111)
        ax.loglog(x[1:], proj_part[1:], '-.m>', label='projection part')
        xlabel("overall degrees of freedom")
        ylabel("energy error")
        legend(loc='upper right')
       
        if options.showFigures:
            show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        print "skipped plotting since matplotlib is not available..."
