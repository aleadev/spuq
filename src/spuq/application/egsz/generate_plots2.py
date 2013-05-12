from __future__ import division

from spuq.application.egsz.multi_vector import supp

import optparse
import numpy as np
import os
from math import sqrt
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

# suppress UFL/FFC warnings
fenics_logger = logging.getLogger("FFC")
fenics_logger.setLevel(logging.WARNING)
fenics_logger = logging.getLogger("UFL")
fenics_logger.setLevel(logging.WARNING)

# ==================
# A Parse Arguments
# ==================
usage = "%prog [options] experiment_directory"
optparser = optparse.OptionParser(usage=usage)

optparser.add_option('--singleP', '--single-P',
                     type='int', default=0, dest='singleP',
                     help='only plot for single degree p or plot for all available if none is specified')

optparser.add_option('--withFigures', '--with-figures',
                     action='store_true', default=False, dest='withFigures',
                     help='export several figures')

optparser.add_option('--showFigures', '--show-figures',
                     action='store_true', default=False, dest='showFigures',
                     help='show several figures')

optparser.add_option('--withMeshes', '--with-meshes',
                      action='store_true', default=False, dest='withMesh',
                      help='export meshes')

optparser.add_option('--meshDofs', '--mesh-dofs',
                     type='int', default=0, dest='meshDofs',
                     help='export meshes that are just larger than given mesh dofs (if possible)')
 
# optparser.add_option('--withSolution', '--with-solution',
#                      action='store_true', default=False, dest='withSolution',
#                      help='export sample parametric and direct solution')

optparser.add_option('--withMI', '--with-mi',
                     action='store_true', default=False, dest='withMI',
                     help='export multiindex statistics')

optparser.add_option('--noTitles', '--no-titles',
                     action='store_false', default=True, dest='withTitles',
                     help='set titles on graphs')

options, args = optparser.parse_args()
if len(args) < 1:
    optparser.error('No experiment directory specified (use -h/--help for help)')
elif len(args) > 1:
    optparser.error('Too many arguments (use -h/--help for help)')
else:
    options.experiment_dir = args[0]
#     if len(args) > 1:
#         options.iteration_level = int(args[1])
#     else:
#         options.iteration_level = -1

# ==================
# B Import Solutions
# ==================
from glob import glob
import pickle

# load simulation data
LOAD_STATS_FN = os.path.join(options.experiment_dir, 'SIM2-STATS-P*.pkl')
SIM_STATS = {}
for fname in glob(LOAD_STATS_FN):
    P = int(fname[fname.find("-P") + 2:fname.find(".pkl")])
    print "loading P{0} statistics from {1}".format(P, fname)
    with open(fname, 'rb') as fin:
        sim_stats = pickle.load(fin)
    print "sim_stats has %s iterations" % len(sim_stats)

    fname_w = fname.replace("STATS", "SOLUTIONS").replace("SIM", "SFEM")
    print "loading P{0} statistics from {1}".format(P, fname_w)
    with open(fname_w, 'rb') as fin:
        w_history = pickle.load(fin)
    print "w_history has %s iterations" % len(w_history)
    
    # prepare data
    D = {}
    if len(sim_stats) > 0:
        print sim_stats[0].keys()
        for k in sim_stats[0].keys():
#            print "DATA", k
            if k not in ["CONF", "OPTS"]:
                D[k] = [s[k] for s in sim_stats]
        # evaluate additional data
        D["NUM-MI"] = [len(m) for m in D["MI"]]
        try:
            D["EFFICIENCY"] = [est / err for est, err in zip(D["ERROR-EST"], D["MC-ERROR-H1A"])]
            D["WITH-MC"] = True
        except:
            D["WITH-MC"] = False
            print "WARNING: No MC data found!"
        # ...from w_history
        D["NUM-Y"] = [len(supp(w.active_indices())) + 1 for w in w_history]
        D["MESH-CELLS"] = [w.basis.basis.mesh.num_cells() for w in w_history]
        D["MESH-HMIN"] = [w.basis.basis.mesh.hmin() for w in w_history]
        D["MESH-HMAX"] = [w.basis.basis.mesh.hmax() for w in w_history]
        D["MESH-HMINinv"] = [1 / h ** 2 for h in D["MESH-HMIN"]]
        D["MESH-HMAXinv"] = [1 / h ** 2 for h in D["MESH-HMAX"]]
        
        # meshes
        if options.withMesh:
            if options.meshDofs > 0:
                for i, dofs in enumerate(D["DOFS"]):
                    if dofs >= options.meshDofs or i == len(D["DOFS"]) - 1:
                        D["MESH"] = w_history[i].basis.basis.mesh
                        break
            else:
                D["MESH"] = w_history[-1].basis.basis.mesh
        w_history = None
        # store data for plotting
        SIM_STATS[P] = D
    else:
        print "SKIPPING P{0} data since it is empty!".format(P)


#================    ===============================
#character           description
#================    ===============================
#``'-'``             solid line style
#``'--'``            dashed line style
#``'-.'``            dash-dot line style
#``':'``             dotted line style
#``'.'``             point marker
#``','``             pixel marker
#``'o'``             circle marker
#``'v'``             triangle_down marker
#``'^'``             triangle_up marker
#``'<'``             triangle_left marker
#``'>'``             triangle_right marker
#``'1'``             tri_down marker
#``'2'``             tri_up marker
#``'3'``             tri_left marker
#``'4'``             tri_right marker
#``'s'``             square marker
#``'p'``             pentagon marker
#``'*'``             star marker
#``'h'``             hexagon1 marker
#``'H'``             hexagon2 marker
#``'+'``             plus marker
#``'x'``             x marker
#``'D'``             diamond marker
#``'d'``             thin_diamond marker
#``'|'``             vline marker
#``'_'``             hline marker
#================    ===============================
# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
        
# ==================
# C Generate Figures
# ==================
if options.withFigures:
       
    try:        
        # --------
        # figure 1
        # --------
        fig1 = plt.figure()
        if options.withTitles:
            fig1.suptitle("residual estimator overview")
        ax = fig1.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['active mi', 'estimator', 'residual', 'tail', 'MC H1A', 'MC L2', 'efficiency']
            else:
                LABELS = ["_nolegend_" for _ in range(7)]
            X = D["DOFS"]
            ax.loglog(X, D["NUM-MI"], '--y+', label=LABELS[0], linewidth=1.5)
            ax.loglog(X, D["ERROR-EST"], '-g*', label=LABELS[1], linewidth=1.5)
            ax.loglog(X, D["ERROR-RES"], '-.cx', label=LABELS[2], linewidth=1.5)
            ax.loglog(X, D["ERROR-TAIL"], '-.m>', label=LABELS[3], linewidth=1.5)
            if D["WITH-MC"]:
                ax.loglog(X, D["MC-ERROR-H1A"], '-b^', label=LABELS[4])
                ax.loglog(X, D["MC-ERROR-L2"], '-ro', label=LABELS[5])
                ax.loglog(X, D["EFFICIENCY"], '-.b>', label=LABELS[6], linewidth=1.5)
        plt.xlabel("overall degrees of freedom")
        plt.ylabel("energy error (number active multi-indices)")
        plt.axhline(y=0)
        plt.axvline(x=0)
        ax.grid(True)
        leg = plt.legend(ncol=1, loc='center right', bbox_to_anchor=(1.05, 0.2))
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize=12)    # the legend text fontsize
        fig1.savefig(os.path.join(options.experiment_dir, 'fig1-estimator-overview.pdf'))
        fig1.savefig(os.path.join(options.experiment_dir, 'fig1-estimator-overview.png'))

        # --------
        # figure 2
        # --------
        fig2 = plt.figure()
        if options.withTitles:
            fig2.suptitle("efficiency residual estimator")
        ax = fig2.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['estimator', 'residual', 'tail', 'MC H1A', 'MC L2', 'efficiency']
            else:
                LABELS = ["_nolegend_" for _ in range(6)]
            X = D["DOFS"]
            ax.loglog(X, D["ERROR-EST"], '-g*', label=LABELS[0], linewidth=1.5)
            ax.loglog(X, D["ERROR-RES"], '-.cx', label=LABELS[1], linewidth=1.5)
            ax.loglog(X, D["ERROR-TAIL"], '-.m>', label=LABELS[2], linewidth=1.5)
            if D["WITH-MC"]:
                ax.loglog(X, D["MC-ERROR-H1A"], '-b^', label=LABELS[3])
#                ax.loglog(X, D["MC-ERROR-L2"], '-ro', label=LABELS[4])
                ax.loglog(X, D["EFFICIENCY"], '-.b>', label=LABELS[5], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("energy error (efficiency)", fontsize=14)
            leg = plt.legend(loc='lower left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2.savefig(os.path.join(options.experiment_dir, 'fig2-estimator.pdf'))
            fig2.savefig(os.path.join(options.experiment_dir, 'fig2-estimator.png'))

        # ---------
        # figure 2a
        # ---------
        fig2a = plt.figure()
        if options.withTitles:
            fig2a.suptitle("residual estimator")
        ax = fig2a.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['estimator', 'MC H1A', 'efficiency']
            else:
                LABELS = ["_nolegend_" for _ in range(3)]
            X = D["DOFS"]
            ax.loglog(X, D["ERROR-EST"], '-g*', label=LABELS[0], linewidth=1.5)
            if D["WITH-MC"]:
                ax.loglog(X, D["MC-ERROR-H1A"], '-b^', label=LABELS[1], linewidth=1.5)
                ax.loglog(X, D["EFFICIENCY"], '-.b>', label=LABELS[2], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("energy error (efficiency)", fontsize=14)
            leg = plt.legend(loc='lower left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2a.savefig(os.path.join(options.experiment_dir, 'fig2a-estimator.pdf'))
            fig2a.savefig(os.path.join(options.experiment_dir, 'fig2a-estimator.png'))

        # ---------
        # figure 2b
        # ---------
        fig2b = plt.figure()
        if options.withTitles:
            fig2b.suptitle("residual estimator")
        ax = fig2b.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['estimator', 'MC H1A']
            else:
                LABELS = ["_nolegend_" for _ in range(2)]
            X = D["DOFS"]
            ax.loglog(X, D["ERROR-EST"], '-g*', label=LABELS[0], linewidth=1.5)
            if D["WITH-MC"]:
                ax.loglog(X, D["MC-ERROR-H1A"], '-b^', label=LABELS[1], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("energy error", fontsize=14)
            leg = plt.legend(loc='lower left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2b.savefig(os.path.join(options.experiment_dir, 'fig2b-estimator.pdf'))
            fig2b.savefig(os.path.join(options.experiment_dir, 'fig2b-estimator.png'))

        # ---------
        # figure 2c
        # ---------
        fig2c = plt.figure()
        if options.withTitles:
            fig2c.suptitle("residual estimator")
        ax = fig2c.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['estimator']
            else:
                LABELS = ["_nolegend_"]
            X = D["DOFS"]
            ax.loglog(X, D["ERROR-EST"], '-g*', label=LABELS[0], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("energy error", fontsize=14)
            leg = plt.legend(loc='lower left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2c.savefig(os.path.join(options.experiment_dir, 'fig2c-estimator.pdf'))
            fig2c.savefig(os.path.join(options.experiment_dir, 'fig2c-estimator.png'))

        # ---------
        # figure 2d
        # ---------
        fig2d = plt.figure()
        if options.withTitles:
            fig2d.suptitle("active multiindices")
        ax = fig2d.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['active mi']
            else:
                LABELS = ["_nolegend_"]
            X = D["DOFS"]
            ax.loglog(X, D["NUM-MI"], '--y+', label=LABELS[0], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("number active mi", fontsize=14)
            leg = plt.legend(loc='upper left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2d.savefig(os.path.join(options.experiment_dir, 'fig2d-mi.pdf'))
            fig2d.savefig(os.path.join(options.experiment_dir, 'fig2d-mi.png'))

        # ---------
        # figure 2e
        # ---------
        fig2e = plt.figure()
        if options.withTitles:
            fig2e.suptitle("active multiindices/stochastic dimensions")
        ax = fig2e.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['active mi', 'active dim']
            else:
                LABELS = ["_nolegend_", "_nolegend_"]
            X = D["DOFS"]
            ax.loglog(X, D["NUM-MI"], '--y+', label=LABELS[0], linewidth=1.5)
            ax.loglog(X, D["NUM-Y"], '-g>', label=LABELS[1], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("active mi/stochastic dim", fontsize=14)
            leg = plt.legend(loc='upper left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2e.savefig(os.path.join(options.experiment_dir, 'fig2e-mi.pdf'))
            fig2e.savefig(os.path.join(options.experiment_dir, 'fig2e-mi.png'))

        # ---------
        # figure 2f
        # ---------
        try:
            fig2f = plt.figure()
            if options.withTitles:
                fig2f.suptitle("active multiindices and C_{a,\delta}")
            ax = fig2f.add_subplot(111)
            for P, D in SIM_STATS.iteritems():
                if P == SIM_STATS.keys()[0]:
                    LABELS = ['active mi', 'C_{a,\delta}']
                else:
                    LABELS = ["_nolegend_", "_nolegend_"]
                X = D["DOFS"]
                ax.loglog(X, D["NUM-MI"], '--y+', label=LABELS[0], linewidth=1.5)
                ax.loglog(X, D["CADELTA"], '-g>', label=LABELS[1], linewidth=1.5)
                plt.xlabel("overall degrees of freedom", fontsize=14)
                plt.ylabel("active mi/C_{a,\delta}", fontsize=14)
                leg = plt.legend(loc='upper left')
                legtext = leg.get_texts()  # all the text.Text instance in the legend
                plt.setp(legtext, fontsize=12)    # the legend text fontsize
                ax.grid(True)
                fig2f.savefig(os.path.join(options.experiment_dir, 'fig2f-mi.pdf'))
                fig2f.savefig(os.path.join(options.experiment_dir, 'fig2f-mi.png'))
        except:
            pass

        # ---------
        # figure 2p
        # ---------
        LABELS = ['estimator', 'residual', 'tail', 'MC H1A']
        for P, D in SIM_STATS.iteritems():
            fig2p = plt.figure()
            if options.withTitles:
                fig2p.suptitle("residual estimator P%i" % P)
            ax = fig2p.add_subplot(111)
            X = D["DOFS"]
            ax.loglog(X, D["ERROR-EST"], '-g*', label=LABELS[0], linewidth=1.5)
            ax.loglog(X, D["ERROR-RES"], '-.cx', label=LABELS[1], linewidth=1.5)
            ax.loglog(X, D["ERROR-TAIL"], '-.m>', label=LABELS[2], linewidth=1.5)
            if D["WITH-MC"]:
                ax.loglog(X, D["MC-ERROR-H1A"], '-b^', label=LABELS[3], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("energy error", fontsize=14)
            leg = plt.legend(loc='upper right')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig2p.savefig(os.path.join(options.experiment_dir, 'fig2-estimator-P%i.pdf' % P))
            fig2p.savefig(os.path.join(options.experiment_dir, 'fig2-estimator-P%i.png' % P))

        # ---------
        # figure 3a
        # ---------
        fig3a = plt.figure()
        if options.withTitles:
            fig3a.suptitle("mesh sizes")
        ax = fig3a.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['cells', '1/min{h}^2', '1/max{h}^2']
            else:
                LABELS = ["_nolegend_", "_nolegend_", "_nolegend_"]
            X = D["DOFS"]
            ax.loglog(X, D["MESH-CELLS"], '--r+', label=LABELS[0], linewidth=1.5)
            ax.loglog(X, D["MESH-HMINinv"], '-y<', label=LABELS[1], linewidth=1.5)
            ax.loglog(X, D["MESH-HMAXinv"], '-g>', label=LABELS[2], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("cells and 1/h^2", fontsize=14)
            leg = plt.legend(loc='upper left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig3a.savefig(os.path.join(options.experiment_dir, 'fig3a-mesh.pdf'))
            fig3a.savefig(os.path.join(options.experiment_dir, 'fig3a-mesh.png'))

        # ---------
        # figure 3b
        # ---------
        fig3b = plt.figure()
        if options.withTitles:
            fig3b.suptitle("mesh sizes")
        ax = fig3b.add_subplot(111)
        for P, D in SIM_STATS.iteritems():
            if P == SIM_STATS.keys()[0]:
                LABELS = ['cells']
            else:
                LABELS = ["_nolegend_"]
            X = D["DOFS"]
            ax.loglog(X, D["MESH-CELLS"], '--r+', label=LABELS[0], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("cells", fontsize=14)
            leg = plt.legend(loc='upper left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig3b.savefig(os.path.join(options.experiment_dir, 'fig3b-mesh.pdf'))
            fig3b.savefig(os.path.join(options.experiment_dir, 'fig3b-mesh.png'))

        # ---------
        # figure 3p
        # ---------
        LABELS = ['cells', '1/min{h}^2', '1/max{h}^2']
        for P, D in SIM_STATS.iteritems():
            fig3p = plt.figure()
            if options.withTitles:
                fig3p.suptitle("mesh sizes P%i" % P)
            ax = fig3p.add_subplot(111)
            X = D["DOFS"]
            ax.loglog(X, D["MESH-CELLS"], '--r+', label=LABELS[0], linewidth=1.5)
            ax.loglog(X, D["MESH-HMINinv"], '-y<', label=LABELS[1], linewidth=1.5)
            ax.loglog(X, D["MESH-HMAXinv"], '-g>', label=LABELS[2], linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("cells and 1/h^2", fontsize=14)
            leg = plt.legend(loc='upper left')
            legtext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(legtext, fontsize=12)    # the legend text fontsize
            ax.grid(True)
            fig3p.savefig(os.path.join(options.experiment_dir, 'fig3-mesh-P%i.pdf' % P))
            fig3p.savefig(os.path.join(options.experiment_dir, 'fig3-mesh-P%i.png' % P))


        if False:                
        
            # --------
            # figure 3d
            # --------
            fig3d = plt.figure()
            fig3d.suptitle("multi-index activation and refinement (iteration %i)" % len(w_history))
            ax = fig3d.add_subplot(111)
            w = w_history[-1]
            mudim = sorted([(mu, w[mu].dim) for mu in w.active_indices()], key=itemgetter(1), reverse=True)
            
            for i, muv in enumerate(mudim):
                mu, v = muv
                if i < max_plot_mu:
                    ms = str(mu)
                    ms = ms[ms.find('=') + 1:-1]
                    
                    d, idx = [], len(w_history) - 1
                    while idx >= 0:
                        try:
                            d.append(w_history[idx][mu].dim)
                            idx -= 1
                        except:
                            break
                    d = d[::-1]
    
                    itoff = len(w_history) - len(d)
                    if len(d) > 0:
                        ax.plot(range(itoff, itoff + len(d)), d, '-g<', label=ms)
            plt.xlabel("iteration")
            plt.ylabel("degrees of freedom")
            leg = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
            ltext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(ltext, fontsize='small')    # the legend text fontsize
            ax.grid(True)
            fig3d.savefig(os.path.join(options.experiment_dir, 'fig3d-mi-activation-final.pdf'))
            fig3d.savefig(os.path.join(options.experiment_dir, 'fig3d-mi-activation-final.png'))
        
            # --------
            # figure 5
            # --------
            fig5 = plt.figure()
            if options.withTitles:
                fig5.suptitle("timings")
    #        ax = fig5.add_subplot(111, aspect='equal')
            ax = fig5.add_subplot(111)
            ax.loglog(x, time_pcg, '--g<', label='pcg')
            ax.loglog(x, time_estimator, '-b^', label='estimator overall')
            ax.loglog(x, time_projection, '--kd', label='projection')
            ax.loglog(x, time_residual, '--ms', label='residual')
            ax.loglog(x, time_inactive_mi, '--c+', label='inactive_mi')
            ax.loglog(x, time_marking, '-ro', label='marking')
            plt.xlabel("overall degrees of freedom")
            plt.ylabel("time in seconds")
            leg = plt.legend(loc='lower right')
            ltext = leg.get_texts()  # all the text.Text instance in the legend
            plt.setp(ltext, fontsize='small')    # the legend text fontsize
            ax.grid(True)
            fig5.savefig(os.path.join(options.experiment_dir, 'fig5-timings.pdf'))
            fig5.savefig(os.path.join(options.experiment_dir, 'fig5-timings.png'))
                    
            # --------
            # figure 8
            # --------
            w = w_history[options.iteration_level]
            N = len(w.active_indices())
            fig8 = plt.figure(figsize=(8, N / 2 + 1))
            plt.plot([0, 0], 'r')
            plt.grid(False)
            plt.axis([0, 3, 0, (N + 1) / 2])
            fig8.suptitle("active multi-indices and dofs (%s MI for iteration %s, dim = %s)" % (str(len(w.active_indices())), str(itnr), str(sum(w.dim.values()))))
            ax = fig8.add_subplot(111)
            for i, mu in enumerate(w.active_indices()):
                ms = str(mu)
                ms = ms[ms.find('=') + 1:-1]
                ms = '{:<40}'.format(ms) + str(w[mu].dim)
                plt.text(0.5, (1 + i) / 2, ms, fontsize=12)
            fig8.savefig(os.path.join(options.experiment_dir, 'fig8-active-mi.pdf'))
            fig8.savefig(os.path.join(options.experiment_dir, 'fig8-active-mi.png'))
    
            # ---------
            # figure 10
            # ---------
            if marking_res is not None:
                fig10 = plt.figure()
                if options.withTitles:
                    fig10.suptitle("marking")
                ax = fig10.add_subplot(111)
                ax.loglog(x[:len(marking_res)], marking_res, '-.m>', label='residual')
                ax.loglog(x[:len(marking_proj)], marking_proj, '-g^', label='projection')
                plt.xlabel("overall degrees of freedom")
                plt.ylabel("number marked elements")
                leg = plt.legend(loc='upper left')
                ltext = leg.get_texts()  # all the text.Text instance in the legend
                plt.setp(ltext, fontsize='small')    # the legend text fontsize
                ax.grid(True)
                fig10.savefig(os.path.join(options.experiment_dir, 'fig10-marking.pdf'))
                fig10.savefig(os.path.join(options.experiment_dir, 'fig10-marking.png'))
    
            # ---------
            # figure 11
            # ---------
            fig11 = plt.figure()
            if options.withTitles:
                fig11.suptitle("degrees of freedom")
            ax = fig11.add_subplot(111)
            ax.loglog(dofs, dofs, '-.m>', label='overall dofs')
            ax.loglog(dofs, mu_max_dim, '-k', label='max dim $w_\mu$', linewidth=1.5)
            ax.loglog(dofs, num_mi, '--y+', label='active mi', linewidth=1.5)
            plt.xlabel("overall degrees of freedom", fontsize=14)
            plt.ylabel("degrees of freedom (active multi-indices)", fontsize=14)
            leg = plt.legend(loc='upper left')
    #        ltext = leg.get_texts()  # all the text.Text instance in the legend
    #        plt.setp(ltext, fontsize='small')    # the legend text fontsize
            ax.grid(True)
            fig11.savefig(os.path.join(options.experiment_dir, 'fig11-dofs.pdf'))
            fig11.savefig(os.path.join(options.experiment_dir, 'fig11-dofs.png'))
    
            # ===============
            # save single pdf
            # ===============
            pp = PdfPages(os.path.join(options.experiment_dir, 'all-figures.pdf'))
            pp.savefig(fig1)
            pp.savefig(fig2)
            pp.savefig(fig3)
            if has_projection:
                pp.savefig(fig3b)
            if options.iteration_level > 0:
                try:
                    pp.savefig(fig3c)
                except:
                    pass
            pp.savefig(fig3d)
            if has_projection:
                pp.savefig(fig4)
            pp.savefig(fig5)
            if has_projection:
                pp.savefig(fig6)
                pp.savefig(fig7)
            pp.savefig(fig8)
            pp.savefig(fig9)
            if marking_res is not None:
                pp.savefig(fig10)
            pp.savefig(fig11)
            pp.close()
           
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
    for P, D in SIM_STATS.iteritems():
        figM = plt.figure()
        figM.suptitle("mesh P%i" % P)
        ax = figM.add_subplot(111, aspect='equal')
#        plt.axis('off')
        mesh = D["MESH"]
        verts = mesh.coordinates()
        cells = mesh.cells()
        
        plot_method = 1
        if plot_method == 0:    # NOTE: this was proposed as a faster method - which in fact does not work properly!
            xlist, ylist = [], []
            for c in cells:
                for i in c:
                    xlist.append(verts[i][0])
                    ylist.append(verts[i][1])
                xlist.append(None)
                ylist.append(None)
            plt.fill(xlist, ylist, facecolor='none', alpha=1, edgecolor='b')
        elif plot_method == 1:
            for c in cells:
                xlist, ylist = [], []
                for i in c:
                    xlist.append(verts[i][0])
                    ylist.append(verts[i][1])
                plt.fill(xlist, ylist, facecolor='none', alpha=1, edgecolor='b')

        figM.savefig(os.path.join(options.experiment_dir, 'mesh-P%i.pdf' % P))
        figM.savefig(os.path.join(options.experiment_dir, 'mesh-P%i.png' % P))


# ==================
# E Generate MI DATA
# ==================
if options.withMI:
    print "generating multi-index data for last iterations..."
    for P, D in SIM_STATS.iteritems():
        itnr = len(D["MI"]) - 1
        print "# multi-indices and dimensions for '{0}' at iteration {1}".format(options.experiment_dir, itnr)
        with file(os.path.join(options.experiment_dir, 'MI-P{0}-{1}.txt'.format(P, itnr)), 'w') as f:
            for mu in D["MI"][-1]:
                ms = str(mu)
                ms = ms[ms.find('=') + 1:-1]
                print D["DOFS"]
                print D["DIM"]
                print D["NUM-MI"]
                mis = '{0:2s} {1:3d}'.format(ms, D["DIM"][-1][mu])
                print mis
                f.write(mis + "\n")
            dofs = D["DOFS"][-1]
            nummi = D["NUM-MI"][-1]
            itnr = len(D["DOFS"])
            f.write("overall dofs = %i and %i active multi-indices for iteration %i\n" % (dofs, nummi, itnr))
            print "overall dofs =", dofs
