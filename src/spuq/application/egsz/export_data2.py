from __future__ import division

from spuq.application.egsz.multi_vector import supp

import optparse
import numpy as np
import os
from math import sqrt
from collections import defaultdict
from operator import itemgetter

# ==================
# A Parse Arguments
# ==================
usage = "%prog [options] experiment_directory"
optparser = optparse.OptionParser(usage=usage)

optparser.add_option('--singleP', '--single-p',
                     action='store_true', default=False, dest='singleP',
                     help='EGSZ1')

options, args = optparser.parse_args()
if len(args) < 1:
    optparser.error('No experiment directory specified (use -h/--help for help)')
elif len(args) > 1:
    optparser.error('Too many arguments (use -h/--help for help)')
else:
    options.experiment_dir = args[0]

# ==================
# B Import Solutions
# ==================
from glob import glob
import pickle

# load simulation data
FNAME = 'SIM2-STATS-P*.pkl' if not options.singleP else 'SIM-STATS.pkl'
LOAD_STATS_FN = os.path.join(options.experiment_dir, FNAME)
print "trying to load", LOAD_STATS_FN
SIM_STATS = {}
for fname in glob(LOAD_STATS_FN):
    if options.singleP:
        P = 1
    else:
        P = int(fname[fname.find("-P") + 2:fname.find(".pkl")])
    print "loading P{0} statistics from {1}".format(P, fname)
    with open(fname, 'rb') as fin:
        sim_stats = pickle.load(fin)
    print "sim_stats has %s iterations" % len(sim_stats)
    
    # prepare data
    D = {}
    if len(sim_stats) > 0:
        print sim_stats[0].keys()
        for k in sim_stats[0].keys():
#            print "DATA", k
            if k not in ["CONF", "OPTS", "PROJ-INACTIVE-ZETA"]:
                D[k] = [s[k] for s in sim_stats]
        # evaluate additional data
        D["NUM-MI"] = [len(m) for m in D["MI"]]
        try:
            if options.singleP:
                D["DIM-Y"] = [len(supp([i[0] for i in ami])) + 1 for ami in D["MI"]]
                # WARNING: EGSZ1 writes out the squared estimator!!!
                D["EST"] = [sqrt(est) for est in D["EST"]]
                D["EFFICIENCY"] = [est / err for est, err in zip(D["EST"], D["MC-H1ERR"])]
            else:
                D["DIM-Y"] = [len(supp(ami)) + 1 for ami in D["MI"]]
                D["EFFICIENCY"] = [est / err for est, err in zip(D["ERROR-EST"], D["MC-ERROR-H1A"])]
            D["WITH-MC"] = True
        except:
            D["WITH-MC"] = False
            print "WARNING: No MC data found!"
        # store data for plotting
        SIM_STATS[P] = D
    else:
        print "SKIPPING P{0} data since it is empty!".format(P)

    # export all data
    with file(os.path.join(options.experiment_dir, 'SIMDATA-export.txt'), 'w') as f:
        for P, D in SIM_STATS.iteritems():
            f.write("==== P{0} ====\n".format(P))
            f.write("DOFS\n" + str(D["DOFS"]))
            if options.singleP:
                f.write("\nERROR-EST\n" + str(D["EST"]))
                f.write("\nERROR-RES\n" + str(D["RES-PART"]))
                f.write("\nERROR-PROJ\n" + str(D["PROJ-PART"]))
                f.write("\nERROR-PCG\n" + str(D["PCG-PART"]))
                f.write("\nNUM-MI\n" + str(D["NUM-MI"]))
                f.write("\nDIM-Y\n" + str(D["DIM-Y"]))
                if D["WITH-MC"]:
                    f.write("\nMC-ERROR-H1A\n" + str(D["MC-H1ERR"]))
#                    f.write("\nMC-ERROR-L2\n" + str(D["MC-L2ERR"]))
#                    f.write("\nEFFICIENCY\n" + str(D["EFFICIENCY"]))
            else:
                f.write("\nERROR-EST\n" + str(D["ERROR-EST"]))
                f.write("\nERROR-RES\n" + str(D["ERROR-RES"]))
                f.write("\nERROR-TAIL\n" + str(D["ERROR-TAIL"]))
                f.write("\nNUM-MI\n" + str(D["NUM-MI"]))
                f.write("\nDIM-Y\n" + str(D["DIM-Y"]))
                if D["WITH-MC"]:
                    f.write("\nMC-ERROR-H1A\n" + str(D["MC-ERROR-H1A"]))
#                    f.write("\nMC-ERROR-L2\n" + str(D["MC-ERROR-L2"]))
#                    f.write("\nEFFICIENCY\n" + str(D["EFFICIENCY"]))
                f.write("\n\n")

    # export data for TeX plotting
    for P, D in SIM_STATS.iteritems():
        print "==== exporting P{0} ====".format(P)
        with file(os.path.join(options.experiment_dir, 'SIMDATA-P%i.dat' % P), 'w') as f:
            mcstr = "\terror\tefficiency" if D["WITH-MC"] else ""
            f.write("dofs\terrest%s\tmi\tydim\tcells\tsumcells\terrres\terrtail" % mcstr)
            for i, d in enumerate(D["DOFS"]):
                f.write("\n" + str(d))
                if options.singleP:
#                    f.write("\nERROR-EST\n" + str(D["EST"]))
#                    f.write("\nNUM-MI\n" + str(D["NUM-MI"]))
                    f.write("\t" + str(D["EST"][i]))
                    if D["WITH-MC"]:
                        f.write("\t" + str(D["MC-H1ERR"][i]))
                        f.write("\t" + str(D["EFFICIENCY"][i]))
                    f.write("\t" + str(D["NUM-MI"][i]))
                    f.write("\t" + str(D["DIM-Y"][i]))
                    f.write("\t" + str(int(D["CELLS"][i]/D["NUM-MI"][i])))
                    f.write("\t" + str(D["CELLS"][i]))
                    f.write("\t" + str(D["ERROR-RES"][i]))
                    f.write("\t" + str(D["ERROR-TAIL"][i]))
                else:
                    f.write("\t" + str(D["ERROR-EST"][i]))
                    if D["WITH-MC"]:
                        f.write("\t" + str(D["MC-ERROR-H1A"][i]))
                        f.write("\t" + str(D["EFFICIENCY"][i]))
                    f.write("\t" + str(D["NUM-MI"][i]))
                    f.write("\t" + str(D["DIM-Y"][i]))
                    f.write("\t" + str(int(D["CELLS"][i]/D["NUM-MI"][i]))
                    f.write("\t" + str(D["CELLS"][i]))
                    f.write("\t" + str(D["ERROR-RES"][i]))
                    f.write("\t" + str(D["ERROR-TAIL"][i]))
            f.write("\n")
