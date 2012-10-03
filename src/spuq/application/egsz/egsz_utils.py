import logging
from math import sqrt
from collections import defaultdict

def setup_logging(level):
    # log level and format configuration
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=__file__[:-2] + 'log', level=level,
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
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)
    logging.getLogger("spuq").addHandler(ch)
    return logger


def stats_plotter(sim_stats, plot_def=None, plot_type='loglog', title='stats', logger=None, legend_loc='upper right', save_image='', code=None, show_plot=True):
    try:
        from matplotlib.pyplot import figure, show, legend
        # restructure data
        D = {}
        # DOFS, L2, H1, EST, RES, PROJ, RES - mu, PROJ - mu, MI
        for key in sim_stats[0].keys():
            D[key] = [s[key] for s in sim_stats]

        # special treatment
        try:
            D["EST"] = map(lambda x:sqrt(x), D["EST"])
        except:
            pass
        try:
            D["EFFEST"] = [est / err for est, err in zip(D["EST"], D["MC-H1ERR"])]
        except:
            pass
        try:
            D["NUM_MI"] = [len(m) for m in D["MI"]]
        except:
            pass
        reserrmu = defaultdict(list)
        for rem in D["RES-mu"]:
            for mu, v in rem:
                reserrmu[mu].append(v)
        D["RESERR-mu"] = reserrmu
        
        print D.keys()
        
        # plot
        fig = figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        if code is None:
            x = D["DOFS"]
            for d in plot_def:
                try:
                    shift = d[3]
                except:
                    shift = 0
                key = d[0]
                marker = d[1]
                label = d[2]
                print 'ax.' + plottype + "(x[" + shift + ":]" + ",D['" + key + "'][" + shift + ":],'" + marker + "'" + ",label='" + label + "')"
                eval('ax.' + plottype + "(x[" + shift + ":]" + ",D['" + key + "'][" + shift + ":],'" + marker + "'" + ",label='" + label + "')", globals, locals)
        else:
            eval(code, globals(), locals())
        legend(log=legend_loc)
        
        if save_image != '':
            fig.savefig(save_image + '.png')
            fig.savefig(save_image + '.eps')

        if show_plot:        
            show()  # this invalidates the figure instances...
    except:
        import traceback
        print traceback.format_exc()
        logger.info("skipped plotting since matplotlib is not available...")
