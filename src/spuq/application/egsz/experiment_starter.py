import optparse
import ConfigParser
from run_SFEM import run_SFEM
from run_MC import run_MC


class ExperimentStarter(object):
    def __init__(self):
        self.opts = self._parse_options()
        self.conf = self._parse_config(self.opts)
        
    def _parse_options(self):
        optparser = optparse.OptionParser()
        
        optparser.add_option('-f', '--conffile', dest='conffile', default='test.conf')
        
        optparser.add_option('--runSFEM', action='store_true', default=False,
                             dest='runSFEM', help='')
        optparser.add_option('--runMC', action='store_true', default=False,
                             dest='runMC', help='')
        optparser.add_option('--noSaveData', action='store_false', default=True,
                             dest='saveData', help='')
        
        optparser.add_option('--plotSolution', action='store_true', default=False,
                             dest='plotSolution', help='')
        optparser.add_option('--plotEstimator', action='store_true', default=False,
                             dest='plotEstimator', help='')
        optparser.add_option('--plotError', action='store_true', default=False,
                             dest='plotError', help='')
        optparser.add_option('--plotMesh', action='store_true', default=False,
                             dest='plotMesh', help='')
        
        optparser.add_option('--clear', type='choice', choices=['none', 'all', 'SFEM', 'MC'], dest='clear', default='none')
        
        optparser.add_option('--debug', action='store_true', default=False,
                             dest='debug', help='')
        
        options, args = optparser.parse_args()
        from os.path import dirname
        basedir = dirname(options.conffile)
    
        if options.debug:
            print "program options", options
            print "basedir", basedir 
        
        options.basedir = basedir
        return options
        
    def _parse_config(self, opts):
        try:
            confparser = ConfigParser.SafeConfigParser()
            if not confparser.read(opts.conffile):
                raise ConfigParser.ParsingError("file not found")
        except ConfigParser.ParsingError, err:
            print "Could not parse:", err

        # extract options
        getter = ("get", "getint", "getfloat", "getboolean")
        option_defs = (("SFEM",
                        {"problem_type":1,
                            "domain":0,
                            "boundary_type":1,
                            "assembly_type":0,
                            "FEM_degree":1,
                            "decay_exp":1,
                            "coeff_type":1,
                            "coeff_scale":2,
                            "freq_scale":2,
                            "freq_skip":1,
                            "gamma":2}),
                   ("SFEM adaptive algorithm",
                        {"iterations":1,
                            "uniform_refinement":3,
                            "initial_Lambda":1,
                            "refine_residual":3,
                            "refine_projection":3,
                            "refine_Lambda":3,
                            "cQ":2,
                            "ceta":2,
                            "theta_eta":2,
                            "theta_zeta":2,
                            "min_zeta":2,
                            "maxh":2,
                            "newmi_add_maxm":1,
                            "theta_delta":2,
                            "max_Lambda_frac":2,
                            "quadrature_degree":1,
                            "projection_degree_increase":1,
                            "refine_projection_mesh":1,
                            "pcg_eps":2,
                            "pcg_maxiter":1,
                            "error_eps":2}),
                   ("LOGGING",
                        {"level":0}),
                   ("MC",
                        {"runs":1,
                            "N":1,
                            "max_h":2})
                   )
        conf = {}
        for sec, optsdict in option_defs:
            conf[sec] = {}
            for key, keytype in optsdict.iteritems():
                exec "conf['" + sec + "']['" + key + "'] = confparser." + getter[keytype] + "('" + sec + "','" + key + "')"

        if opts.debug:
            for sec in confparser.sections():
                print "section", sec
                for opt in confparser.options(sec):
                    print "\t", opt, "=", confparser.get(sec, opt)
                print ""
            print conf

        return conf

    def start(self):
        # check if data should be cleared
        if self.opts.clear == 'SFEM' or self.opts.clear == 'all':
            print "clearing SFEM data"
            # TODO
        if self.opts.clear == 'MC' or self.opts.clear == 'all':
            print "clearing MC data"
            # TODO

        # start SFEM
        if self.opts.runSFEM:
            print "="*60
            print "starting SFEM"
            print "="*60
            run_SFEM(self.opts, self.conf)
        
        # start MC
        if self.opts.runMC:
            print "="*60
            print "starting MC"
            print "="*60
            run_MC(self.opts, self.conf)

# -----------------------------------------------------------------------------

# get configuration and start experiments
starter = ExperimentStarter()
starter.start()
