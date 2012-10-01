import optparse
import ConfigParser

# options
# =======

optparser = optparse.OptionParser()

optparser.add_option('-f', '--conffile', dest='conffile', default ='test.conf')

optparser.add_option('--runSFEM', action='store_false', default=False,
                     dest='runSFEM', help='')
optparser.add_option('--runMC', action='store_false', default=False,
                     dest='runSFEM', help='')

optparser.add_option('--plotSolution', action='store_false', default=False,
                     dest='plotSolution', help='')
optparser.add_option('--plotEstimator', action='store_false', default=False,
                     dest='plotEstimator', help='')
optparser.add_option('--plotMesh', action='store_false', default=False,
                     dest='plotMesh', help='')

optparser.add_option('--clear', type='choice', choices=['none', 'all', 'SFEM', 'MC'], dest='clear', default ='none')

options, args = optparser.parse_args()
print "program options", options

# config
# ======

try:
    confparser = ConfigParser.SafeConfigParser()
    if not confparser.read(options.conffile):
        raise ConfigParser.ParsingError("file not found")
except ConfigParser.ParsingError, err:
    print "Could not parse:", err

for sec in confparser.sections():
    print "section", sec
    for opt in confparser.options(sec):
        print "\t", opt, "=", confparser.get(sec, opt)
    print ""
