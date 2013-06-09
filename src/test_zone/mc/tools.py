import os
from dolfin.cpp.io import File
import scitools.easyviz as p
import numpy as np
import scipy.stats as stats
from matplotlib.pyplot import xlabel, ylabel
import time as t

class tools:
    logLevel = 0
    folder = ""

    @classmethod
    def status(cls, str, lvl = 3):
        if cls.logLevel >= lvl:
            print str

    @classmethod
    def setLogLevel(cls, lvl):
        cls.logLevel = lvl

    @classmethod
    def getLogLevel(cls):
        return cls.logLevel

    @classmethod
    def setFolder(cls, name):
        cls.folder = name + "/"

    @classmethod
    def saveInFile(cls, name, function):
        function.rename(name, name)
        fileName = cls.folder + name + ".pvd"
        print "Saved", name, "as", fileName
        fileHandle = File(fileName)
        fileHandle << function

    @classmethod
    def computeError(cls, solution, solutionMean):
            error = np.math.sqrt(np.mean(np.power(solution - solutionMean, 2)))
            return error


    @classmethod
    def plotConvergenceMean(cls, approx, exact, regression = False):
        nApprox = approx.shape[0]
        errorMean = np.empty(nApprox)
        for curSampleN in range(0, nApprox):
            errorMean[curSampleN] = cls.computeError(np.mean(approx[:curSampleN + 1], 0), exact)

        p.figure()
        # lSpace = p.linspace(0, errorMean.size, errorMean.size)
        lSpace = list(i * (len(exact) ** 2) for i in range(nApprox))
        p.loglog(lSpace, errorMean, "k-", xlabel = "samples", ylabel = "error",
                   legend = "E[(Q_M - E[Q])^2]^{1/2}")
        print("CompCost: {compCost:5.2e}\t Error: {error:5.4e}"
              .format(compCost = lSpace[-1], error = errorMean[-1]))
        if regression:
            p.hold('on')
            lineSpace = np.array((lSpace[0], lSpace[-1]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(lSpace, np.log10(errorMean))
            line = np.power(10, slope * lineSpace + intercept)
            p.plot(lineSpace, line, "k:", legend = "{slope:.2}x + c" \
                   .format(slope = slope))

    @classmethod
    def plotConvergence(cls, scale, error4scale, title, legend, regression = False):
        p.figure()
        p.loglog(scale, error4scale, "k-d", xlabel = "samples", ylabel = "error",)  # semilogy
        p.title(title)
        p.legend(legend)
        if regression and len(error4scale) > 1:
            p.hold('on')
            lineSpace = np.array((scale[0], scale[-1]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(scale, np.log10(error4scale))
            line = np.power(10, slope * lineSpace + intercept)
            p.plot(lineSpace, line, "k:", legend = "{slope:.2}x + c" \
                   .format(slope = slope))
        p.hardcopy(cls.folder + title + "  " + t.strftime('%Y-%m-%d_%H-%M') + ".pdf")
