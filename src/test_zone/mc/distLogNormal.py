from dolfin.functions.expression import Expression
from math import *
from scipy.stats import *
from tools import *

class distLogNormal:
    expression = None
    nModes = 0
    kappaMeanString = ""
    sigma = 1
    mu = 0
    scale = 1


    def __init__(self, nModes, kappaMeanString, sigma, mu, scale):
        self.nModes = nModes
        self.kappaMeanString = kappaMeanString
        self.sigma = sigma
        self.mu = mu
        self.scale = scale

        self.expression = self.composeModes(nModes)

    def composeMode(self, m):
        m += 1
        alpha_m = 1 * (m ** (-1))
        k_m = floor(-1. / 2 + sqrt(1. / 4 + 2 * m))
        beta_one_m = m - k_m * (k_m + 1) / 2.
        beta_two_m = k_m - beta_one_m
        a_m_expression = "{alpha_m}*cos(2*3.14*{beta_one_m}*x[0])*cos(2*3.14*{beta_two_m}*x[1])"\
                         .format(alpha_m = alpha_m, beta_one_m = beta_one_m, \
                         beta_two_m = beta_two_m)
        return a_m_expression


    def composeModes(self, nModes):
        tools.status("\tComposing " + repr(nModes) + " modes", 2)

        modes = ""
        parameterList = {}

        tools.status("\tCreate expression strings", 3)

        for curMode in range(nModes):
            modes += "+ coeff" + repr(curMode) + "*" + self.composeMode(curMode)
            # parameterList += ",coeff" + repr(curMode) + "=0"
            parameterList["coeff" + repr(curMode)] = 0

        tools.status("\tCompile expression", 3)

        expString = "Expression(\"" + self.kappaMeanString + " + " + \
                          modes + "\"" + ", **parameterList )"
        modesExpr = eval(expString)

        tools.status("", 3)
        return modesExpr

    def setCoeffs(self, coeffs):
        tools.status("\tSetting coefficiants", 3)

        for curMode in range(self.nModes):
            setString = "self.expression.coeff" + repr(curMode) \
                            + " = " + repr(coeffs[curMode])
            exec(setString)

        return self.expression


    def sample(self, nModes = 0):
        if nModes == 0:
            nModes = self.nModes
        nModes = min(nModes, self.nModes)
        tools.status("\tSampling coefficiants", 3)
#         coeffs = lognorm.rvs(self.sigma, loc = self.mu, scale = self.scale, \
#                               size = self.nModes)
        coeffs = np.zeros(self.nModes)
        coeffs[:nModes] = norm.rvs(self.sigma, loc = self.mu, scale = self.scale, \
                              size = nModes)

        return self.setCoeffs(coeffs)
