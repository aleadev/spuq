import numpy as np
from ufl import *
from dolfin import *
import distLogNormal as dist
from dolfin.functions.function import Function
from dolfin.functions.functionspace import FunctionSpace
from dolfin.mesh.refinement import refine
from dolfin.fem.interpolation import interpolate
from dolfin.fem.assembling import assemble


class mlmc:
    initNumSamples = 0

    optNumSamples4Level = [0]
    y4Level = [[]]
    space4Level = []
    variance4level = [None]
    mean4level = [None]
    compCost4lvl = [0]

    problem = None
    constCompCost = 1
    maxLevels = 0
    adaptiveKL = False
    adaptiveMesh = False

    def __init__(self, initNumSamples, constCompCost, maxLevels, initSpace, initMesh,
                 problem, adaptiveKL = False, adaptiveMesh = False):
        self.initNumSamples = initNumSamples
        self.space4Level.append(initSpace)
        self.problem = problem
        self.constCompCost = constCompCost
        self.maxLevels = maxLevels
        self.adaptiveKL = adaptiveKL
        self.adaptiveMesh = adaptiveMesh

    def runCycle(self):
        # initial sampling
        print("Level {level}".format(level = 0))
        self.sampleSolutions()
        self.updateStatistics()
        self.sampleSolutions()

        meanFunction = Function(self.space4Level[0])
        meanFunction.vector().set_local(np.mean(self.y4Level[0], 0))
        self.mean4level[0] = meanFunction

        for level in range(1, self.maxLevels):
            print("Level {level}".format(level = level))
            self.addLevel()
            self.sampleSolutions()
            self.updateStatistics()
            self.sampleSolutions()

            levelFunction = Function(self.space4Level[level])
            levelFunction.vector().set_local(np.mean(self.y4Level[level], 0))
            self.mean4level.append(levelFunction)

            for curLevel in range(level):
                tempFunction = Function(self.space4Level[curLevel])
                tempFunction.vector().set_local(np.mean(self.y4Level[curLevel], 0))
                self.mean4level[level] += tempFunction

    def addLevel(self):
        self.optNumSamples4Level.append(0)
        self.y4Level.append([])
        lastSpace = self.space4Level[-1]
        lastMesh = lastSpace.mesh()
        lastFamiliy = lastSpace.ufl_element().family()
        lastDegree = lastSpace.ufl_element().degree()
        if self.adaptiveMesh:
            solution = self.problem.solveMean(lastSpace)
            _, cell_markers = self.problem.estimate(solution)
            self.space4Level.append(FunctionSpace(refine(lastMesh, cell_markers),
                lastFamiliy, lastDegree))
        else:
            self.space4Level.append(FunctionSpace(refine(lastMesh),
                lastFamiliy, lastDegree))
        self.variance4level.append(None)
        self.compCost4lvl.append(0)

    def sampleSolutions(self):
        """Samples a random field and computes corresponding solutions"""

        for curLevel in range(self.numLevels()):
            if self.optNumSamples4Level[curLevel] == 0:
                self.optNumSamples4Level[curLevel] = self.initNumSamples
            self.computeSamples4Level(curLevel, self.optNumSamples4Level[curLevel]
                                      - len(self.y4Level[curLevel]))

    def computeSamples4Level(self, level, numSamples):
        if numSamples <= 0:
            return

        samples = self.y4Level[level]
        if len(samples) == 0:
            print("\tSampling {samples} initial samples for level {level}"\
                     .format(samples = numSamples, level = level))
        else:
            print("\tSampling {samples} additional samples for level {level}"\
                     .format(samples = numSamples, level = level))
        if level == 0:
            V = self.space4Level[level]
            for curSample in range(numSamples):
                if self.adaptiveKL == True:
                    self.problem.sample(nModes = 1 / V.mesh().hmax())
                else:
                    self.problem.sample()
                solution = self.problem.solve(V)
                samples.append(solution.vector().array())
        else:
            V_fine = self.space4Level[level]
            V_coarse = self.space4Level[level - 1]

            for curSample in range(numSamples):
                if self.adaptiveKL == True:
                    self.problem.sample(nModes = 1 / V_fine.mesh().hmax())
                else:
                    self.problem.sample()
                solutionFine = self.problem.solve(V_fine)
                solutionCoarse = self.problem.solve(V_coarse)
                y = solutionFine.vector().array() - interpolate(solutionCoarse, V_fine).vector().array()
                samples.append(y)

    def updateStatistics(self):
        for curLevel in range(len(self.space4Level)):
            samples = self.y4Level[curLevel]
            self.variance4level[curLevel] = np.var(samples, 0)
            compCost = self.space4Level[curLevel].dim() ** 2 / 1000.

            # Simple for uniform meshes
#             optSamples = self.constCompCost * np.sqrt(
#                 np.sqrt(np.sqrt(np.mean(np.abs(self.variance4level[curLevel]))) / compCost))

            # Same but for general meshes
            space = self.space4Level[curLevel]
            domainArea = assemble(interpolate(Constant(1), space) * dx)
            varianceMean = Function(space)
            varianceMean.vector().set_local(np.abs(self.variance4level[curLevel]))
            varianceIntMean = np.sqrt(assemble(varianceMean * dx) / domainArea)
            optSamples = self.constCompCost * np.sqrt(np.sqrt(
                varianceIntMean / compCost))

            # Alernate from ???
#             l = curLevel + 2
#             maxL = len(self.space4Level) + 1
#             epsOptSamples = 0.1
#             optSamples = l ** (2 + 2 * epsOptSamples) * 2 ** (2 * (maxL - l))

            self.optNumSamples4Level[curLevel] = np.rint(optSamples).astype(int)
        self.compCost4lvl[-1] = sum(list(len(samples) * (space.dim() ** 2)
            for (space, samples) in zip(self.space4Level, self.y4Level)))

    def numLevels(self):
        return len(self.optNumSamples4Level)

    def numberDoF(self):
        totalDoF = 0
        for space in self.space4Level:
            totalDoF += space.dim()
        return totalDoF

    def numberSampledDoF(self):
        total = 0
        for level in range(len(self.space4Level)):
            total += self.space4Level[level].dim() * len(self.y4Level[level])
        return total

    def getStatistics(self):
        level = range(len(self.space4Level))
        dims = list(space.dim() for space in self.space4Level)
        samples = list(len(samp) for samp in self.y4Level)
        return level, dims, samples
