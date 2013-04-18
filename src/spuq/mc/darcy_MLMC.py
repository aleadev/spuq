"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization with FEniCS.

-Laplace(u) = f on the unit square.
u = u0 on tssssssssshe boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""
import numpy as np
import time as t
from tools import *
from ufl import *
from dolfin import *
import distLogNormal as dist
from mlmc import *
from problem_SPDE import problem_SPDE
from dolfin.cpp.mesh import UnitSquareMesh, Mesh
from dolfin.functions.expression import Expression
from dolfin.cpp.common import parameters, set_log_level
from dolfin.fem.bcs import DirichletBC
from dolfin.fem.projection import project
from dolfin.cpp.io import interactive
from dolfin.common.plotting import plot


'''
INPUT SETTINGS
'''
set_log_level(ERROR)            # FENICS Info Level: DEBUG, PROGRESS, ERROR
parameters["num_threads"] = 4   # Number of CPUs
tools.setLogLevel(2)            # MLMC Info Level: 0-4
tools.setFolder("darcy_MLMC")   # relative Sub-Folder for file outpu


nDoF4Side = 10                  # max ~ 600 for 3 GB
elementType = 'Lagrange'        # Type of the FE
elementPower = 1                # Degree of the FE
adaptiveMesh = False


kappaMeanString = '(sin(x[0]*2*pi)*sin(x[1]*2*pi) + 2)*10+100'  # Mean of Kappa
f = Expression('pow((x[0]*x[1]*(1-x[0])*(1-x[1])*1000),2)')     # RHS f
p0 = Constant(0)                                                # u_D on Dirichlet boundary
def p0_boundary(x, on_boundary):                               # Location of the Dirichlet boundary
    return on_boundary
# initDomain = refine(Mesh("lshape_klein.xml.gz"))                # Mesh of the Domain
initDomain = UnitSquareMesh(nDoF4Side, nDoF4Side)

mu = 0                  # Parameter mu for the distribution
sigma = 0.5             # Parameter sigma for the distribution
scale = 1               # scale of the distribution
nModes = 10           # Number of Modes to generate

maxLevels = 5           # Number of Levels for MLMC: max 5/10
initNumSamples = 10     # Initial Number of samples per level
constCompCost = 8000    # Computational Cost Constant: 8000/12000
adaptiveKL = False




'''
Start of Main Programm
'''
kappaMean = Expression(kappaMeanString)
initSpace = FunctionSpace(initDomain, elementType, elementPower)
V = initSpace
kappa = dist.distLogNormal(nModes, kappaMeanString, sigma, mu, scale)
problem = problem_SPDE(f, kappa, p0, p0_boundary)
timePassed = t.clock()


"""
SOLVING
"""
sampler = mlmc(initNumSamples, constCompCost, maxLevels, initSpace, initDomain,
               problem, adaptiveKL = adaptiveKL, adaptiveMesh = adaptiveMesh)
sampler.runCycle()


"""
EVALUATION
"""
timeGesamt = t.clock() - timePassed
tools.status("Estimate error", 1)
# refSpace = sampler.space4Level[-1]
# meanExact = problem.solveMean(refSpace)
# error4lvl = list(tools.computeError(
#                 meanExact.vector().array(),
#                 project(meanApprox, refSpace).vector().array()
#                 ) for meanApprox in sampler.mean4level)
error4lvl = problem.computeError4level(sampler.mean4level, sampler.space4Level[-1])

tools.status("_"*33 + "\nGesamtzeit\t\t{:0>2.0f}:{:0>2.0f} min"
             .format(timeGesamt / 60, timeGesamt % 60), 1)
tools.status("nDoF\t\t\t" + repr(sampler.numberDoF()), 1)
tools.status("nSampledDoF\t\t" + repr(sampler.numberSampledDoF()), 1)
tools.status("", 1)
levels, dims, samples = sampler.getStatistics()
print "Level\t  DoF\tSamples\t  CompCost  Error\n" + "" + "-"*44
for (number, dims, samples, compCost, error) in zip(levels, dims, samples, sampler.compCost4lvl, error4lvl):
    print("{number:5}\t{dims:5}\t{samples:7}\t  {compCost:5.2e}  {error:5.2e}"
        .format(number = number, dims = dims, samples = samples, compCost = compCost, error = error))
print

tools.plotConvergence(sampler.compCost4lvl, error4lvl, title = "Convergence Plot (MLMC)",
                      legend = 'E[(Q_M - E[Q])^2]^{1/2}', regression = True)
# plot(sampler.space4Level[-1].mesh(), title = "Mesh on level {level}"
#      .format(level = len(sampler.space4Level) - 1))
# interactive()
raw_input("Press Enter to continue...")
print("\nFinished!")
