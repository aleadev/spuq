from dolfin.functions.function import TrialFunction, TestFunction, Function
from distLogNormal import distLogNormal
import numpy as np
from ufl.operators import nabla_grad, div, grad, dot, avg, sqrt
from ufl.objects import dx, dS
from dolfin.fem.bcs import DirichletBC
from dolfin.fem.solving import solve
from dolfin.functions.specialfunctions import FacetNormal, CellSize
from dolfin.functions.functionspace import FunctionSpace
from dolfin.fem.assembling import assemble
from dolfin.cpp.mesh import MeshFunction, cells
from dolfin.mesh.refinement import refine
from dolfin.fem.projection import project
from dolfin.fem.interpolation import interpolate
class problem_SPDE:
    rhs_f = None
    kappa = None
    bcExpression = None
    bcBoundary = None
    fraction = 0.4


    def __init__(self, rhs_f, kappa, bcExpression, bcBoundary, fraction = 0.4):
        self.rhs_f = rhs_f
        self.kappa = kappa
        self.bcExpression = bcExpression
        self.bcBoundary = bcBoundary
        self.fraction = fraction


    def solve(self, spaceV):
        u = TrialFunction(spaceV)
        v = TestFunction(spaceV)
        solution = Function(spaceV)

        a = self.kappa.expression * np.inner(nabla_grad(u), nabla_grad(v)) * dx
        L = self.rhs_f * v * dx

        bc = DirichletBC(spaceV, self.bcExpression, self.bcBoundary)

        solve(a == L, solution, bc)

        return solution


    def estimate(self, solution):
        mesh = solution.function_space().mesh()

        # Define cell and facet residuals
        R_T = -(self.rhs_f + div(grad(solution)))
        n = FacetNormal(mesh)
        R_dT = dot(grad(solution), n)

        # Will use space of constants to localize indicator form
        Constants = FunctionSpace(mesh, "DG", 0)
        w = TestFunction(Constants)
        h = CellSize(mesh)

        # Define form for assembling error indicators
        form = (h ** 2 * R_T ** 2 * w * dx + avg(h) * avg(R_dT) ** 2 * 2 * avg(w) * dS)
    #            + h * R_dT ** 2 * w * ds)

        # Assemble error indicators
        indicators = assemble(form)

        # Calculate error
        error_estimate = sqrt(sum(i for i in indicators.array()))

        # Take sqrt of indicators
        indicators = np.array([sqrt(i) for i in indicators])

        # Mark cells for refinement based on maximal marking strategy
        largest_error = max(indicators)
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            cell_markers[c] = indicators[c.index()] > (self.fraction * largest_error)

        return error_estimate, cell_markers


    def solveMean(self, spaceV):
        self.kappa.setCoeffs(np.zeros(self.kappa.nModes))

        return self.solve(spaceV)


    def sample(self, nModes = 0):
        self.kappa.sample(nModes)


    def computeError4level(self, meanApprox4lvl, lastSpace, meanExact = None):
        if meanExact == None:
            solMesh = lastSpace.mesh()
            solFamily = lastSpace.ufl_element().family()
            solDegree = lastSpace.ufl_element().degree()
            refSpace = FunctionSpace(refine(refine(solMesh)), solFamily, solDegree)
            meanExact = self.solveMean(refSpace)

        refSpace = meanExact.function_space()
        error4lvl = list(
                np.sqrt(assemble(
                    (project(meanApprox, refSpace) - interpolate(meanExact, refSpace)) ** 2 * dx
                )) for meanApprox in meanApprox4lvl)

        return error4lvl
