from __future__ import division

from math import ceil
from numpy.random import random, shuffle
from scipy.special import zeta
from collections import namedtuple

from spuq.application.egsz.coefficient_field import ParametricCoefficientField
from spuq.application.egsz.multi_vector import MultiVectorSharedBasis
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import takes, anything, optional
from spuq.application.egsz.fem_discretisation import FEMPoisson
from spuq.application.egsz.fem_discretisation import FEMNavierLame

from dolfin import Expression, Mesh, refine, CellFunction, FiniteElement, Constant
import ufl

import logging

logger = logging.getLogger(__name__)


class SampleProblem(object):

    # Definition of PDE types
    POISSON = "poisson"
    NAVIER_LAME = "navier_lame"
    pde_types = [POISSON, NAVIER_LAME]

    # Definitions for coefficient fields
    func_defs = dict()
    func_defs[("cos", 1)] = "A*B*cos(freq*pi*m*x[0])"
    func_defs[("cos", 2)] = "A*B*cos(freq*pi*m*x[0])*cos(freq*pi*n*x[1])"
    func_defs[("sin", 1)] = "A*B*sin(freq*pi*(m+1)*x[0])"
    func_defs[("sin", 2)] = "A*B*sin(freq*pi*(m+1)*x[0])*sin(freq*pi*(n+1)*x[1])"
    func_defs[("monomials", 1)] = "A*B*pow(x[0],freq*m)"
    func_defs[("monomials", 2)] = "A*B*pow(x[0],freq*m)*pow(x[1],freq*n)"
    func_defs[("constant", 1)] = "A*B*1.0"
    func_defs[("constant", 1)] = "1.0+A-A+B-B"
    func_defs[("constant", 2)] = func_defs[("constant", 1)]

    old_coeff_types = dict()
    old_coeff_types["EF-square-cos"] = ("cos", "decay-inf")
    old_coeff_types["EF-square-sin"] = ("sin", "decay-inf")
    old_coeff_types["monomials"] = ("monomials", "decay-inf")
    old_coeff_types["linear"] = ("monomials", "constant")
    old_coeff_types["constant"] = ("constant", "constant")

    # Defintions for right hand side functiosn
    defaults = dict()
    defaults[(NAVIER_LAME, "rhs")] = "zero"
    defaults[(POISSON, "rhs")] = "constant"

    rhs_defs = dict()
    rhs_defs[(NAVIER_LAME, "zero")] = Constant((0.0, 0.0))
    rhs_defs[(POISSON, "zero")] = Constant(0.0)
    rhs_defs[(POISSON, "constant")] = Constant(1.0)

    # Definitions for boundary conditions
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    BoundaryDef = namedtuple("BoundaryDef", "type where func")

    boundary_defs = dict()
    boundary_defs[(NAVIER_LAME, "dirichlet_normal_stress")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant((0.0, 0.0))),
                                                              BoundaryDef(type=DIRICHLET, where=RIGHT, func=Constant((0.3, 0.0)))]
    boundary_defs[(NAVIER_LAME, 0)] = boundary_defs[(NAVIER_LAME, "dirichlet_normal_stress")]

    boundary_defs[(NAVIER_LAME, "dirichlet_shear_stress")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant((0.0, 0.0))),
                                                             BoundaryDef(type=DIRICHLET, where=RIGHT, func=Constant((0.0, 0.3)))]
    boundary_defs[(NAVIER_LAME, 1)] = boundary_defs[(NAVIER_LAME, "dirichlet_shear_stress")]

    boundary_defs[(NAVIER_LAME, "dirichlet_normal_shear")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant((0.0, 0.0))),
                                                             BoundaryDef(type=DIRICHLET, where=RIGHT, func=Constant((1.0, 1.0)))]
    boundary_defs[(NAVIER_LAME, 2)] = boundary_defs[(NAVIER_LAME, "dirichlet_normal_shear")]

    boundary_defs[(NAVIER_LAME, "neumann_shear")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant((0.0, 0.0))),
                                                      BoundaryDef(type=NEUMANN, where=RIGHT, func=Constant((0.0, 1.0)))]
    boundary_defs[(NAVIER_LAME, 3)] = boundary_defs[(NAVIER_LAME, "neumann_shear")]

    boundary_defs[(NAVIER_LAME, "dirichlet_top_neumann_normal")] = [BoundaryDef(type=DIRICHLET, where=TOP, func=Constant((0.0, 0.0))),
                                                                    BoundaryDef(type=NEUMANN, where=RIGHT, func=Constant((1.0, 0.0)))]
    boundary_defs[(NAVIER_LAME, 4)] = boundary_defs[(NAVIER_LAME, "dirichlet_top_neumann_normal")]



    boundary_defs[(POISSON, "dirichlet_zero_left")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant(0.0))]
    boundary_defs[(POISSON, 0)] = boundary_defs[(POISSON, "dirichlet_zero_left")]
    boundary_defs[(POISSON, "dirichlet_inhomogeneous")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant(0.0)),
                                                           BoundaryDef(type=DIRICHLET, where=RIGHT, func=Constant(1.0))]
    boundary_defs[(POISSON, 1)] = boundary_defs[(POISSON, "dirichlet_inhomogeneous")]
    boundary_defs[(POISSON, "dirichlet_zero_all")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant(0.0)),
                                                      BoundaryDef(type=DIRICHLET, where=RIGHT, func=Constant(0.0)),
                                                      BoundaryDef(type=DIRICHLET, where=TOP, func=Constant(0.0)),
                                                      BoundaryDef(type=DIRICHLET, where=BOTTOM, func=Constant(0.0))]
    boundary_defs[(POISSON, 2)] = boundary_defs[(POISSON, "dirichlet_zero_all")]
    boundary_defs[(POISSON, "dirichlet_neumann1")] = [BoundaryDef(type=DIRICHLET, where=LEFT, func=Constant(0.0)),
                                                      BoundaryDef(type=NEUMANN, where=RIGHT, func=Constant(1.0))]
    boundary_defs[(POISSON, 3)] = boundary_defs[(POISSON, "dirichlet_neumann1")]
    boundary_defs[(POISSON, "dirichlet_neumann2")] = [BoundaryDef(type=DIRICHLET, where=TOP, func=Constant(1.0)),
                                                      BoundaryDef(type=NEUMANN, where=RIGHT, func=Constant(1.0))]
    boundary_defs[(POISSON, 4)] = boundary_defs[(POISSON, "dirichlet_neumann2")]

    # definitions for random variables
    UNIFORM = "uniform"
    NORMAL = "normal"

    rv_defs = dict()
    rv_defs[UNIFORM] = lambda i: UniformRV(a= -1, b=1)
    rv_defs[NORMAL] = lambda i: NormalRV(mu=0.5)


    @classmethod
    @takes(anything, Mesh, int, optional(int), optional(tuple))
    def setupMesh(cls, mesh, num_refine=0, randref=(1.0, 1.0)):
        """Create a set of N meshes based on provided mesh. Parameters
        num_refine>=0 and randref specify refinement
        adjustments. num_refine specifies the number of refinements
        per mesh, randref[0] specifies the probability that a given
        mesh is refined, and randref[1] specifies the probability that
        an element of the mesh is refined (if it is refined at all).
        """
        assert num_refine >= 0

        assert 0 < randref[0] <= 1.0
        assert 0 < randref[1] <= 1.0

        # create set of (refined) meshes
        meshes = list();
        m = Mesh(mesh)
        for _ in range(num_refine):
            if randref[0] == 1.0 and randref[1] == 1.0:
                m = refine(m)
            elif random() <= randref[0]:
                cell_markers = CellFunction("bool", m)
                cell_markers.set_all(False)
                cell_ids = range(m.num_cells())
                shuffle(cell_ids)
                num_ref_cells = int(ceil(m.num_cells() * randref[1]))
                for cell_id in cell_ids[0:num_ref_cells]:
                    cell_markers[cell_id] = True
                m = refine(m, cell_markers)
        return m

    @classmethod
    @takes(anything, dict, callable)
    def setupMultiVector(cls, mesh, mis, setup_vec):
        w = MultiVectorSharedBasis()
        for mu in mis:
            w[mu] = setup_vec(mesh)
        return w

    @staticmethod
    def get_decay_start(exp, gamma=1):
        start = 1
        while zeta(exp, start) >= gamma:
            start += 1
        return start

    @classmethod
    #@takes(anything, str, str, optional(dict))
    def setupCF2(cls, functype, amptype, rvtype='uniform', gamma=0.9, decayexp=2, freqscale=1, freqskip=0, N=1, scale=1, dim=2, secondparam=None):
        try:
            rvs = cls.rv_defs[rvtype]
        except KeyError:
            raise ValueError("Unknown RV type %s", rvtype)

        try:
            func = cls.func_defs[(functype, dim)]
        except KeyError:
            raise ValueError("Unknown function type %s for dim %s", functype, dim)
            
        if amptype == "decay-inf":
            start = SampleProblem.get_decay_start(decayexp, gamma)
            amp = gamma / zeta(decayexp, start)
            ampfunc = lambda i: amp / (float(i) + start) ** decayexp
            logger.info("type is decay_inf with start = " + str(start) + " and amp = " + str(amp))
        elif amptype == "constant": 
            amp = gamma / N
            ampfunc = lambda i: gamma * (i < N)
        else:
            raise ValueError("Unknown amplitude type %s", amptype)

        logger.info("amp function: %s", str([ampfunc(i) for i in range(10)]))
        element = FiniteElement('Lagrange', ufl.triangle, 1)
        # NOTE: the explicit degree of the expression should influence the quadrature order during assembly
        degree = 3

        mis = MultiindexSet.createCompleteOrderSet(dim)
        for i in range(freqskip + 1):
            mis.next()

        a0 = Expression("B", element=element, B=scale)
        if dim == 1:
            a = (Expression(func, freq=freqscale, A=ampfunc(i), B=scale,
                            m=int(mu[0]), degree=degree, element=element) for i, mu in enumerate(mis))
        else:
            a = (Expression(func, freq=freqscale, A=ampfunc(i), B=scale,
                            m=int(mu[0]), n=int(mu[1]),
                            degree=degree, element=element) for i, mu in enumerate(mis))

        if secondparam is not None:
            from itertools import izip
            a0 = (a0, secondparam[0])
            a = ((am, bm) for am, bm in izip(a, secondparam[1]))
        return ParametricCoefficientField(a0, a, rvs)

    @classmethod
    @takes(anything, str, optional(dict))
    def setupCF(cls, cftype, decayexp=2, gamma=0.9, freqscale=1, freqskip=0, N=2, rvtype='uniform', scale=1, dim=2, secondparam=None):
        try:
            func_type, amp_type = cls.old_coeff_types[cftype]
        except KeyError:
            raise ValueError('Unsupported coefficient type: %s', cftype)
        return cls.setupCF2(func_type, amp_type, rvtype=rvtype, gamma=gamma, decayexp=decayexp, freqscale=freqscale, freqskip=freqskip, N=N, scale=scale, dim=dim, secondparam=secondparam)

    @classmethod
    @takes(anything, int, str, int)
    def setupPDE(cls, boundary_type, domain_name, problem_type, boundaries, coeff_field):
        pde_type = cls.pde_types[problem_type]

        # define source term
        #f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)
        rhs_type = cls.defaults[(pde_type, "rhs")]
        f = cls.rhs_defs[(pde_type, rhs_type)]

        Dirichlet_boundary = []
        uD = []
        Neumann_boundary = []
        g = []

        try:
            for bc_def in cls.boundary_defs[(pde_type, boundary_type)]:
                btype, where, func = bc_def
                if btype == cls.NEUMANN:
                    Neumann_boundary.append(boundaries[where])
                    g.append(func)
                elif btype == cls.DIRICHLET:
                    Dirichlet_boundary.append(boundaries[where])
                    uD.append(func)
                else:
                    assert False
        except KeyError:
            # keine Ahnung gerade
            raise

        a0 = coeff_field.mean_func
        
        if pde_type == cls.NAVIER_LAME:
            pde = FEMNavierLame(lmbda0=a0[0], mu0=a0[1], f=f,
                                dirichlet_boundary=Dirichlet_boundary, uD=uD,
                                neumann_boundary=Neumann_boundary, g=g)
        elif pde_type == cls.POISSON:
            pde = FEMPoisson(a0=a0, f=f,
                             dirichlet_boundary=Dirichlet_boundary, uD=uD,
                             neumann_boundary=Neumann_boundary, g=g)
        else:
            assert False

        return pde, Dirichlet_boundary, uD, Neumann_boundary, g, f
