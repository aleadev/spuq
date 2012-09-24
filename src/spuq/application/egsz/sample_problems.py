from __future__ import division

from math import ceil
from exceptions import NameError
from numpy.random import random, shuffle
from scipy.special import zeta

from spuq.application.egsz.coefficient_field import ParametricCoefficientField
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import takes, anything, optional

from dolfin import Expression, Mesh, refine, CellFunction, FiniteElement, Constant
import ufl

import logging

logger = logging.getLogger(__name__)


class SampleProblem(object):
    @classmethod
    @takes(anything, Mesh, int, optional(int), optional(tuple))
    def setupMeshes(cls, mesh, N, num_refine=0, randref=(1.0, 1.0)):
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
        for _ in range(N):
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
            meshes.append(m)
        return meshes

    @classmethod
    @takes(anything, dict, callable)
    def setupMultiVector(cls, mi_mesh, setup_vec):
        w = MultiVectorWithProjection(cache_active=True)
        for mu, mesh in mi_mesh.iteritems():
            w[mu] = setup_vec(mesh)
        return w

    @classmethod
    #@takes(anything, str, str, optional(dict))
    def setupCF2(cls, functype, amptype, rvtype='uniform', gamma=0.9, decayexp=2, freqscale=1, freqskip=0, N=1, scale=1, dim=2):
        if rvtype == "uniform":
            rvs = lambda i: UniformRV(a= -1, b=1)
        elif rvtype == "normal":
            rvs = lambda i: NormalRV(mu=0.5)
        else:
            raise ValueError("Unkown RV type %s", rvtype)

        if functype == "cos":
            if dim == 1:
                func = "A*B*cos(freq*pi*m*x[0])"
            else:
                func = "A*B*cos(freq*pi*m*x[0])*cos(freq*pi*n*x[1])"
        elif functype == "sin":
            if dim == 1:
                func = "A*B*sin(freq*pi*(m+1)*x[0])"
            else:
                func = "A*B*sin(freq*pi*(m+1)*x[0])*sin(freq*pi*(n+1)*x[1])"
        elif functype == "monomials":
            if dim == 1:
                func = "A*B*pow(x[0],freq*m)"
            else:
                func = "A*B*pow(x[0],freq*m)*pow(x[1],freq*n)"
        elif functype == "constant":
#            func = "A*B*1.0"
#            func = "(A/A)*(B/B)*1.0"
            func = "1.0+A+B-A-B"
        else:
            raise ValueError("Unkown func type %s", functype)
            
        def get_decay_start(exp, gamma=1):
            start = 1
            while zeta(exp, start) >= gamma:
                start += 1
            return start

        if amptype == "decay-inf":
            start = get_decay_start(decayexp, gamma)
            amp = gamma / zeta(decayexp, start)
            logger.info("type is decay_inf with start = " + str(start) + " and amp = " + str(amp))
            ampfunc = lambda i: amp / (float(i) + start) ** decayexp
        elif amptype == "constant": 
            amp = gamma / N
            ampfunc = lambda i: gamma * (i < N)
        else:
            raise ValueError("Unkown amplitude type %s", amptype)

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

        return ParametricCoefficientField(a0, a, rvs)

    @classmethod
    @takes(anything, str, optional(dict))
    def setupCF(cls, cftype, decayexp=2, gamma=0.9, freqscale=1, freqskip=0, N=1, rvtype='uniform', scale=1, dim=2):
        if cftype == "EF-square-cos":
            return cls.setupCF2("cos", "decay-inf", rvtype=rvtype, gamma=gamma, decayexp=decayexp, freqscale=freqscale, freqskip=freqskip, N=N, scale=scale, dim=dim)
        elif cftype == "EF-square-sin":
            return cls.setupCF2("sin", "decay-inf", rvtype=rvtype, gamma=gamma, decayexp=decayexp, freqscale=freqscale, freqskip=freqskip, N=N, scale=scale, dim=dim)
        elif cftype == "monomials":
            return cls.setupCF2("monomials", "decay-inf", rvtype=rvtype, gamma=gamma, decayexp=decayexp, freqscale=freqscale, freqskip=freqskip, N=N, scale=scale, dim=dim)
        elif cftype == "linear":
            return cls.setupCF2("monomials", "constant", rvtype=rvtype, gamma=gamma, N=2, scale=scale, dim=dim)
        elif cftype == "constant":
            return cls.setupCF2("constant", "constant", rvtype=rvtype, gamma=gamma, N=2, scale=scale, dim=dim)
        else:
            raise ValueError('Unsupported function type: %s', cftype)

        return ParametricCoefficientField(a0, a, rvs)
