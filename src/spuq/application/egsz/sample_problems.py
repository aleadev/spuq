from __future__ import division

from spuq.application.egsz.coefficient_field import ParametricCoefficientField
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.type_check import takes, anything, optional

from dolfin import Expression, Mesh, refine, CellFunction, FiniteElement
import ufl
from itertools import count
from exceptions import KeyError, NameError
from numpy import random
from math import ceil

class SampleProblem(object):
    @classmethod
    @takes(anything, Mesh, int, optional(dict))
    def setupMeshes(cls, mesh, N, params=None):
        """Create set of N meshes based on provided mesh. Parameters refine>=0 and 0<random<=1.0 specify refinement adjustments."""
        try:
            ref = int(params["refine"])
            assert ref >= 0
        except (KeyError, NameError):
            ref = 0
        try:
            randref = params["random"]
            assert 0 < randref[0] <= 1.0
            assert 0 < randref[1] <= 1.0
        except (KeyError, NameError):
            randref = (1.0, 1.0)
            # create set of (refined) meshes
        meshes = list();
        for _ in range(N):
            m = Mesh(mesh)
            for _ in range(ref):
                cell_markers = CellFunction("bool", m)
                if randref == 1.0:
                #                    cell_markers.set_all(True)
                    m = refine(m)
                else:
                    cell_markers.set_all(False)
                    if random.random() <= randref[0]:
                        cids = set()
                        while len(cids) < ceil(m.num_cells() * randref[1]):
                            cids.add(random.randint(0, m.num_cells()))
                        for cid in cids:
                            cell_markers[cid] = True
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
    @takes(anything, str, optional(dict))
    def setupCF(cls, cftype, decayexp=2, amp=1, freqscale=1, rvtype='uniform'):
        """create parametric coefficient field of cftype (EF-square-cos,EF-square-sin,monomials,linear,constant,zero) with
            decay exponent, amplification and random variable type (uniform,normal)"""
        # mean value
        a0 = Expression("1.0", element=FiniteElement('Lagrange', ufl.triangle, 1))
        # random variables
        if rvtype == "uniform":
            rvs = lambda _: UniformRV().scale(0.5)
        else:
            rvs = lambda _: NormalRV(mu=0.5)

        if cftype == "EF-square-cos":
            # eigenfunctions on unit square
            mis = MultiindexSet.createCompleteOrderSet(2)
            mis.next()
            a = (Expression('A*cos(freq*pi*m*x[0])*cos(freq*pi*n*x[1])', freq=freqscale, A=amp / (int(i) + 2) ** decayexp,
                    m=int(mu[0]), n=int(mu[1]), degree=3,
                    element=FiniteElement('Lagrange', ufl.triangle, 1)) for i, mu in enumerate(mis))
        elif cftype == "EF-square-sin":
            # eigenfunctions on unit square
            mis = MultiindexSet.createCompleteOrderSet(2)
            mis.next()
            a = (Expression('A*sin(freq*pi*m*x[0])*sin(freq*pi*n*x[1])', freq=freqscale, A=amp / (int(i) + 2) ** decayexp,
                    m=int(mu[0]), n=int(mu[1]), degree=3,
                    element=FiniteElement('Lagrange', ufl.triangle, 1)) for i, mu in enumerate(mis))
        elif cftype == "monomials":
            # monomials
            mis = MultiindexSet.createCompleteOrderSet(2)
            mis.next()
            p_str = lambda A, m, n: str(A) + "*" + "*".join(["x[0]" for _ in range(m)]) + "+" + "*".join(["x[1]" for _ in range(n)])
            pex = lambda i, mn: Expression(p_str(amp / (int(i) + 1) ** decayexp, mn[0], mn[1]), degree=3,
                element=FiniteElement('Lagrange', ufl.triangle, 1))
            a = (pex(i, mn + freqscale) for i, mn in enumerate(mis))
        elif cftype == "linear":
            # linear functions
            a = (Expression("A*(x[0]+x[1])", A=amp / (int(i) + 1) ** decayexp,
                element=FiniteElement('Lagrange', ufl.triangle, 1)) for i in count())
        elif cftype == "constant":
            # constant functions
            a = (Expression(str(amp / (int(i) + 1) ** decayexp),
                element=FiniteElement('Lagrange', ufl.triangle, 1)) for i in count())
        elif cftype == "zero":
            # zero functions
            a = (Expression("0.0", element=FiniteElement('Lagrange', ufl.triangle, 1)) for _ in count())
        else:
            raise TypeError('unsupported function type')

        return ParametricCoefficientField(a0, a, rvs)
