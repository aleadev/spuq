from __future__ import division

from spuq.application.egsz.coefficient_field import ParametricCoefficientField
from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.utils.parametric_array import ParametricArray
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
        w = MultiVectorWithProjection()
        for mu, mesh in mi_mesh.iteritems():
            w[mu] = setup_vec(mesh)
        return w

    @classmethod
    @takes(anything, str, optional(dict))
    def setupCF(cls, cftype, params=None):
        try:    # decay exponent
            k = params["exp"]
        except:
            k = 2
        try:    # amplitude
            amp_val = params["amp"] 
            def amp(i):
                if i > 0:
                    return amp_val
                else:
                    return 1 
        except:
            amp = lambda _: 1
        # mean value
        a0 = Expression("1.0", element=FiniteElement('Lagrange', ufl.triangle, 1))
        # random variables
        #        rvs = (NormalRV(mu=0.5) for _ in count())
        rvs = lambda _: UniformRV().scale(0.5)

        if cftype == "EF-square":
            # eigenfunctions on unit square
            mis = MultiindexSet.createCompleteOrderSet(2)
            a = (Expression('A*sin(pi*m*x[0])*sin(pi*n*x[1])', A=amp(int(i)) / (int(i) + 2) ** k, m=int(mu[0]) + 1,
                n=int(mu[1]) + 1, degree=2,
                #            a = (Expression('A*sin(pi*m*x[0])*sin(pi*n*x[1])', A=1 / (mu[0] + mu[1] + 1) ** 2, m=int(mu[0]), n=int(mu[1]), degree=2,
                element=FiniteElement('Lagrange', ufl.triangle, 1)) for i, mu in enumerate(mis))
        elif cftype == "monomials":
            # monomials
            mis = MultiindexSet.createCompleteOrderSet(2)
            p_str = lambda A, m, n: str(A) * "*" + "*".join(["x[0]" for _ in range(m)]) + "+" + "*".join(
                ["x[1]" for _ in range(n)])
            pex = lambda i, mn: Expression(p_str(amp(int(i)) / (int(i) + 2) ** k, mn[0], mn[1]), degree=2,
                element=FiniteElement('Lagrange', ufl.triangle, 1))
            a = (pex(i, mn) for i, mn in enumerate(mis))
        elif cftype == "linear":
            # linear functions
            a = (Expression("A*(x[0]+x[1])", A=amp(int(i)) / (int(i) + 2) ** k,
                element=FiniteElement('Lagrange', ufl.triangle, 1)) for i in count())
        elif cftype == "constant":
            # constant functions
            a = (Expression(str(amp(int(i)) / (int(i) + 2) ** k),
                element=FiniteElement('Lagrange', ufl.triangle, 1)) for i in count())
        elif cftype == "zero":
            # zero functions
            a = (Expression("0.0", element=FiniteElement('Lagrange', ufl.triangle, 1)) for _ in count())
        else:
            raise TypeError('unsupported function type')

        return ParametricCoefficientField(a0, a, rvs)
