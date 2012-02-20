from __future__ import division

from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.math_utils.multiindex import Multiindex
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.utils.testing import assert_equal, assert_almost_equal, skip_if, test_main, assert_raises

try:
    from dolfin import Expression, FunctionSpace, UnitSquare, interpolate, Constant
    from spuq.application.egsz.residual_estimator import ResidualEstimator
#    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_vector import FEniCSVector
    HAVE_FENICS = True
except:
    HAVE_FENICS = False
    
@skip_if(not HAVE_FENICS, "FEniCS not installed.")
def test_estimator():
    # setup solution multi vector
    mis = [Multiindex([0]),
           Multiindex([1]),
           Multiindex([0, 1]),
           Multiindex([0, 2])]
    mesh = UnitSquare(4, 4)
    fs = FunctionSpace(mesh, "CG", 1)
    F = [interpolate(Expression("*".join(["x[0]"] * i)) , fs) for i in range(1, 5)]
    vecs = [FEniCSVector(f) for f in F]

    w = MultiVectorWithProjection()
    for mi, vec in zip(mis, vecs):
        w[mi] = vec
#    v = A * w

    # define coefficient field
    a = [Expression('2.+sin(pi*I*x[0]*x[1])', I=i, degree=2, element=fs.ufl_element())
                                                                for i in range(1, 4)]
    rvs = [UniformRV(), NormalRV(mu=0.5)]
    coeff_field = CoefficientField(a, rvs)

    # define source term
    f = Constant("1.0")

    # evaluate residual and projection error estimators
    resind, reserr = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, f)
    projind = ResidualEstimator.evaluateProjectionError(w, coeff_field)
    print resind[mis[0]].as_array().shape, projind[mis[0]].as_array().shape
    print "RESIDUAL:", resind[mis[0]].as_array()
    print "PROJECTION:", projind[mis[0]].as_array()
    print "residual error estimate for mu"
    for mu in reserr:
        print "\t", mu, " is ", reserr[mu]
    
    assert_equal(w.active_indices(), resind.active_indices())
    print "active indices are ", resind.active_indices()
    print "additional projection error indices are ", set(projind.active_indices()) - set(resind.active_indices())
    
#    # MultiindexSet
#    m = 2
#    p = 3
#    mi = MultiindexSet.createCompleteOrderSet(m, p)
#
#    # define articifial coefficient field
#    F = list()
#    for i, j in enumerate(mi.arr.tolist()):
#        ex1 = Expression('x[0]*x[0]+A*sin(10.*x[1])', A=i)
#        Dex1 = Expression('2.*x[0]+A*10.*sin(10.*x[1])', A=i)
#        F.append(FEniCSExpression(fexpression=ex1, Dfexpression=Dex1))
#    CF = CoefficientField(F, (NormalRV(),))
#
#    # set zero initial solution
#    # init MultiVector
#    mesh = UnitSquare(5, 5)
#    degree = 1
#    V = FunctionSpace(mesh, "CG", degree)
#    initvector = FEniCSVector(basis=V)
#    wN = MultiVector(mi, initvector)
#
#    # define source term
#    f = FEniCSExpression("1.0", constant=True)
#
#    # evaluate residual and projection error estimators
#    RE = ResidualEstimator(CF, f, wN, 10, FEniCSBasis.PROJECTION.INTERPOLATION)
#    res = RE.evaluateResidualEstimator()
#    proj = RE.evaluateProjectionError()
#    print mesh
#    print res.shape, proj.shape
#    print res, shape

test_main()
