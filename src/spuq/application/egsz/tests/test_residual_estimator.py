from __future__ import division
from operator import itemgetter

from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.math_utils.multiindex import Multiindex
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.utils.testing import assert_equal, assert_almost_equal, skip_if, test_main, assert_raises

try:
    from dolfin import Expression, FunctionSpace, UnitSquare, interpolate, Constant, MeshFunction, cells, refine
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
    aN = 4
    a = [Expression('2.+sin(2.*pi*I*x[0]*x[1])', I=i, degree=3, element=fs.ufl_element())
                                                                for i in range(1, aN)]
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


@skip_if(not HAVE_FENICS, "FEniCS not installed.")
def test_estimator_refinement():
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

    # define coefficient field
    aN = 10
    a = [Expression('2.+sin(2.*pi*I*x[0]*x[1])', I=i, degree=3, element=fs.ufl_element())
                                                                for i in range(1, aN)]
    rvs = [NormalRV(mu=0.5) for _ in range(1, aN - 1)]
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


    # ===================
    # MARK algorithm test
    # ===================

    # setup marking function
    mesh_markers = {}
    for mu in w.active_indices():
        mesh = w[mu].basis._fefs.mesh()
        mesh_markers[mu] = MeshFunction("bool", mesh, mesh.topology().dim())
        mesh_markers[mu].set_all(False)

    # residual marking
    theta_eta = 0.3
    global_res = sum([res[1] for res in reserr.items()])
    allresind = [(resind[mu].coeffs[i], i, mu) for i in range(len(resind[mu].coeffs)) for mu in resind.active_indices()]
    allresind = sorted(allresind, key=itemgetter(1))
    # TODO: check that indexing and cell ids are consistent (it would be safer to always work with cell indices) 
    marked_res = 0
    for res in allresind:
        if marked_res >= theta_eta * global_res:
            break
        mesh_markers[res[2]][res[1]] = True
        marked_res += res[0]
        
    print "RES MARKED elements:\n", [(mu, mesh_markers[mu].array().sum()) for mu in mesh_markers]
    
    # projection marking
    theta_zeta = 0.8
    max_zeta = max([max(projind[mu].coeffs) for mu in projind.active_indices()])
    for mu in projind.active_indices():
        indmu = [i for i, p in enumerate(projind[mu].coeffs) if p >= theta_zeta * max_zeta]
        print "PROJ MARKING", len(indmu), "elements in", mu
        for i in indmu:
            mesh_markers[mu][i] = True

    print "FINAL MARKED elements:\n", [(mu, mesh_markers[mu].array().sum()) for mu in mesh_markers]

    # refine meshes
#    mesh = refine(mesh, cell_markers)
    
    # new multiindex activation
#    am_f, am_rv = coeff_field[0]
#    beta = am_rv.orth_polys.get_beta(1)
#    beta[0]
#    print "additional projection error indices are ", set(projind.active_indices()) - set(resind.active_indices())


test_main()
