from __future__ import division
from operator import itemgetter
from collections import defaultdict

from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.math_utils.multiindex import Multiindex
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.utils.testing import assert_equal, assert_almost_equal, skip_if, test_main, assert_raises

try:
    from dolfin import Expression, FunctionSpace, UnitSquare, interpolate, Constant, MeshFunction, cells, refine, plot, interactive
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


    # refinement loop
    refinements = 1

    for refinement in range(refinements):
        print "REFINEMENT LOOP iteration ", refinement + 1
        
        # evaluate residual and projection error estimates
        resind, reserr = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, f)
        projind = ResidualEstimator.evaluateProjectionError(w, coeff_field)
    
        # ===================
        # MARK algorithm test
        # ===================
    
        # setup marking sets
        mesh_markers = defaultdict(set)
    
        # residual marking
        theta_eta = 0.3
        global_res = sum([res[1] for res in reserr.items()])
        allresind = list()
        for mu, resmu in resind.iteritems():
            allresind = allresind + [(resmu.coeffs[i], i, mu) for i in range(len(resmu.coeffs))]
        allresind = sorted(allresind, key=itemgetter(1))
        # TODO: check that indexing and cell ids are consistent (it would be safer to always work with cell indices) 
        marked_res = 0
        for res in allresind:
            if marked_res >= theta_eta * global_res:
                break
            mesh_markers[res[2]].add(res[1])
            marked_res += res[0]
            
        print "RES MARKED elements:\n", [(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()]
        
        # projection marking
        theta_zeta = 0.8
        max_zeta = max([max(projind[mu].coeffs) for mu in projind.active_indices()])
        for mu, vec in projind.iteritems():
            indmu = [i for i, p in enumerate(vec.coeffs) if p >= theta_zeta * max_zeta]
            mesh_markers[mu] = mesh_markers[mu].union(set(indmu)) 
            print "PROJ MARKING", len(indmu), "elements in", mu
    
        print "FINAL MARKED elements:\n", [(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()]
    
        # create refined multi vector
        new_w = MultiVectorWithProjection()
        for mu, cell_ids in mesh_markers.iteritems():
            new_w[mu] = w[mu].refine(cell_ids, True)
        w = new_w

    
    # new multiindex activation
#    am_f, am_rv = coeff_field[0]
#    beta = am_rv.orth_polys.get_beta(1)
#    beta[0]
#    print "additional projection error indices are ", set(projind.active_indices()) - set(resind.active_indices())

    
        print "REFINED MESHES elements:\n", [(mu, vec.basis.mesh.num_cells()) for mu, vec in w.iteritems()]

    # show refined meshes
    plot_meshes = True    
    if plot_meshes:
        for mu, vec in new_w.iteritems():
            plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
        interactive()

test_main()
