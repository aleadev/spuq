from __future__ import division
from math import ceil
from operator import itemgetter
from collections import defaultdict
from itertools import count

from spuq.application.egsz.multi_vector import MultiVectorWithProjection
from spuq.application.egsz.coefficient_field import CoefficientField
from spuq.math_utils.multiindex import Multiindex
from spuq.stochastics.random_variable import NormalRV, UniformRV
from spuq.utils.testing import assert_equal, assert_almost_equal, skip_if, test_main, assert_raises

try:
    from dolfin import (Expression, Function, FunctionSpace, UnitSquare, interpolate, Constant, MeshFunction,
                            Mesh, FiniteElement, cells, refine, plot, interactive, norm, solve)
    import ufl
    from spuq.application.egsz.residual_estimator import ResidualEstimator
    from spuq.application.egsz.fem_discretisation import FEMPoisson
    from spuq.fem.fenics.fenics_basis import FEniCSBasis
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
    a = [Expression('2.+sin(20.*pi*I*x[0]*x[1])', I=i, degree=3, element=fs.ufl_element())
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
    # define source term
    f = Constant("1.0")
#    f = Expression("10.*exp(-(pow(x[0] - 0.6, 2) + pow(x[1] - 0.4, 2)) / 0.02)", degree=3)

    # set default vector for new indices
    mesh0 = refine(Mesh('lshape.xml'))
    fs0 = FunctionSpace(mesh0, "CG", 1)
    B = FEniCSBasis(fs0)
    u0 = Function(fs0)
    diffcoeff = Constant("1.0") 
    fem_A = FEMPoisson.assemble_lhs(diffcoeff, B)
    fem_b = FEMPoisson.assemble_rhs(f, B)
    solve(fem_A, u0.vector(), fem_b)
    vec0 = FEniCSVector(u0)

    # setup solution multi vector
    mis = [Multiindex([0]),
           Multiindex([1]),
           Multiindex([0, 1]),
           Multiindex([0, 2])]
    N = len(mis)

#    meshes = [UnitSquare(i + 3, 3 + N - i) for i in range(N)]
    meshes = [refine(Mesh('lshape.xml')) for _ in range(N)]
    fss = [FunctionSpace(mesh, "CG", 1) for mesh in meshes]

    # solve Poisson problem
    w = MultiVectorWithProjection()
    for i, mi in enumerate(mis):
        B = FEniCSBasis(fss[i])
        u = Function(fss[i])
        fem_A = FEMPoisson.assemble_lhs(diffcoeff, B)
        fem_b = FEMPoisson.assemble_rhs(f, B)
        solve(fem_A, u.vector(), fem_b)
        w[mi] = FEniCSVector(u)
#        plot(w[mi]._fefunc)

    # define coefficient field
    aN = 15
#    a = [Expression('2.+sin(2.*pi*I*x[0]+x[1]) + 10.*exp(-pow(I*(x[0] - 0.6)*(x[1] - 0.3), 2) / 0.02)', I=i, degree=3,
    a = [Expression('2./I+sin(2.*pi*I*x[0]+x[1])/I', I=i, degree=3,
                        element=FiniteElement('Lagrange', ufl.triangle, 1)) for i in range(1, aN)]
    rvs = [NormalRV(mu=0.5) for _ in range(1, aN - 1)]
    coeff_field = CoefficientField(a, rvs)

    # refinement loop
    # ===============
    refinements = 5

    for refinement in range(refinements):
        print "*****************************"
        print "REFINEMENT LOOP iteration ", refinement + 1
        print "*****************************"
        
        # evaluate residual and projection error estimates
        # ================================================
        maxh = 1 / 10
        resind, reserr = ResidualEstimator.evaluateResidualEstimator(w, coeff_field, f)
        projind = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh)
    
        # testing -->
        projglobal = ResidualEstimator.evaluateProjectionError(w, coeff_field, maxh, local=False)
        for mu, val in projglobal.iteritems():
            print "GLOBAL Projection Error for", mu, "=", val
        # <-- testing
    
        # ==============
        # MARK algorithm
        # ==============
    
        # setup marking sets
        mesh_markers = defaultdict(set)
    
        # residual marking
        # ================
        theta_eta = 0.8
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
        # ==================
        theta_zeta = 0.8
        min_zeta = 1e-10
        max_zeta = max([max(projind[mu].coeffs) for mu in projind.active_indices()])
        print "max_zeta =", max_zeta
        if max_zeta >= min_zeta:
            for mu, vec in projind.iteritems():
                indmu = [i for i, p in enumerate(vec.coeffs) if p >= theta_zeta * max_zeta]
                mesh_markers[mu] = mesh_markers[mu].union(set(indmu)) 
                print "PROJ MARKING", len(indmu), "elements in", mu
        
            print "FINAL MARKED elements:\n", [(mu, len(cell_ids)) for mu, cell_ids in mesh_markers.iteritems()]
        else:
            print "NO PROJECTION MARKING due to very small projection error!"
    
        # new multiindex activation
        # =========================
        # determine possible new indices
        theta_delta = 0.9
        maxm = 10
        a0_f, _ = coeff_field[0]
        Ldelta = {}
        Delta = w.active_indices()     
        deltaN = int(ceil(0.1 * len(Delta)))               # max number new multiindices
        for mu in Delta:
            norm_w = norm(w[mu].coeffs, 'L2')
            for m in count(1):
                mu1 = mu.inc(m)
                if mu1 not in Delta:
                    if m > maxm or m >= len(coeff_field):  # or len(Ldelta) >= deltaN
                        break 
                    am_f, am_rv = coeff_field[m]
                    beta = am_rv.orth_polys.get_beta(1)
                    # determine ||a_m/\overline{a}||_{L\infty(D)} (approximately)
                    f = Function(w[mu]._fefunc.function_space())
                    f.interpolate(a0_f)
                    min_a0 = min(f.vector().array())
                    f.interpolate(am_f)
                    max_am = max(f.vector().array())
                    ainfty = max_am / min_a0
                    assert isinstance(ainfty, float)
                    
#                    print "A***", beta[1], ainfty, norm_w
#                    print "B***", beta[1] * ainfty * norm_w
#                    print "C***", theta_delta, max_zeta
#                    print "D***", theta_delta * max_zeta
#                    print "E***", bool(beta[1] * ainfty * norm_w >= theta_delta * max_zeta)
                    
                    if beta[1] * ainfty * norm_w >= theta_delta * max_zeta:
                        val1 = beta[1] * ainfty * norm_w
                        if mu1 not in Ldelta.keys() or (mu1 in Ldelta.keys() and Ldelta[mu1] < val1):
                            Ldelta[mu1] = val1
                    
        print "POSSIBLE NEW MULTIINDICES ", sorted(Ldelta.iteritems(), key=itemgetter(1), reverse=True)
        Ldelta = sorted(Ldelta.iteritems(), key=itemgetter(1), reverse=True)[:min(len(Ldelta), deltaN)]
        # add new multiindices to solution vector
        for mu, _ in Ldelta:
            w[mu] = vec0
        print "SELECTED NEW MULTIINDICES ", Ldelta
    
        # create new refined (and enlarged) multi vector
        # ==============================================
        for mu, cell_ids in mesh_markers.iteritems():
            vec = w[mu].refine(cell_ids, with_prolongation=False)
            fs = vec._fefunc.function_space()
            B = FEniCSBasis(fs)
            u = Function(fs)
            fem_A = FEMPoisson.assemble_lhs(diffcoeff, B)
            fem_b = FEMPoisson.assemble_rhs(f, B)
            solve(fem_A, vec.coeffs, fem_b)
            w[mu] = vec

    # show refined meshes
    plot_meshes = True    
    if plot_meshes:
        for mu, vec in w.iteritems():
            plot(vec.basis.mesh, title=str(mu), interactive=False, axes=True)
            plot(vec._fefunc)
        interactive()

test_main()
