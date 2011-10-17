.. meta::
   :http-equiv=xrefresh: 5


======================================================================
 Concept for the implementation of the residual based error estimator
======================================================================

In this document all necessary formulas for the implementation of the
residual-based a posteriori error estimator for the SGFEM method
proposed by C. Gittelson and Ch. Schwab shall be collected.

Basic guidelines for this document

* all necessary formulas shall be collected, such that a complete
  workflow could be implemented from it
* math formulas should all be supplemented by code sketches
* the prose need not be nice, better complete


Model setting
=============

Elliptic boundary value problem

.. math::
   :label: elliptic_spde

   -\nabla(a\cdot\nabla u)=f \mathrm{\ in\ } D

Homogeneous Dirichlet boundary conditions :math:`u=0` on
:math:`D`

:math:`D` is a Lipschitz domain. For the application we will restrict
this to :math:`[0,1]^2`.
Definition in |fenics|_ will look like::

  # create mesh and define function space
  mesh = UnitSquare(1, 1)
  V = FunctionSpace(mesh, 'CG', 1)
  

Coefficient
-----------

The coefficient has the form

.. math:: a(y,x) = \bar{a}(x) + \sum_{m=1}^M y_m a_m(x)

.. note:: How do we define the functions :math:`a_m(x)` on the
   FunctionSpace object ``V`` in |fenics|_ how to we embed
   this into a spuq Basis object?

What kind of functions shall we take for the :math:`a_m` first?
Piecewise? Trigonometric? What about :math:`\bar{a}`? Constant?

Specification of the random variables :math:`y_m`. Need to be defined
on :math:`[-1,1]` with :math:`\#supp(y_m)=\infty`. Take simply uniform
:math:`U(-1,1)`? Needs to be symmetric?

Operator
--------

Continuous operator
~~~~~~~~~~~~~~~~~~~


.. math:: \bar{A} = -\nabla  \bar{a}(x) \cdot\nabla

.. math:: \bar{\mathcal{A}} = \mathrm{Id} \otimes \bar{A}

.. math:: A_m = -\nabla  a_m(x) \cdot\nabla

.. math:: K_m = r(y) \mapsto r(y) y_m

.. math:: \mathcal{A}_m = K_m \otimes A_m

.. math:: 
  :label: continuous_operator

  \mathcal{A} = \bar{\mathcal{A}} + \sum_{m=1}^\infty
  \mathcal{A}_m

Discrete operator
~~~~~~~~~~~~~~~~~

For each :math:`\mu` we have some :math:`V_{\mu}`

solution has the form 

.. math:: w(y,x) = \sum_{\mu\in\Lambda} w_{\mu}(x) P_{\mu}(y)

ordered basis :math:`B_{\mu}=\{b_{\mu,i}\}` of :math:`V_{\mu}`

.. math:: w(y,x) = \sum_{\mu\in\Lambda} \sum_{i=1}^{\#B_{\mu}}
   w_{\mu,i} b_{\mu,i}(x) P_{\mu}(y)

for each vector :math:`w_{\mu}=[\dots w_{\mu,i} \dots]^T` of length
:math:`\#B_{\mu}` we discretise equation :eq:`continuous_operator`.


Evaluation of the discrete operator :math:`A_m`::

  A = MultiOperator( a, rvs )
  P = PDE()
  def MultiOperator.apply( w ):
      beta = rvs.get_orthogonal_poly().monic_coeffs
      v = MultiVector()
      delta = w.active_set()
      for mu in delta:
          A0 = P.assemble( a[0], w[mu].basis )
          v[mu] = A0 * w[mu] 
          for m in xrange(1,100):
              Am = P.assemble( a[m], w[mu].basis )
              mu1 = mu.add( (m,1) )
              if mu1 in Delta:
                  v[mu] += Am * beta(m, mu[m] + 1) * w[mu1].basis.project(w[mu].basis.mesh,INTERPOLATE)
              mu2 = mu.add( (m,-1) )
              if mu2 in Delta:
                  v[mu] += Am * beta(m, mu[m]) * w[mu2].basis.project(w[mu].basis.mesh,INTERPOLATE)
      return v



Algorithms
==========
   

Solve algorithm
---------------

Solve algorithm::

   def solve( eps, w0, eta0, chi ):
     w=w0; eta=eta0;
     for i in xrange(1,):
       [w,zeta]=pcg( w, chi*xi )
       (eta,eta_S)=error_estimator( w, zeta )
       if eta<=eps:
         return w
       w=refine(w,eta_S)

Identification of variables: 

* ``eps`` = :math:`\epsilon`, threshold for the total estimated error  
* ``w0`` = :math:`w_N^0`, initial solution, is a collection of
  multiindices with associated vectors that include the basis used for
  this multiindex; the parameter :math:`\mathcal{V}^{1 or 0}` is
  included in ``w0``
* ``xi0`` = :math:`\xi^0` error bound of the initial solution (?),
  estimate :math:`\xi^0:=(1-\gamma)^{-1/2}\|f\|_{V^*}` (see note 3)
* ``chi`` = :math:`\chi` parameter that determines the accuracy of the
  solver; between 0 and 1 (exclusive)

.. note:: maybe we can pass :math:`\zeta^0` instead of :math:`\xi^0`
  and compute :math:`\xi^0` using the error estimator, i.e. swapping
  lines 2 and 3 of the algorithm

.. note:: why does :math:`\mathcal{V}` have a different index
   than :math:`w` in the paper; should be the same

.. note:: we rename :math:`\xi` to :math:`\eta`; further the error
  estimator returns also the local error, not only the global one


Error estimator
---------------

Definitions
~~~~~~~~~~~

The residual error estimator follows from a partial integration of the residual

.. math:: \langle r_\mu(w_N),v\rangle = \int_D f\delta_{\mu 0} - \sigma_\mu(w_N)\cdot\nabla v\;dx,\quad v\in H^1_0(\Omega),

for some given approximation :math:`w_N\in\mathcal{V}_N`.

The flux :math:`\sigma_\mu` for :math:`\mu\in\Lambda`  is defined by

.. math:: \sigma_\mu(w_N) := \bar{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla(\beta^m_{\mu_m+1}\Pi^{\mu+\epsilon_m}_\mu w_{N,\mu+\epsilon_m} + \beta^m_{\mu_m}\Pi^{\mu-\epsilon_m}_\mu w_{N,\mu-\epsilon_m}).

We have to evaluate the volume and edge contributions in elements :math:`T\in\mathcal{T}_\mu` and on edges :math:`S\in\mathcal{S}_\mu` of the error estimator,

.. math:: \eta_{\mu,T}(w_N) &:= h_T||\bar{a}^{-1/2}(f\delta_{\mu 0} + \nabla\cdot\sigma_\mu(w_N))||_{L^2(T)}\\
          \eta_{\mu,S} (w_N) &:= h_S^{1/2} ||\bar{a}^{-1/2} [\sigma_\mu(w_N)]_S ||_{L^2(S)}

These sum up to the total error estimator

.. math:: \eta_\mu(w_N) := \left( \sum_{T\in\mathcal{T}_\mu} \eta_{\mu,T}(w_N)^2 + \sum_{S\in\mathcal{S}_\mu} \eta_{\mu,S}(w_N)^2 \right)^{1/2}.


Note that for conforming piecewise affine approximations (i.e. continuous linear elements) the divergence of :math:`\sigma_\mu` simplifies to

.. math:: \nabla\cdot\sigma_\mu(w_N) = \nabla\bar{a}\cdot\nabla w_{N,\mu} + \sum_{m=1}^\infty \nabla a_m\cdot\nabla( \beta^m_{\mu_m+1}\Pi^{\mu+\epsilon_m}_\mu w_{N,\mu+\epsilon_m} + \beta^m_{\mu_m}\Pi^{\mu-\epsilon_m}_\mu w_{N,\mu-\epsilon_m} ).



Algorithm for the evaluation of :math:`\sigma_\mu`::

  w = MultiVector()
  m = MultiIndex( (...) )
  T0 = IntialMesh()
  w[m] = FenicsVector(T0)
  # sigma_mu
  # a = (Function, Function, Function, ... )
  newDelta = extend(Delta)
  for mu in newDelta:
      sigma_x = a[0]( w[mu].mesh.nodes ) * w[mu].dx() 
      for m in xrange(1,100):
          mu1 = mu.add( (m,1) )
          if mu1 in Delta:
              sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]+1) *\
                          w[mu1].project( w[mu].mesh ).dx()
          mu2 = mu.add( (m,-1) )
          if mu2 in Delta:
              sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]) *\
                          w[mu2].project( w[mu].mesh ).dx()


The function ``error_estimator``::

  def error_estimator( w, zeta, c_eta, c_Q ):
    

Projection :math:`\Pi_\mu^\nu:V_\nu\to V_\mu` for some
:math:`\mu,\nu\in\Lambda` can be an arbitrary map such as the
:math:`L^2`-projection, the :math:`\mathcal{A}`-orthogonal projection
or nodal interpolation.


The projection error is evaluated by

.. math:: \delta_\mu := \sum_{m=1}^\infty \left\Lvert\frac{a_m}{\overline{a}}\right\Rvert_{L^\infty(D)}\left\{\beta^m_{\mu+1}\left\vert \Pi^{\mu+e_m}_\mu w_{N,\mu+e_m} - w_{N,\mu+e_m} \right\vert_{H^1_0(D)} + \beta^m_{\mu_m}\left\vert \Pi^{\mu-e_m}_\mu w_{N,\mu-e_m} - w_{N,\mu-e_m} \right\vert_{H^1_0(D)} \right\}


Refinement
----------

The marking/refinement procedure is three-fold:

#. (for active indices :math:`\mu\in\Lambda`) evaluation of the residual error estimator :math:`\hat{\eta}_{\mu,S}(w_N)` and edge marking of respective FEM meshes :math:`\mathcal{T}_\mu`
#. (for active indices and their *neighbourhood*) estimation of the projection errors and marking of respective FEM meshes
#. activation of new indices based on the projection estimation of 2.


FEM residuals
~~~~~~~~~~~~~

We employ an edge-based Dörfler marking strategy for all edges :math:`S\in\mathcal{S}_\mu` with the edge indicator

.. math::
   \hat{\eta}_{\mu,S} := \left( \eta_{\mu,S}(w_N)^2 + \frac{1}{d+1} \sum_{T:\ S\in\mathcal{S}_\mu \cap \partial T} \eta_{\mu,T}(w_N)^2 \right)^{1/2}

such that, for some parameter :math:`0<\vartheta_\eta<1`, a set

.. math::
   \hat{\mathcal{S}}_\mu \subset \bigcup_{\mu\in\Lambda} \{\mu\}\times\mathcal{S}_\mu

of small cardinality is obtained for which holds

.. math::
   \sum_{(\mu,S)\in\hat{\mathcal{S}}_\mu} \hat{\eta}_{\mu,S}^2 \geq \vartheta_\eta^2 \sum_{\mu\in\Lambda} \eta_\mu(w_N)^2.

Let :math:`\mathcal{T}_N:=\bigcup_{\mu\in\Lambda}\{\mu\}\times\mathcal{T}_\mu` encode the set of all elements of all meshes paired with the respective multiindex :math:`\mu`, i.e. for all element :math:`T\in\mathcal{T}_\mu` for any :math:`\mu\in\Lambda` there is a tuple :math:`(\mu,T)\in\mathcal{T}_N`.

Let :math:`\mathcal{T}_\eta\subset\mathcal{T}_N` be the subset of elements which have at least one edge in :math:`\hat{\mathcal{S}}_\mu` and mark these elements for refinement.


Projection errors
~~~~~~~~~~~~~~~~~

TODO


Activation of new indices
~~~~~~~~~~~~~~~~~~~~~~~~~

TODO



PCG
---

This should be implemented as a standard preconditioned conjugate
gradient solver, where the special treatment necessary for the
specific structure of :math:`w_N` is hidden in a generalised vector
class (``FEMVector``) that takes care of that.

Meaning of the variables

* :math:`\rho` = ``r`` residual
* :math:`s` = ``s`` preconditioned residual
* :math:`v` = ``v`` search direction
* :math:`w` = ``w`` solution
* :math:`\zeta` is the enery norm (w.r.t. :math:`\bar{\mathcal{A}}`)
  of the preconditioned residual :math:`s`,
  i.e. :math:`\|s\|^2_{\bar{\mathcal{A}}}`

Algorithm::

  def pcg( A, A_bar, w0, eps ):
    # use forgetful_vector for vectors 
    w[0] = w0
    r[0] = f - apply(A, w[0])
    v[0] = solve(A_bar, r[0])
    zeta[0] = r[0].inner(s[0])
    for i in count(1):
      if zeta[i-1] <= eps**2:
        return (w[i-1], zeta[i-1])
      z[i-1] = apply(A, v[i-1])
      alpha[i-1] = z[i-1].inner(v[i-1])
      w[i] = w[i-1] + zeta[i-1] / alpha[i-1] * v[i-1]
      r[i] = r[i-1] - zeta[i-1] / alpha[i-1] * z[i-1]
      s[i] = solve(A_bar, r[i])
      zeta[i] = r[i].inner(s[i])
      v[i] = s[i] - zeta[i] / zeta[i-1] * v[i-1]

Data structures
===============

Vectors
-------

Sketch for the generalised vector class for ``w`` which we call ``MultiVector``::

  class MultiVector(object):
    #map multiindex to Vector (=coefficients + basis)
    def __init__(self):
      self.mi2vec = dict()
    
    def extend( self, mi, vec ):
      self.mi2vec[mi] = vec
    
    def active_indices( self ):
      return self.mi2vec.keys()
    
    def get_vector( self, mi ):
      return self.mi2vec[mi]
    
    def __add__(self, other):
      assert self.active_indices() == other.active_indices()
      newvec = FooVector()
      for mi in self.active_indices():
        newvec.extend( mi, self.get_vector(mi)+other.get_vector(mi))
      return newvec
    
    def __mul__():
      pass
        
    def __sub__():
      pass

The ``MultiVector`` needs a set of *normal* vectors which represent
a solution on a single FEM mesh::

  class FEMVector(FullVector):
    INTERPOLATE = "interpolate"

    def __init__(self, coeff, basis ):
      assert isinstance( basis, FEMBasis )
      self.FullVector.__init__(coeff, basis)
      
    def project(self, basis, type=FEMVector.INTERPOLATE):
      assert isinstance( basis, FEMBasis )
      newcoeff = FEMBasis.project( self.coeff, self.basis, basis, type )
      return FEMVector( newcoeff, basis )

The ``FEMVector``s need a basis which should be fixed to a
``FEMBasis`` and derivatives (which could be a |fenics| or dolfin basis
or whatever FEM software is underlying it)::

  class FEMBasis(FunctionBasis):
    def __init__(self, mesh):
      self.mesh = mesh
      
    def refine(self, faces):
      (newmesh, prolongate, restrict)=self.mesh.refine( faces )
      newbasis = FEMBasis( newmesh )
      prolop = Operator( prolongate, self, newbasis )
      restop = Operator( restrict, newbasis, self )
      return (newbasis, prolop, restop)
      
    @override
    def evaluate(self, x):
      # pass to dolfin 
      pass
      
    @classmethod
    def project( coeff, oldbasis, newbasis, type ):
      # let dolfin do the transfer accoring to type
      pass      

The FEMBasis needs a mesh class for refinement and transfer of
solutions from one mesh to another. This mesh shall have derived class
that encapsulat specific Mesh classes (that come e.g. from Dolfin) ::

  # in spuq.fem?
  class FEMMesh( object ):
    def refine( self, faces ):
      return NotImplemented

  # in spuq.adaptors.fenics
  class FenicsMesh( FEMMesh ):
    def __init__(self):
      from dolfin import Mesh
      self.fenics_mesh = Mesh()

    def refine( self, faces ):
      new_fenics_mesh = self.fenics_mesh.refine(faces)
      prolongate = lambda x: fenics.project( x, fenics_mesh,
                                             new_fenics_mesh ) 
      restrict = lambda x: fenics.project( x, new_fenics_mesh, 
                                           fenics_mesh )
      return (Mesh( new_fenics_mesh ), prolongate, restrict)

Refinement::

  b0 = FEMBasis( FEniCSMesh() )
  coeffs = whatever()
  v0 = FEMVector( coeffs, b0 )
  faces = marking_strategy( foo )
  (b1, prol, rest) = b0.refine( faces )
  v1 = prol( v0 )
  assert v1.get_basis() == b1
  assert v1.__class__ == v2.__class__

.. note: The |fenics| specific stuff should go into a specific package
         e.g. spuq.fenics or spuq.adaptors.fenics so that we can also
         use other FEM packages if we want 

Questions
=========

* What kind of requirements are there for the 
  projectors :math:`\Pi_\mu^\nu`?


.. |fenics| replace:: FEniCS
.. _fenics: http://fenicsproject.org/


