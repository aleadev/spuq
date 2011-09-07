.. meta::
   :http-equiv=refresh: 5


======================================================================
 Concept for the implementation of the residual based error estimator
======================================================================

In this document all necessary formulas for the implementation of the
residual-based a posteriori error estimator for the SGFEM method
proposed by C. Gittelson and Ch. Schwab shall be collected.

Basic guidelines for this document

* all necessary formulas shall be collected, such that a complete
  workflow could be implemented from it
* prose need not be nice


Model setting
=============

Elliptic boundary value problem

.. math::
   :label: elliptic_spde

   -\nabla(a\cdot\nabla u)=f \mathrm{\ in\ } D

Homogeneous Dirichlet boundary conditions :math:`u=0` on
:math:`D`

:math:`D` is a Lipschitz domain. For the application we will restrict
this to a rectangular domain, maybe simply :math:`[0,1]^2`, ok?
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



Refinement
----------



PCG
---

This should be implemented as a standard preconditioned conjugate
gradient solver, where the special treatment necessary for the
peculiar structure of :math:`w_N` is hidden in a generalised vector
class that takes care of that.


Questions
=========

* Is :math:`\Lambda` adaptively enlarged? Probably yes; we let it
  denote the set of *active* multiindices.
* What kind of requirements are there for the 
  projectors :math:`\Pi_\mu^\nu`?


.. |fenics| replace:: FEniCS
.. _fenics: http://fenicsproject.org/


