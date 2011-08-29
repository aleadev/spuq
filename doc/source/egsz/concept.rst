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
   

.. |fenics| replace:: FEniCS
.. _fenics: http://fenicsproject.org/


