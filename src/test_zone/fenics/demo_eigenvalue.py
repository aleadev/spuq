"""
This program illustrates basic use of the SLEPc eigenvalue solver for
a standard eigenvalue problem.
"""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008.
# Modified by Marie Rognes, 2009.
#
# First added:  2007-11-28
# Last changed: 2009-10-09

# Begin demo

from dolfin import *

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting."
    exit()

if not has_slepc():
    print "DOLFIN has not been configured with SLEPc. Exiting."
    exit()

# Define mesh, function space
mesh = UnitSquare(10, 10)
#mesh = Mesh("lshape.xml")
#for _ in range(2):
#    mesh = refine(mesh)

R = list()
UR = list()
for p in range(1, 3):
    V = FunctionSpace(mesh, "CG", p)
    
    # Define basis and bilinear form
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    
    # Assemble stiffness form
    A = PETScMatrix()
    assemble(a, tensor=A)
    
    # Create eigensolver
    eigensolver = SLEPcEigenSolver(A)
#    eigensolver.parameters["solver"] = "arnoldi"
    
    # Compute all eigenvalues of A x = \lambda x
    print "Computing eigenvalues for p", p
    eigensolver.solve()
    
    # Extract largest (first) eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(10)
    R.append(r)
    
    # Initialize function and assign eigenvector
    ur = Function(V)
    ur.vector()[:] = rx 
    UR.append(ur)
        
# Plot eigenfunctions
print "Largest eigenvalue: ", R
for p, ur in enumerate(UR):
    plot(ur, title="1st EF order" + str(p + 1))
interactive()
