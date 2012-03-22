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
mesh = UnitSquare(2, 2)
#mesh = Mesh("lshape.xml")
for _ in range(2):
    mesh = refine(mesh)
V1 = FunctionSpace(mesh, "CG", 2)
V2 = FunctionSpace(mesh, "CG", 2)
V3 = FunctionSpace(mesh, "CG", 1)

# Define basis and bilinear form
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
a1 = dot(grad(u1), grad(v1)) * dx
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
a2 = dot(grad(u2), grad(v2)) * dx
u3 = TrialFunction(V3)
v3 = TestFunction(V3)
a3 = dot(grad(u3), grad(v3)) * dx

# Assemble stiffness form
A1 = PETScMatrix()
assemble(a1, tensor=A1)
A2 = PETScMatrix()
assemble(a2, tensor=A2)
A3 = PETScMatrix()
assemble(a3, tensor=A3)

# Create eigensolver
eigensolver1 = SLEPcEigenSolver(A1)
eigensolver2 = SLEPcEigenSolver(A2)
eigensolver3 = SLEPcEigenSolver(A3)

# Compute all eigenvalues of A x = \lambda x
print "Computing eigenvalues 1..."
eigensolver1.solve()
print "Computing eigenvalues 2..."
eigensolver2.solve()
print "Computing eigenvalues 3..."
eigensolver3.solve()

# Extract largest (first) eigenpair
r1, c1, rx1, cx1 = eigensolver1.get_eigenpair(0)
r2, c2, rx2, cx2 = eigensolver2.get_eigenpair(0)
r3, c3, rx3, cx3 = eigensolver3.get_eigenpair(0)

print "Largest eigenvalue: ", r1, "=?=", r2, "=?=", r3

# Initialize function and assign eigenvector
ur1 = Function(V1)
ur1.vector()[:] = rx1 
ur2 = Function(V2)
ur2.vector()[:] = rx2 
ur3 = Function(V3)
ur3.vector()[:] = rx3 

# Plot eigenfunction
plot(ur1, title="1st EF order 2")
plot(ur2, title="1st EF order 2")
plot(ur3, title="1st EF order 1")
interactive()
