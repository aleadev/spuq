from numpy.random import rand, randn

from spuq.linalg.vector import *
from spuq.linalg.operator import *

A = MatrixOperator( 1 + rand(3, 5) )
B = MatrixOperator( 1 + rand(7, 3) )
print A.domain.dim, A.codomain.dim
print B.domain.dim, B.codomain.dim

x = FlatVector( rand( 5,1 ) )
print x

# operators can be multiplied
C = B * A
print C.domain.dim, C.codomain.dim

# operator composition can be performed in a number of ways
print B(A(x))
print (B * A)(x)

print B * A * x
print B * (A * x)
print (B * A) * x

# similar as above, only as matrices
print (B*A).as_matrix() * x.as_vector()
print B.as_matrix() * (A.as_matrix() * x.as_vector())

# you can transpose (composed) operators
AT=A.transpose()
BT=B.transpose()
CT=C.transpose()

y = FullVector( rand( CT.domain_dim(),1 ) )
print CT*y
print AT*(BT*y)

# can add and subtract operators
print (B * (A+A))*x
print C*x+C*x
print (C-C)*x
print C*x-C*x

# you can pre- and post-multiply vectors with scalars
print 3*x-x*3

# you can multiply operators with scalars or vectors with scalars
print (3*C)*x
print (C*3)*x
print 3*(C*x)

