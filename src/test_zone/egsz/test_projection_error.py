from dolfin import *

mesh1 = UnitSquare(5, 5)
mesh2 = refine(mesh1)
V1 = FunctionSpace(mesh1, 'CG', 1)
V2 = FunctionSpace(mesh2, 'CG', 1)
ex1 = Expression('sin(a*pi*x[0])+sin(b*pi*x[1])', a=1, b=1)
ex2 = Expression('a*x[0]+b*x[1]', a=1, b=1)
for ex in (ex1, ex2):
    f2 = interpolate(ex, V2)
    f21 = interpolate(f2, V1)
    f212 = interpolate(f21, V2)
    print "L2", errornorm(f2, f21, 'L2'), errornorm(f2, f212, 'L2')
    print "H1", errornorm(f2, f21, 'H1'), errornorm(f2, f212, 'H1')
