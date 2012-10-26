from dolfin import Function, UnitSquare, FunctionSpace, VectorFunctionSpace, Expression
from spuq.utils.plot.plotter import Plotter

N = 30
mesh = UnitSquare(N,N)
V = FunctionSpace(mesh,'CG',1)
f = Function(V)
ex = Expression("x[0]*x[1]")
f.interpolate(ex)

Plotter.figure()
Plotter.plotMesh(f)
Plotter.axes()
Plotter.show()

VV = VectorFunctionSpace(mesh,'CG',1)
ff = Function(VV)
exx = Expression(["x[0]*x[1]/10.","sin(2*pi*x[0])/10."])
ff.interpolate(exx)

Plotter.figure()
Plotter.plotMesh(ff, displacement=True)
Plotter.axes()
Plotter.show()
