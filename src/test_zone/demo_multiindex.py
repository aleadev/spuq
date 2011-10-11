from spuq.utils.multiindex_set import *
from spuq.stoch.random_variable import *
from spuq.stoch.stochastic_basis import *
from spuq.utils.plot.plotter import Plotter

I = MultiindexSet.createCompleteOrderSet(2, 4)
print I

N = NormalRV()
U = UniformRV()
print N, U



gpc1 = GPCBasis(I, [N, U])
gpc2 = GPCBasis(I, [N, N])
print gpc1.sample(3)
s1 = gpc1.sample(100)
s2 = gpc2.sample(100)
Plotter.figure(1)
Plotter.scatterplot(I.arr[:7, :], s1)
Plotter.figure(2)
Plotter.scatterplot(I.arr[8:15, :], s2)
Plotter.figure(3)
Plotter.histplot(s1[:6, :], bins=50)
