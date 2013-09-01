import numpy as np
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.sgfem.hermite_triple import evaluate_Hermite_triple
from spuq.sgfem.legendre_triple import evaluate_Legendre_triple
from matplotlib.pyplot import figure, show, spy

I = MultiindexSet.createCompleteOrderSet(4, 3, reversed=True).arr
J = MultiindexSet.createCompleteOrderSet(4, 2, reversed=True).arr

#H = evaluate_Hermite_triple(I, I, J)
#print "shape of H:", H.shape

L = evaluate_Legendre_triple(I, I)
print len(L), L[0].shape

# fig = figure()
# spy(np.sum(H, axis=2))
fig = figure()
spy(np.sum(L))
show()
