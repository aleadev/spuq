import numpy as np
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.sgfem.hermite_triple import evaluate_Hermite_triple
from matplotlib.pyplot import figure, show, spy

I = MultiindexSet.createCompleteOrderSet(4, 3, reversed=True).arr
J = MultiindexSet.createCompleteOrderSet(4, 2, reversed=True).arr

T = evaluate_Hermite_triple(I, I, J)

fig = figure()
spy(np.sum(T, axis=2))
show()
