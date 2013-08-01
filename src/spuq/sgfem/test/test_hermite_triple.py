import numpy as np
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.sgfem.hermite_triple import evaluate_Hermite_triple
from matplotlib.pyplot import spy

I = MultiindexSet.createCompleteOrderSet(4,2).arr
J = MultiindexSet.createCompleteOrderSet(4,3).arr

T = evaluate_Hermite_triple(I, I, J)
spy(T)
