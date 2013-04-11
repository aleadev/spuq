from spuq.stochastics.covariance import GaussianCovariance, TransformedCovariance
from spuq.stochastics.kl import KLexpansion
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.math_utils.multiindex_set import MultiindexSet
from spuq.math_utils.multiindex import Multiindex

from dolfin import UnitSquareMesh, FunctionSpace
import numpy as np

# setup covariance
GCV = GaussianCovariance(sigma=1, a=1)

# construct discrete function space
N = 3
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'CG', 1)
basis = FEniCSBasis(V)

# evaluate KL expansion
KL = KLexpansion(GCV, basis, M=5)

# evaluate transformed covariance
# initial multiindices
mis = [Multiindex(mis) for mis in MultiindexSet.createCompleteOrderSet(3, 1)]
sigma_, mu_ = 1, 0
phi = lambda gamma: np.exp(sigma_*gamma + mu_)
TCV = TransformedCovariance(mis, phi, KL, N=3)

# evaluate KL of transfor
KL2 = KLexpansion(TCV, basis, M=5)
# get pce coefficients of KL
r = eval_pce_from_KL(mis, KL, TCV._phii)
