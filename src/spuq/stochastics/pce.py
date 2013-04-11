import numpy as np
from scipy.misc import factorial


def eval_pce_from_KL(alphas, KL, phii, target_basis=None):
    # evaluate pce coefficients from (optionally projected) KL expansion
    def binom(a):
        if a.order == 0: return 0
        return factorial(a.order) / np.prod(map(lambda x: float(factorial(x)), a.as_array))
    Balphas = [(binom(a), a) for a in alphas]
    G = KL.g(target_basis = target_basis)
    # EZ (3.68)
    r = np.array([[lambda x: A * phii[a.order] * g(x) ** a for A, a in Balphas] for g in G])
    if target_basis is None:
        return r
    else:
        c4dof = target_basis.get_dof_coordinates()
        # TODO: evaluate at all dofs
        raise NotImplementedError
    # TODO: PCE/GPC class?!
