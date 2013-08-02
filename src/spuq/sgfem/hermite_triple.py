import numpy as np

def evaluate_Hermite_triple(I_a, I_b, I_c):
    max_ind = max([np.max(I_a), np.max(I_b), np.max(I_c)])
    triples = compute_Hermite_triples(max_ind)
    # this corresponds to algorithm multiplication_tensor_blocked_1
    na = I_a.shape[0]
    nb = I_b.shape[0]
    nc = I_c.shape[0]
    m = I_a.shape[1]
    strides = np.cumprod(triples.shape)
    # note that numpy.tile behaves differently than Matlab repmat
    # http://scipy-user.10969.n7.nabble.com/Numpy-MATLAB-difference-in-array-indexing-td999.html
    # it probably could be avoided altogether here...
    for i in range(m):
        ind = np.transpose(np.tile(I_a[:, i, np.newaxis, np.newaxis], [1, nb, nc]), [0, 1, 2])
        ind = ind + strides[1] * np.transpose(np.tile(I_b[:, i, np.newaxis, np.newaxis], [1, na, nc]), [1, 0, 2])
        ind = ind + strides[2] * np.transpose(np.tile(I_c[:, i, np.newaxis, np.newaxis], [1, na, nb]), [1, 2, 0])
        Mi = triples[ind]
        if i == 0:
            M = Mi
        else:
            M *= Mi
    M = np.reshape(M, [na, nb, nc])
    return M


# http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
def compute_Hermite_triples(p):
    from scipy.misc import factorial
    I, J, K = np.mgrid[:p + 1, :p + 1, :p + 1]
    S = I + J + K
    S2 = S / 2
    ind = np.logical_and(np.logical_and(np.mod(S, 2) == 0, I <= J + K), np.logical_and(J <= J + I, K <= I + J))
#    ind=mod(S,2)==0 & I<=J+K & J<=K+I & K<=I+J;

    M = np.zeros_like(S)
    fac = factorial(np.array(range(p + 1)))
    M[ind] = fac[I[ind]] * fac[J[ind]] * fac[K[ind]] / (fac[S2[ind] - I[ind]] * fac[S2[ind] - J[ind]] * fac[S2[ind] - K[ind]])
    return M
