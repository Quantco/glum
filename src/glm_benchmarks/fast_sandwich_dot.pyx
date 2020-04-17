import cython
import numpy as np


def sandwich_cython(double[:,:] x, double[:] d):
    # Calculate x.T @ x
    # (x.T @ D @ x)[i, j] = sum_k x[k, i] (D @ x)[k, j]
    # = sum_k x[k, i] sum_m D[k, m] x[m, j]
    # = sum_k x[k, i] D[k, k] x[k, j]
    n_row = x.shape[0]
    n_col = x.shape[1]
    out = np.zeros((n_col, n_col))
    for m in range(n_row):
        for i in range(n_col):
            for j in range(i + 1):
                out[i, j] += x[m, i] * x[m, j] * d[m]

    for i in range(n_col):
        for j in range(i, n_col):
            out[i, j] = out[j, i]

    return out
