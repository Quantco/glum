import numpy as np
from cython cimport floating
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sandwich_categorical(int[:] indices, int[:] indptr, floating[:] d):
    """
    Returns a 1d array. The sandwich output is a diagonal matrix with this array on
    the diagonal.

    If X is N x K, indices has length N, and indptr has length K + 1. d should have
    length N.
    """
    # Numpy: tmp = d[indices]
    cdef Py_ssize_t n_rows = len(indices)
    cdef Py_ssize_t n_cols = len(indptr) - 1
    cdef Py_ssize_t k, i, j
    cdef floating[:] tmp = np.empty(n_rows, dtype=float)
    cdef floating[:] res = np.empty(n_cols, dtype=float)

    # tmp = d[indices]
    for k in prange(n_rows, nogil=True):
       tmp[k] = d[indices[k]]

    cdef floating val
    for i in prange(n_cols, nogil=True):
        val = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            val = val + tmp[j]
        res[i] = val
    return res
