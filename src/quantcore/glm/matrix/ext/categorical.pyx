import numpy as np
cimport numpy as np

from cython cimport floating
cimport cython
from cython.parallel import prange
ctypedef np.uint8_t uint8

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def sandwich_categorical(int[:] indices, int[:] indptr, floating[:] d, rows, cols, dtype):
    """
    Returns a 1d array. The sandwich output is a diagonal matrix with this array on
    the diagonal.

    If X is N x K, indices has length N, and indptr has length K + 1. d should have
    length N.
    """
    # Numpy: tmp = d[indices]
    cdef Py_ssize_t n_rows = len(indices)
    cdef Py_ssize_t n_cols = len(indptr) - 1
    cdef Py_ssize_t k, i, j, ki, ii
    cdef floating[:] tmp = np.empty(n_rows, dtype=dtype)
    cdef floating[:] res
    cdef floating val
    cdef int[:] rowsview
    cdef uint8[:] row_included
    cdef int[:] colsview
    cdef int n_active_rows, n_active_cols

    # tmp = d[indices]
    if rows is None:
        for k in prange(n_rows, nogil=True):
           tmp[k] = d[indices[k]]
    else:
        rowsview = rows
        n_active_rows = rows.shape[0]
        row_included = np.zeros(n_rows, dtype=np.uint8)
        for ki in prange(n_active_rows, nogil=True):
            k = rowsview[ki]
            row_included[k] = True
            tmp[k] = d[indices[k]]

    #TODO: Use this as an experiment for figuring out how to reduce the duplicate code here.
    # Alternatively, we can just have cols and rows be full lists and not make
    # this optimization.
    if cols is None and rows is None:
        res_out = np.empty(n_cols, dtype=dtype)
        res = res_out
        for i in prange(n_cols, nogil=True):
            val = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                val = val + tmp[j]
            res[i] = val
    elif cols is None:
        res_out = np.empty(n_cols, dtype=dtype)
        res = res_out
        for i in prange(n_cols, nogil=True):
            val = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                if not row_included[j]:
                    continue
                val = val + tmp[j]
            res[i] = val
    elif rows is None:
        colsview = cols
        n_active_cols = cols.shape[0]
        res_out = np.empty(n_active_cols, dtype=dtype)
        res = res_out
        for ii in prange(n_active_cols, nogil=True):
            i = colsview[ii]
            val = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                val = val + tmp[j]
            res[ii] = val
    else:
        colsview = cols
        n_active_cols = cols.shape[0]
        res_out = np.empty(n_active_cols, dtype=dtype)
        res = res_out
        for ii in prange(n_active_cols, nogil=True):
            i = colsview[ii]
            val = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                if not row_included[j]:
                    continue
                val = val + tmp[j]
            res[ii] = val
    return res_out
