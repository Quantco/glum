# distutils: extra_compile_args=-fopenmp -O3 -ffast-math -march=native -msse -msse2 -mavx
# distutils: extra_link_args=-fopenmp
import numpy as np
cimport numpy as np
import scipy as sp
import scipy.sparse
from cython cimport view
import cython
from cython.parallel import parallel, prange
from libc.math cimport ceil, sqrt
cimport openmp
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sandwich(A, AT, double[:] d):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef double[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    cdef double[:] ATdata = AT.data
    cdef int[:] ATindices = AT.indices
    cdef int[:] ATindptr = AT.indptr

    cdef int ncols = Aindptr.shape[0] - 1
    cdef int nrows = d.shape[0]
    cdef int nnz = Adata.shape[0]
    out = np.zeros((ncols,ncols))
    cdef double[:, :] out_view = out

    cdef int AT_idx, A_idx
    cdef int AT_row, A_col
    cdef int i, j, k
    cdef double A_val, AT_val

    for j in prange(ncols, nogil=True):
        for A_idx in range(Aindptr[j], Aindptr[j+1]):
            k = Aindices[A_idx]
            A_val = Adata[A_idx] * d[k]
            for AT_idx in range(ATindptr[k], ATindptr[k+1]):
                i = ATindices[AT_idx]
                if i < j:
                    continue
                AT_val = ATdata[AT_idx]
                out_view[i, j] = out_view[i, j] + AT_val * A_val

    out += np.tril(out, -1).T
    return out

cdef extern from "dense.c":
    void _dense_sandwich(double*, double*, double*, int, int) nogil

def dense_sandwich(double[:,:] X, double[:] d):
    cdef int m = X.shape[1]
    cdef int n = X.shape[0]

    out = np.zeros((m,m))
    cdef double[:, :] out_view2 = out
    cdef double* outp = &out_view2[0,0]

    cdef double* Xp = &X[0,0]
    cdef double* dp = &d[0]
    _dense_sandwich(Xp, dp, outp, m, n)
    out += np.tril(out, -1).T
    return out
