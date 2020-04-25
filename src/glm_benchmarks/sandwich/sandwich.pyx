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

# cdef extern from "sparse.c":
#     void _sparse_sandwich(double*, int*, int*, double*, int*, int*, double*, double*, int, int, int) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich(A, AT, double[:] d):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef double[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    cdef double[:] ATdata = AT.data
    cdef int[:] ATindices = AT.indices
    cdef int[:] ATindptr = AT.indptr

    cdef double* Adatap = &Adata[0]
    cdef int* Aindicesp = &Aindices[0]
    cdef double* ATdatap = &ATdata[0]
    cdef int* ATindicesp = &ATindices[0]
    cdef int* ATindptrp = &ATindptr[0]

    cdef double* dp = &d[0]

    cdef int m = Aindptr.shape[0] - 1
    cdef int n = d.shape[0]
    cdef int nnz = Adata.shape[0]
    out = np.zeros((m,m))
    cdef double[:, :] out_view = out
    cdef double* outp = &out_view[0,0]

    cdef int AT_idx, A_idx
    cdef int AT_row, A_col
    cdef int i, j, k
    cdef double A_val, AT_val

    for j in prange(m, nogil=True):
        for A_idx in range(Aindptr[j], Aindptr[j+1]):
            k = Aindicesp[A_idx]
            A_val = Adatap[A_idx] * dp[k]
            for AT_idx in range(ATindptrp[k], ATindptrp[k+1]):
                i = ATindicesp[AT_idx]
                if i > j:
                    break
                AT_val = ATdatap[AT_idx]
                outp[j * m + i] = outp[j * m + i] + AT_val * A_val

    out += np.tril(out, -1).T
    return out


cdef extern from "dense.c":
    void _dense_sandwich(double*, double*, double*, int, int) nogil

def dense_sandwich(double[:,:] X, double[:] d):
    cdef int m = X.shape[1]
    cdef int n = X.shape[0]

    out = np.zeros((m,m))
    cdef double[:, :] out_view = out
    cdef double* outp = &out_view[0,0]

    cdef double* Xp = &X[0,0]
    cdef double* dp = &d[0]
    _dense_sandwich(Xp, dp, outp, m, n)
    out += np.tril(out, -1).T
    return out

#  OLD
#                            time
# n_rows  method
# 10000   fast_sandwich  0.001677
# 100000  fast_sandwich  0.010822
# 300000  fast_sandwich  0.021473
# 1000000 fast_sandwich  0.077476
# 2000000 fast_sandwich  0.159175
# 4000000 fast_sandwich  0.320003
# 
# NEW
# 10000    fast_sandwich  0.000329
# 100000   fast_sandwich  0.002943
# 300000   fast_sandwich  0.011041
# 1000000  fast_sandwich  0.042850
# 2000000  fast_sandwich  0.088717
# 4000000  fast_sandwich  0.190790
# 10000000 fast_sandwich  0.463217
