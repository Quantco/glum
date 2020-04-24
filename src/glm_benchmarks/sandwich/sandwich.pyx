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

cdef extern from "sparse.c":
    void _sparse_sandwich(double*, int*, int*, double*, int*, int*, double*, double*, int, int, int) nogil

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

    cdef int m = Aindptr.shape[0] - 1
    cdef int n = d.shape[0]
    cdef int nnz = Adata.shape[0]
    out = np.zeros((m,m))
    cdef double[:, :] out_view = out
    cdef double* outp = &out_view[0,0]
    _sparse_sandwich(
        &Adata[0], &Aindices[0], &Aindptr[0],
        &ATdata[0], &ATindices[0], &ATindptr[0],
        &d[0], outp, m, n, nnz
    );
    out += np.triu(out, 1).T
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
#         naive          0.004520
# 100000  fast_sandwich  0.010822
#         naive          0.040298
# 300000  fast_sandwich  0.021473
#         naive          0.151977
# 1000000 fast_sandwich  0.077476
#         naive          0.550102
# 2000000 fast_sandwich  0.159175
#         naive          1.116225
# 4000000 fast_sandwich  0.320003
#         naive          2.248568
