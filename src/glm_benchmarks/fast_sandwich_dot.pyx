# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import cython
from cython.parallel import parallel, prange
import numpy as np

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def sandwich_cythonF(double[:,:] X, double[:] d, double[:, :] res):
# 
#     cdef int nrows = X.shape[0]
#     cdef int ncols = X.shape[1]
# 
#     cdef int entry_idx, i, j, k, block
#     cdef int nblocks = 100
#     cdef int blocksize = nrows / nblocks
#     cdef int blockstart
#     cdef int nentries = ncols * (ncols + 1) / 2
#     for entry_idx in prange(nentries, nogil=True):
#         i = int(floor(-0.5 + sqrt(0.25 + 2 * entry_idx)))
#         j = (entry_idx - i * (i + 1) / 2)
#         for block in range(nblocks):
#             blockstart = block * blocksize
#             for k in range(blockstart, blockstart + blocksize):
#                 res[i,j] += X[k,i] * d[k] * X[k,j]
# 
#     for i in range(ncols):
#         for j in range(0, i):
#             res[i,j] = res[j,i]

@cython.boundscheck(False)
@cython.wraparound(False)
def sandwich_cythonF(double[:,:] X, double[:] d, double[:, :] res):

    cdef int nrows = X.shape[0]
    cdef int ncols = X.shape[1]

    cdef int i, j, k, block
    cdef int nblocks = 100
    cdef int blocksize = nrows / nblocks
    cdef int blockstart

    cdef double[:,:] XT = np.empty((ncols, nrows))
    for i in range(nrows):
        for j in range(ncols):
            XT[j,i] = X[i,j] * d[i]

    # for i in prange(ncols, nogil=True):
    for i in range(ncols):
        for block in range(nblocks):
            blockstart = block * blocksize
            for j in range(i, ncols):
                for k in range(blockstart, blockstart + blocksize):
                    res[i,j] += XT[i,k] * X[k,j]

    for i in range(ncols):
        for j in range(0, i):
            res[i,j] = res[j,i]

def fast_sandwich_dot(X, d):
    import time
    start = time.time()
    print(time.time() - start)
    out = np.zeros((X.shape[1], X.shape[1]))

    X1 = np.sqrt(d)[:,None] * X
    from scipy.linalg.blas import dsyrk
    dsyrk(alpha=1.0, a=X1, c=out)
    # fnc = sandwich_cythonF
    # # fnc = sandwich_cythonF if X.flags["F_CONTIGUOUS"] else sandwich_cythonC
    # fnc(X, d, out)
    # out = (X.T * d) @ X
    print('sandwich dot took', time.time() - start)
    return out
