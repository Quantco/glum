from ctypes import POINTER, byref, c_char, c_char_p, c_double, c_int, cdll

import numpy as np
import scipy as sp
import scipy.sparse

mkl = cdll.LoadLibrary("libmkl_rt.dylib")


def mkl_csr_matvec(A, x, transpose=False):
    """
    Parameters
    ----------
    A : scipy.sparse csr matrix
    x : numpy 1d array
    """

    if not sp.sparse.isspmatrix_csr(A):
        raise TypeError("The matrix must be a scipy sparse CSR matrix.")

    if x.ndim != 1:
        raise TypeError("The vector to be multiplied must be a 1d array.")

    if x.dtype.type is not np.double:
        x = x.astype(np.double, copy=True)

    # Allocate the result of the matrix-vector multiplication.
    result = np.empty(A.shape[transpose])

    # Set the parameters for simply computing A.dot(x) for a general matrix A.
    alpha = byref(c_double(1.0))
    beta = byref(c_double(0.0))
    matrix_description = c_char_p(bytes("G  C  ", "utf-8"))

    # Get pointers to the numpy arrays.
    data_ptr = A.data.ctypes.data_as(POINTER(c_double))
    indices_ptr = A.indices.ctypes.data_as(POINTER(c_int))
    indptr_begin = A.indptr[:-1].ctypes.data_as(POINTER(c_int))
    indptr_end = A.indptr[1:].ctypes.data_as(POINTER(c_int))
    x_ptr = x.ctypes.data_as(POINTER(c_double))
    result_ptr = result.ctypes.data_as(POINTER(c_double))

    transpose_flag = byref(c_char(bytes(["n", "t"][transpose], "utf-8")))
    n_row, n_col = [byref(c_int(size)) for size in A.shape]
    mkl.mkl_dcsrmv(
        transpose_flag,
        n_row,
        n_col,
        alpha,
        matrix_description,
        data_ptr,
        indices_ptr,
        indptr_begin,
        indptr_end,
        x_ptr,
        beta,
        result_ptr,
    )
    return result
