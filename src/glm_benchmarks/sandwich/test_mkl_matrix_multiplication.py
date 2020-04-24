# from sparse_matrix_multiplications import mkl_csr_matvec
import numpy as np
import scipy as sp
import scipy.sparse

import glm_benchmarks.spblas.mkl_spblas as mkl_spblas

atol = 10e-12
rtol = 10e-12


def test_matvec():

    A = simulate_matrix(seed=0)
    x = np.random.randn(A.shape[1])
    Ax_bench = A.dot(x)

    # For CSR format.
    #    assert np.allclose(
    #        mkl_csr_matvec(A, x), Ax_bench, atol=atol, rtol=rtol
    #    )
    assert np.allclose(mkl_spblas.mkl_matvec(A, x), Ax_bench, atol=atol, rtol=rtol)
    # For CSC format.
    assert np.allclose(
        mkl_spblas.mkl_matvec(A.tocsc(), x), Ax_bench, atol=atol, rtol=rtol
    )


def test_transpose_matvec():

    A = simulate_matrix(seed=1)
    x = np.random.randn(A.shape[0])
    ATx_bench = A.T.dot(x)

    # For CSR format.
    #    assert np.allclose(
    #        mkl_csr_matvec(A, x, transpose=True), ATx_bench, atol=atol, rtol=rtol
    #    )
    assert np.allclose(
        mkl_spblas.mkl_matvec(A, x, transpose=True), ATx_bench, atol=atol, rtol=rtol
    )
    # For CSC format.
    assert np.allclose(
        mkl_spblas.mkl_matvec(A.tocsc(), x, transpose=True),
        ATx_bench,
        atol=atol,
        rtol=rtol,
    )


def test_matmat():

    shape = np.array([1000, 500])
    B = simulate_matrix(shape=shape, seed=0)
    A = simulate_matrix(shape=np.flip(shape), seed=1)
    AB_bench = A.dot(B).toarray()

    for return_dense in (False, True):
        AB = mkl_spblas.mkl_matmat(A, B, return_dense=return_dense)
        if not return_dense:
            AB = AB.toarray()
        assert np.allclose(AB, AB_bench, atol=atol, rtol=rtol)

    A = A.tocsc()
    B = B.tocsc()
    for return_dense in (False, True):
        AB = mkl_spblas.mkl_matmat(A, B, return_dense=return_dense)
        if not return_dense:
            AB = AB.toarray()
        assert np.allclose(AB, AB_bench, atol=atol, rtol=rtol)


def test_transpose_matmat():

    shape = np.array([1000, 500])
    B = simulate_matrix(shape=shape, seed=0)
    A = simulate_matrix(shape=shape, seed=1)
    ATB_bench = A.T.dot(B).toarray()

    for return_dense in (False, True):
        ATB = mkl_spblas.mkl_matmat(A, B, transpose=True, return_dense=return_dense)
        if not return_dense:
            ATB = ATB.toarray()
        assert np.allclose(ATB, ATB_bench, atol=atol, rtol=rtol)

    A = A.tocsc()
    B = B.tocsc()
    for return_dense in (False, True):
        ATB = mkl_spblas.mkl_matmat(A, B, transpose=True, return_dense=return_dense)
        if not return_dense:
            ATB = ATB.toarray()
        assert np.allclose(ATB, ATB_bench, atol=atol, rtol=rtol)


def test_fast_matmul2():

    shape = np.array([100, 50])
    A = simulate_matrix(shape=shape, seed=0).tocsc()
    d = np.ones(shape[0])
    true = A.T.dot(A).toarray()
    AT = A.T.tocsc()

    out = mkl_spblas.fast_matmul2(
        A.data, A.indices, A.indptr, AT.data, AT.indices, AT.indptr, d
    )
    np.testing.assert_almost_equal(true, out)


def simulate_matrix(nonzero_frac=0.05, shape=[1000, 500], seed=0):

    np.random.seed(seed)
    nnz = int(np.prod(shape) * nonzero_frac)
    row_index = np.random.randint(shape[0], size=nnz)
    col_index = np.random.randint(shape[1], size=nnz)
    A = sp.sparse.csr_matrix((np.random.randn(nnz), (row_index, col_index)), shape)
    return A
