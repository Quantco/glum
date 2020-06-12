import numpy as np
import pytest
import scipy as sp
import scipy.sparse

from quantcore.glm.matrix.ext.dense import dense_sandwich
from quantcore.glm.matrix.ext.sparse import sparse_sandwich


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fast_sandwich_sparse(dtype):
    np.random.seed(123)
    for i in range(10):
        nrows, ncols = np.random.randint(200, size=2)

        A = simulate_matrix(shape=(nrows, ncols), seed=None, dtype=dtype).tocsc()

        d = np.random.rand(A.shape[0]).astype(dtype)
        true = (A.T.multiply(d)).dot(A).toarray()

        out = sparse_sandwich(
            A,
            A.tocsr(),
            d,
            np.arange(A.shape[0], dtype=np.int32),
            np.arange(A.shape[1], dtype=np.int32),
        )
        np.testing.assert_allclose(true, out, atol=np.sqrt(np.finfo(dtype).eps))


def test_fast_sandwich_dense():
    for i in range(5):
        A = simulate_matrix(shape=np.random.randint(1000, size=2))
        d = np.random.rand(A.shape[0])

        d[np.random.choice(np.arange(A.shape[0]), size=10, replace=False)] = 0.0

        check(A, d, np.arange(A.shape[1], dtype=np.int32))

        cols = np.random.choice(
            np.arange(A.shape[1]), size=np.random.randint(A.shape[1]), replace=False
        ).astype(np.int32)
        check(A, d, cols)


def check(A, d, cols):
    Asub = A[:, cols]
    true = (Asub.T.multiply(d)).dot(Asub).toarray()
    nonzero = np.where(np.abs(d) > 1e-14)[0].astype(np.int32)
    out = dense_sandwich(np.asfortranarray(A.toarray()), d, nonzero, cols)
    np.testing.assert_allclose(true, out, atol=np.sqrt(np.finfo(np.float64).eps))


def simulate_matrix(nonzero_frac=0.05, shape=(100, 50), seed=0, dtype=np.float64):

    if seed is not None:
        np.random.seed(seed)
    nnz = int(np.prod(shape) * nonzero_frac)
    row_index = np.random.randint(shape[0], size=nnz)
    col_index = np.random.randint(shape[1], size=nnz)
    A = sp.sparse.csr_matrix(
        (np.random.randn(nnz).astype(dtype), (row_index, col_index)), shape
    )
    return A
