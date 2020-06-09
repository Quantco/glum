import numpy as np
import pytest
import scipy as sp
import scipy.sparse

from quantcore.glm.matrix.sandwich.sandwich import dense_sandwich, sparse_sandwich


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fast_sandwich_sparse(dtype):
    A = simulate_matrix(dtype=dtype).tocsc()

    d = np.ones(A.shape[0], dtype=dtype)
    true = A.T.dot(A).toarray()
    AT = A.T.tocsc()

    out = sparse_sandwich(A, AT, d)
    np.testing.assert_allclose(true, out, atol=np.sqrt(np.finfo(dtype).eps))


def test_fast_sandwich_dense():
    A = simulate_matrix().tocsc()
    d = np.ones(A.shape[0])
    true = A.T.dot(A).toarray()

    out2 = dense_sandwich(np.asfortranarray(A.toarray()), d)
    np.testing.assert_allclose(true, out2, atol=np.sqrt(np.finfo(np.float64).eps))


def simulate_matrix(nonzero_frac=0.05, shape=(100, 50), seed=0, dtype=np.float64):

    np.random.seed(seed)
    nnz = int(np.prod(shape) * nonzero_frac)
    row_index = np.random.randint(shape[0], size=nnz)
    col_index = np.random.randint(shape[1], size=nnz)
    A = sp.sparse.csr_matrix(
        (np.random.randn(nnz).astype(dtype), (row_index, col_index)), shape
    )
    return A
