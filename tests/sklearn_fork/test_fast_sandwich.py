import numpy as np
import scipy as sp
import scipy.sparse

from glm_benchmarks.sandwich.sandwich import dense_sandwich, sparse_sandwich


def test_fast_sandwich():
    shape = np.array([100, 50])
    A = simulate_matrix(shape=shape, seed=0).tocsc()

    d = np.ones(shape[0])
    true = A.T.dot(A).toarray()
    AT = A.T.tocsc()

    out = sparse_sandwich(A, AT, d)
    out2 = dense_sandwich(np.asfortranarray(A.toarray()), d)
    np.testing.assert_almost_equal(true, out)
    np.testing.assert_almost_equal(true, out2)


def simulate_matrix(nonzero_frac=0.05, shape=[1000, 500], seed=0):

    np.random.seed(seed)
    nnz = int(np.prod(shape) * nonzero_frac)
    row_index = np.random.randint(shape[0], size=nnz)
    col_index = np.random.randint(shape[1], size=nnz)
    A = sp.sparse.csr_matrix((np.random.randn(nnz), (row_index, col_index)), shape)
    return A
