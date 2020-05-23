import numpy as np
import pytest
import scipy.sparse as sps

from glm_benchmarks.matrix.sandwich.sandwich import csr_dense_sandwich
from glm_benchmarks.matrix.split_matrix import SplitMatrix
from glm_benchmarks.sklearn_fork._glm import DenseGLMDataMatrix

N = 100


@pytest.fixture
def X():
    X = np.zeros((N, 4))
    X[:, 0] = 1.0
    X[:10, 1] = 0.5
    X[-20:, 2] = 0.25
    X[:, 3] = 2.0
    return X


def test_split_matrix_init(X: np.ndarray):
    for T, D, S in [(0.05, 4, 0), (0.1, 3, 1), (0.2, 2, 2), (0.3, 2, 2), (1.0, 0, 4)]:
        fully_dense = SplitMatrix(sps.csc_matrix(X), T)
        assert fully_dense.dense_indices.shape[0] == D
        assert fully_dense.sparse_indices.shape[0] == S


def test_sandwich_dense(X: np.ndarray):
    np.random.seed(0)
    n, k = X.shape
    d = np.random.random((n,))
    A = sps.random(n, 2).tocsr()
    result = csr_dense_sandwich(A, X, d)
    expected = A.T.A @ np.diag(d) @ X
    np.testing.assert_allclose(result, expected)


def test_sandwich(X: np.ndarray):
    for i in range(10):
        n = np.random.randint(8, 300)
        m = np.random.randint(2, n)
        X = sps.random(n, m, density=0.2)
        v = np.random.rand(n)
        Xsplit = SplitMatrix(sps.csc_matrix(X), 0.2)
        y1 = Xsplit.sandwich(v)
        y2 = ((X.T.multiply(v)) @ X).toarray()
        np.testing.assert_allclose(y1, y2, atol=1e-12)
        maxdiff = np.max(np.abs(y1 - y2))
        assert maxdiff < 1e-12


@pytest.mark.parametrize("scale_predictors", [True, False])
def test_standardize(X: np.ndarray, scale_predictors):
    weights = np.random.rand(X.shape[0])
    weights /= weights.sum()

    X_GLM = DenseGLMDataMatrix(X.copy())
    X_GLM_standardized, means, stds = X_GLM.standardize(weights, scale_predictors)

    X_split = SplitMatrix(sps.csc_matrix(X), 0.2)
    X_split_standardized, means_split, stds_split = X_split.standardize(
        weights, scale_predictors
    )

    np.testing.assert_almost_equal(means, means_split)
    np.testing.assert_almost_equal(stds, stds_split)

    X_GLM_unstandardized = X_GLM_standardized.unstandardize(means, stds)
    np.testing.assert_almost_equal(X_GLM_unstandardized, X)
