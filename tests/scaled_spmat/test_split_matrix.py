import numpy as np
import pytest
import scipy.sparse as sps

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


def test_sandwich(X: np.ndarray):
    Xsplit = SplitMatrix(sps.csc_matrix(X), 0.2)
    v = np.random.rand(Xsplit.shape[0])
    y1 = Xsplit.sandwich(v)
    y2 = (X.T * v) @ X
    np.testing.assert_almost_equal(y1, y2)


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


@pytest.mark.parametrize("matrix_shape", [(4,), (4, 1), (4, 2)])
def test_dot(X: np.ndarray, matrix_shape):
    v = np.ones(matrix_shape)
    result = SplitMatrix(sps.csc_matrix(X), 0.2).dot(v)
    expected = X.dot(v)
    np.testing.assert_allclose(result, expected)


def test_dot_raises(X: np.ndarray):
    with pytest.raises(ValueError):
        SplitMatrix(sps.csc_matrix(X), 0.2).dot(np.ones((5, 1)))


@pytest.mark.parametrize("matrix_shape", [(N,), (1, N), (2, N)])
def test_r_matmul(X: np.ndarray, matrix_shape):
    v = np.ones(matrix_shape)
    result = v @ SplitMatrix(sps.csc_matrix(X), 0.2)
    expected = v @ X
    np.testing.assert_allclose(result, expected)
