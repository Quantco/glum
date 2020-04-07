import numpy as np
import pytest
from scipy import sparse as sps
from scipy.sparse.linalg import lsqr

from .glmnet_qc import GlmnetGaussianModel, fit_glmnet, fit_pathwise, get_r2

n_rows = 10
n_cols = 5


@pytest.fixture
def y() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, n_rows)


@pytest.fixture
def x() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, (n_rows, n_cols))


@pytest.fixture
def x_sparse() -> sps.spmatrix:
    np.random.seed(0)
    mat = sps.random(n_rows, n_cols, density=0.2)
    all_zero_cols = np.where((mat != 0).sum(0) == 0)[0]
    if len(all_zero_cols) > 0:
        for i in all_zero_cols:
            mat[0, i] = -1
    return mat


def test_predict(x: np.ndarray) -> None:
    model = GlmnetGaussianModel(np.zeros(n_rows), x, 0, 0)
    prediction = model.predict()
    np.testing.assert_almost_equal(prediction, np.zeros(n_rows))


def test_r2(x: np.ndarray, y: np.ndarray) -> None:
    model = GlmnetGaussianModel(y, x, 0, 0)
    r2 = get_r2(model, y)
    np.testing.assert_almost_equal(r2, 0)


def test_glmnet_unpenalized(y: np.ndarray, x: np.ndarray) -> None:
    design_mat = np.hstack((np.ones((len(y), 1)), x))
    expected = np.linalg.lstsq(design_mat, y, rcond=None)[0]

    # if it starts at the solution, it stays there
    model = fit_glmnet(y, x, 0, 0.5, start_params=expected[1:])
    np.testing.assert_almost_equal(model.intercept, expected[0])
    np.testing.assert_almost_equal(model.params, expected[1:])


def test_glmnet_unpenalized_sparse(y: np.ndarray, x_sparse: sps.spmatrix) -> None:
    design_mat = sps.hstack((np.ones((len(y), 1)), x_sparse))
    expected = lsqr(design_mat, y)[0]

    # if it starts at the solution, it stays there
    model = fit_glmnet(y, x_sparse, 0, 0.5, start_params=expected[1:], solver="sparse")
    np.testing.assert_almost_equal(model.intercept, expected[0])
    np.testing.assert_almost_equal(model.params, expected[1:])


def test_set_params_dense(y: np.ndarray, x: np.ndarray) -> None:
    """
    Non-trivial because glmnet will normalize and then un-normalize.
    """
    np.random.seed(0)
    start_params = np.random.normal(0, 1, x.shape[1])
    model = fit_glmnet(y, x, 0, 0, 0, start_params=start_params)
    np.testing.assert_almost_equal(model.params, start_params)


def test_set_params_sparse(y: np.ndarray, x_sparse: sps.spmatrix) -> None:
    """
    Non-trivial because glmnet will normalize and then un-normalize.
    """
    np.random.seed(0)
    start_params = np.random.normal(0, 1, n_cols)
    model = fit_glmnet(y, x_sparse, 0, 0, 0, start_params=start_params, solver="sparse")
    np.testing.assert_almost_equal(model.params, start_params)


def test_glmnet_ridge(y: np.ndarray, x: np.ndarray) -> None:
    penalty = 100
    design_mat = np.hstack((np.ones((len(y), 1)), x))
    # regularization is on standardized coefficients
    tik_mat = len(y) * penalty * np.diag(design_mat.std(0))
    mat = design_mat.T.dot(design_mat) + tik_mat
    vec = design_mat.T.dot(y)
    expected = np.linalg.lstsq(mat, vec, rcond=None)[0]

    model = fit_glmnet(y, x, penalty, 0)

    # These are not all that close for some reason
    np.testing.assert_almost_equal(model.params, expected[1:], 3)
    np.testing.assert_almost_equal(model.intercept, expected[0], 3)


def test_glmnet_ridge_sparse(y: np.ndarray, x: np.ndarray) -> None:
    penalty = 100
    design_mat = np.hstack((np.ones((len(y), 1)), x))
    # regularization is on standardized coefficients
    tik_mat = len(y) * penalty * np.diag(design_mat.std(0))
    mat = design_mat.T.dot(design_mat) + tik_mat
    vec = design_mat.T.dot(y)
    expected = np.linalg.lstsq(mat, vec, rcond=None)[0]

    model = fit_glmnet(y, sps.csc_matrix(x), penalty, 0, solver="sparse")

    # These are not all that close for some reason
    np.testing.assert_almost_equal(model.params, expected[1:], 3)
    np.testing.assert_almost_equal(model.intercept, expected[0], 3)


def test_fit_pathwise(y: np.ndarray, x: np.ndarray) -> None:
    design_mat = np.hstack((np.ones((len(y), 1)), x))
    expected = np.linalg.lstsq(design_mat, y, rcond=None)[0]

    model = fit_pathwise(y, x, 1)
    np.testing.assert_almost_equal(model.intercept, expected[0])
    np.testing.assert_almost_equal(model.params, expected[1:], 6)


def test_fit_pathwise_sparse(y: np.ndarray, x: np.ndarray) -> None:
    design_mat = np.hstack((np.ones((len(y), 1)), x))
    expected = np.linalg.lstsq(design_mat, y, rcond=None)[0]

    model = fit_pathwise(y, sps.csc_matrix(x), 1, solver="sparse")
    np.testing.assert_almost_equal(model.intercept, expected[0])
    np.testing.assert_almost_equal(model.params, expected[1:], 6)


def test_penalty_scaling(y: np.ndarray, x: np.ndarray) -> None:
    model_1 = fit_glmnet(y, x, 1, 0.5)
    model_2 = fit_glmnet(y, x, 2, 0.5, penalty_scaling=np.ones(x.shape[1]) * 0.5)
    np.testing.assert_almost_equal(model_1.intercept, model_2.intercept)
    np.testing.assert_almost_equal(model_1.params, model_2.params)
