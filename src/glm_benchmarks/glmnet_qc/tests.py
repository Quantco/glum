import numpy as np
import pytest
import statsmodels.api as sm
from glmnet_python import glmnet
from scipy import sparse as sps
from scipy.sparse.linalg import lsqr

from glm_benchmarks.glmnet_qc.model import GaussianCanonicalModel, GlmnetModel

from .glmnet_qc import (
    _get_coordinate_wise_update_naive,
    _get_coordinate_wise_update_sparse,
    fit_glmnet,
    fit_glmnet_gaussian_canonical,
    fit_pathwise,
)

n_rows = 10
n_cols = 5


@pytest.fixture
def y() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, n_rows)


@pytest.fixture
def x() -> np.ndarray:
    np.random.seed(0)
    return np.hstack((np.ones((n_rows, 1)), np.random.normal(0, 1, (n_rows, n_cols))))


@pytest.fixture
def x_sparse() -> sps.spmatrix:
    np.random.seed(0)
    mat = sps.hstack((np.ones((n_rows, 1)), sps.random(n_rows, n_cols, density=0.2)))
    all_zero_cols = np.where((mat != 0).sum(0) == 0)[0]
    if len(all_zero_cols) > 0:
        for i in all_zero_cols:
            mat[0, i] = -1
    return mat.tocsc()


def test_predict(x: np.ndarray) -> None:
    model = GlmnetModel(np.zeros(n_rows), x, "gaussian", 0, 0)
    prediction = model.predict()
    np.testing.assert_almost_equal(prediction, np.zeros(n_rows))


def test_predict_custom_link(x: np.ndarray):
    model = GlmnetModel(np.zeros(n_rows), x, "gaussian", 0, 0, link_name="log")
    prediction = model.predict()
    np.testing.assert_almost_equal(prediction, np.ones(n_rows))


def test_r2(x: np.ndarray, y: np.ndarray):
    model = GlmnetModel(y, x, "gaussian", 0, 0)
    r2 = model.get_r2(y)
    np.testing.assert_almost_equal(r2, 0)


def test_r2_custom_link(x: np.ndarray, y: np.ndarray):
    model = GlmnetModel(y, x, "gaussian", 0, 0, link_name="log")
    r2 = model.get_r2(y)
    np.testing.assert_almost_equal(r2, 0)


def test_fit_glmnet_gaussian__unpenalized_unweighted(y: np.ndarray, x: np.ndarray):
    expected = np.linalg.lstsq(x, y, rcond=None)[0]

    # if it starts at the solution, it stays there
    model = fit_glmnet_gaussian_canonical(y, x, 0, 0.5, start_params=expected)
    np.testing.assert_almost_equal(model.params, expected)


def test_fit_glmnet_gaussian__unpenalized_weighted(y: np.ndarray, x: np.ndarray):
    np.random.seed(0)
    w = np.random.uniform(0, 1, len(y))
    expected = sm.WLS(y, x, w).fit().params

    # if it starts at the solution, it stays there
    model = fit_glmnet_gaussian_canonical(y, x, 0, 0.5, w, start_params=expected)
    np.testing.assert_almost_equal(model.params, expected)


def test_fit_glmnet__unpenalized_unweighted(y: np.ndarray, x: np.ndarray):
    expected = np.linalg.lstsq(x, y, rcond=None)[0]

    # if it starts at the solution, it stays there
    model = fit_glmnet(y, x, 0, 0.5, start_params=expected)
    np.testing.assert_almost_equal(model.params, expected)


def test_fit_glmnet__unpenalized_unweighted_sparse(
    y: np.ndarray, x_sparse: sps.spmatrix
):
    expected = lsqr(x_sparse, y)[0]

    # if it starts at the solution, it stays there
    model = fit_glmnet(y, x_sparse, 0, 0.5, start_params=expected)
    np.testing.assert_almost_equal(model.params, expected)


def test_set_params_dense(y: np.ndarray, x: np.ndarray) -> None:
    """
    used to be non-trivial, but now it is.
    """
    np.random.seed(0)
    start_params = np.random.normal(0, 1, x.shape[1])
    model = fit_glmnet(y, x, 0, 0, n_iters=0, start_params=start_params)
    np.testing.assert_almost_equal(model.params, start_params)


def test_set_params_sparse(y: np.ndarray, x_sparse: sps.spmatrix) -> None:
    """
    used to be non-trivial, but now it is.
    """
    np.random.seed(0)
    start_params = np.random.normal(0, 1, x_sparse.shape[1])
    model = fit_glmnet(y, x_sparse, 0, 0, n_iters=0, start_params=start_params)
    np.testing.assert_almost_equal(model.params, start_params)


def _get_ridge_solution(y_: np.ndarray, x_: np.ndarray, penalty: float):

    penalty_factor = np.ones(x_.shape[1]) * penalty
    penalty_factor[0] = 0
    tik_mat = np.diag(penalty_factor * len(y_))
    mat = x_.T.dot(x_) + tik_mat
    vec = x_.T.dot(y_)
    return np.linalg.lstsq(mat, vec, rcond=None)[0]


def test_fit_glmnet_gc__ridge_unweighted(y: np.ndarray, x: np.ndarray) -> None:
    """
    min 1/2 sum_i w_i (y_i - x_i^T beta)^2 + 1/2 penalty * sum_k beta_k^2
    0 = sum_i w_i (y_i - x_i^T beta) x_i + penalty * beta
    sum_i w_i y_i x_i = (sum_i w_i x_i x_i^T  + eye * penalty) beta

    """
    penalty = 100
    expected = _get_ridge_solution(y, x, penalty)

    model = fit_glmnet_gaussian_canonical(y, x, penalty, 0)

    # These are not all that close for some reason
    np.testing.assert_almost_equal(model.params, expected, 3)


def test_fit_glmnet__ridge_unweighted(y: np.ndarray, x: np.ndarray) -> None:
    """
    min 1/2 sum_i w_i (y_i - x_i^T beta)^2 + 1/2 penalty * sum_k beta_k^2
    0 = sum_i w_i (y_i - x_i^T beta) x_i + penalty * beta
    sum_i w_i y_i x_i = (sum_i w_i x_i x_i^T  + eye * penalty) beta

    """
    penalty = 100
    expected = _get_ridge_solution(y, x, penalty)
    model = fit_glmnet(y, x, penalty, 0)
    # These are not all that close for some reason
    np.testing.assert_almost_equal(model.params, expected, 3)


def test_fit_glmnet_cv__ridge_sparse(y: np.ndarray, x: np.ndarray) -> None:
    penalty = 100
    expected = _get_ridge_solution(y, x, penalty)

    model = fit_glmnet_gaussian_canonical(y, x, penalty, 0)

    # These are not all that close for some reason
    np.testing.assert_almost_equal(model.params, expected, 3)


def test_glmnet_ridge_sparse(y: np.ndarray, x: np.ndarray) -> None:
    penalty = 100
    expected = _get_ridge_solution(y, x, penalty)

    model = fit_glmnet(y, x, penalty, 0)

    # These are not all that close for some reason
    np.testing.assert_almost_equal(model.params, expected, 3)


def test_fit_pathwise(y: np.ndarray, x: np.ndarray) -> None:
    expected = np.linalg.lstsq(x, y, rcond=None)[0]

    model = fit_pathwise(y, x, 1)
    np.testing.assert_almost_equal(model.params, expected, 6)


def test_fit_pathwise_sparse(y: np.ndarray, x: np.ndarray) -> None:
    expected = np.linalg.lstsq(x, y, rcond=None)[0]

    model = fit_pathwise(y, sps.csc_matrix(x), 1)
    np.testing.assert_almost_equal(model.params, expected, 6)


def test_penalty_scaling(y: np.ndarray, x: np.ndarray) -> None:
    scale = 100
    model_1 = fit_glmnet(y, x, 1, 0.5)
    model_2 = fit_glmnet(
        y, x, scale, 0.5, penalty_scaling=model_1.penalty_scaling / scale
    )

    np.testing.assert_almost_equal(model_1.params, model_2.params)


def glmnet_poisson_tester(alpha: float, l1_ratio: float) -> None:
    X = np.array([[-2, -1, 1, 2], [0, 0, 1, 1.0]]).T
    y = np.array([0, 1, 1, 2.0])
    glm = fit_glmnet(
        y,
        np.hstack((np.ones((4, 1)), X)),
        alpha,
        l1_ratio,
        n_iters=20,
        distribution="poisson",
    )

    glmnet_m = glmnet(
        x=X.copy(),
        y=y.copy(),
        family="poisson",
        alpha=l1_ratio,
        lambdau=np.array([alpha]),
        standardize=False,
        thresh=1e-7,
    )

    np.testing.assert_almost_equal(glm.params[0], glmnet_m["a0"], 4)
    np.testing.assert_almost_equal(glm.params[1:], glmnet_m["beta"][:, 0], 4)


def test_glmnet_poisson_ridge():
    glmnet_poisson_tester(1.0, 0.0)


def test_glmnet_poisson_lasso():
    glmnet_poisson_tester(0.1, 1.0)


def test_glmnet_poisson_net():
    glmnet_poisson_tester(0.1, 0.5)


def test_get_coordinate_wise_update_sparse(y: np.ndarray, x_sparse: sps.spmatrix):
    model = GaussianCanonicalModel(y, x_sparse.tocsc(), 0.5, 0)
    resid = y - model.predict()
    sparse_param, sparse_resid = _get_coordinate_wise_update_sparse(
        model, 0, resid.copy()
    )
    dense_model = GaussianCanonicalModel(y, x_sparse.A, 0.5, 0)

    np.testing.assert_allclose(model.predict(), dense_model.predict())

    dense_param, dense_resid = _get_coordinate_wise_update_naive(
        dense_model, 0, resid.copy()
    )

    np.testing.assert_allclose(sparse_param, dense_param)
    np.testing.assert_allclose(sparse_resid, dense_resid)
