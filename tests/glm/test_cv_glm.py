import numpy as np
import pytest
import tabmat as tm
from scipy import sparse as sparse
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNetCV, RidgeCV

from glum import GeneralizedLinearRegressorCV

GLM_SOLVERS = ["irls", "lbfgs", "cd", "trust-constr"]


@pytest.mark.parametrize("l1_ratio", [0.5, 1, [0.3, 0.6], np.array([0.3, 0.6])])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize(
    "convert_x_fn",
    [
        np.asarray,
        sparse.csc_matrix,
        sparse.csr_matrix,
        tm.DenseMatrix,
        lambda x: tm.SparseMatrix(sparse.csc_matrix(x)),
        lambda x: tm.from_csc(sparse.csc_matrix(x)),
    ],
)
def test_normal_elastic_net_comparison(l1_ratio, fit_intercept, convert_x_fn):
    """
    Compare against sklearn's ElasticNetCV.

    Don't test l1_ratio = 0 because automatic grid generation is not supported in
    ElasticNetCV for l1_ratio = 0.
    """
    n_samples = 100
    n_alphas = 2
    n_features = 10
    tol = 1e-9

    n_predict = 10
    X, y, _ = make_regression(
        n_samples=n_samples + n_predict,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=0.5,
        coef=True,
        random_state=42,
    )
    X = convert_x_fn(X)
    y = y[0:n_samples]
    X, T = X[0:n_samples], X[n_samples:]

    x_arr = X if isinstance(X, np.ndarray) else X.A
    t_arr = T if isinstance(T, np.ndarray) else T.A
    elastic_net = ElasticNetCV(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        fit_intercept=fit_intercept,
        tol=tol,
    ).fit(x_arr, y)
    el_pred = elastic_net.predict(t_arr)

    glm = GeneralizedLinearRegressorCV(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        fit_intercept=fit_intercept,
        link="identity",
        gradient_tol=tol,
        min_alpha_ratio=1e-3,
    ).fit(X, y)

    glm_pred = glm.predict(T)

    np.testing.assert_allclose(glm.l1_ratio_, elastic_net.l1_ratio_)
    np.testing.assert_allclose(glm.alphas_, elastic_net.alphas_)
    np.testing.assert_allclose(glm.alpha_, elastic_net.alpha_)
    np.testing.assert_allclose(glm.intercept_, elastic_net.intercept_)
    np.testing.assert_allclose(glm.coef_, elastic_net.coef_)
    np.testing.assert_allclose(glm_pred, el_pred)
    # need to divide mse by number of folds
    np.testing.assert_allclose(
        np.moveaxis(np.squeeze(glm.deviance_path_), 0, -1), elastic_net.mse_path_ / 5
    )


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_normal_ridge_comparison(fit_intercept):
    """
    Test against sklearn's RidgeCV.

    Not testing l1_ratio = 0 because automatic grid generation is not supported
    in ElasticNetCV for l1_ratio = 0.
    """
    n_samples = 100
    n_features = 2
    tol = 1e-9
    alphas = [1e-4]

    n_predict = 10
    X, y, coef = make_regression(
        n_samples=n_samples + n_predict,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=0.5,
        coef=True,
        random_state=42,
    )
    y = y[0:n_samples]
    X, T = X[0:n_samples], X[n_samples:]

    ridge = RidgeCV(fit_intercept=fit_intercept, cv=5, alphas=alphas).fit(X, y)
    el_pred = ridge.predict(T)

    glm = GeneralizedLinearRegressorCV(
        fit_intercept=fit_intercept,
        link="identity",
        gradient_tol=tol,
        alphas=alphas,
        l1_ratio=0,
        min_alpha_ratio=1e-3,
    ).fit(X, y)
    glm_pred = glm.predict(T)

    np.testing.assert_allclose(glm.alpha_, ridge.alpha_)
    np.testing.assert_allclose(glm_pred, el_pred, atol=4e-6)
    np.testing.assert_allclose(glm.intercept_, ridge.intercept_, atol=4e-7)
    np.testing.assert_allclose(glm.coef_, ridge.coef_, atol=3e-6)
