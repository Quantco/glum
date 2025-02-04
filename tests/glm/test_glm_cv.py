import numpy as np
import pandas as pd
import pytest
import sklearn as skl
import tabmat as tm
from scipy import sparse

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
    X, y, _ = skl.datasets.make_regression(
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

    x_arr = X if isinstance(X, np.ndarray) else X.toarray()
    t_arr = T if isinstance(T, np.ndarray) else T.toarray()
    elastic_net = skl.linear_model.ElasticNetCV(
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
    X, y, coef = skl.datasets.make_regression(
        n_samples=n_samples + n_predict,
        n_features=n_features,
        n_informative=n_features - 2,
        noise=0.5,
        coef=True,
        random_state=42,
    )
    y = y[0:n_samples]
    X, T = X[0:n_samples], X[n_samples:]

    ridge = skl.linear_model.RidgeCV(
        fit_intercept=fit_intercept, cv=5, alphas=alphas
    ).fit(X, y)
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


# TODO: different distributions
# Specify rtol since some are more accurate than others
@pytest.mark.parametrize(
    "params",
    [
        {"solver": "irls-ls", "rtol": 1e-6},
        {"solver": "lbfgs", "rtol": 2e-4},
        {"solver": "trust-constr", "rtol": 2e-4},
        {"solver": "irls-cd", "selection": "cyclic", "rtol": 2e-5},
        {"solver": "irls-cd", "selection": "random", "rtol": 6e-5},
    ],
    ids=lambda params: ", ".join(f"{key}={val}" for key, val in params.items()),
)
@pytest.mark.parametrize("use_offset", [False, True])
def test_solver_equivalence_cv(params, use_offset):
    n_alphas = 3
    n_samples = 100
    n_features = 10
    gradient_tol = 1e-5

    X, y = skl.datasets.make_regression(
        n_samples=n_samples, n_features=n_features, random_state=2
    )

    if use_offset:
        np.random.seed(0)
        offset = np.random.random(len(y))
    else:
        offset = None

    est_ref = GeneralizedLinearRegressorCV(
        random_state=2,
        n_alphas=n_alphas,
        gradient_tol=gradient_tol,
        min_alpha_ratio=1e-3,
    )
    est_ref.fit(X, y, offset=offset)

    est_2 = (
        GeneralizedLinearRegressorCV(
            n_alphas=n_alphas,
            max_iter=1000,
            gradient_tol=gradient_tol,
            **{k: v for k, v in params.items() if k != "rtol"},
            min_alpha_ratio=1e-3,
        )
        .set_params(random_state=2)
        .fit(X, y, offset=offset)
    )

    def _assert_all_close(x, y):
        return np.testing.assert_allclose(x, y, rtol=params["rtol"], atol=1e-7)

    _assert_all_close(est_2.alphas_, est_ref.alphas_)
    _assert_all_close(est_2.alpha_, est_ref.alpha_)
    _assert_all_close(est_2.l1_ratio_, est_ref.l1_ratio_)
    _assert_all_close(est_2.coef_path_, est_ref.coef_path_)
    _assert_all_close(est_2.deviance_path_, est_ref.deviance_path_)
    _assert_all_close(est_2.intercept_, est_ref.intercept_)
    _assert_all_close(est_2.coef_, est_ref.coef_)
    _assert_all_close(
        skl.metrics.mean_absolute_error(est_2.predict(X), y),
        skl.metrics.mean_absolute_error(est_ref.predict(X), y),
    )


def test_formula():
    """Model with formula and model with externally constructed model matrix should
    match.
    """
    n_samples = 100
    n_alphas = 2
    tol = 1e-9

    np.random.seed(10)
    data = pd.DataFrame(
        {
            "y": np.random.rand(n_samples),
            "x1": np.random.rand(n_samples),
            "x2": np.random.rand(n_samples),
        }
    )
    formula = "y ~ x1 + x2"

    model_formula = GeneralizedLinearRegressorCV(
        family="normal",
        formula=formula,
        fit_intercept=False,
        n_alphas=n_alphas,
        gradient_tol=tol,
    ).fit(data)

    y = data["y"]
    X = data[["x1", "x2"]]

    model_pandas = GeneralizedLinearRegressorCV(
        family="normal",
        fit_intercept=False,
        n_alphas=n_alphas,
        gradient_tol=tol,
    ).fit(X, y)

    np.testing.assert_almost_equal(model_pandas.coef_, model_formula.coef_)
    np.testing.assert_array_equal(
        model_pandas.feature_names_, model_formula.feature_names_
    )
