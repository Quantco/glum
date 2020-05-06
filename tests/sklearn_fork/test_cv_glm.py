import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNetCV

from glm_benchmarks.sklearn_fork import GeneralizedLinearRegressorCV

GLM_SOLVERS = ["irls", "lbfgs", "cd"]


@pytest.mark.parametrize("l1_ratio", [0.5, 1, [0.3, 0.6]])
def test_normal_elastic_net_comparison(l1_ratio):
    """
    Not testing l1_ratio = 0 because automatic grid generation is not supported
    in ElasticNetCV for l1_ratio = 0.
    """
    n_samples = 100
    n_alphas = 10
    n_features = 10

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

    elastic_net = ElasticNetCV(l1_ratio, n_alphas=n_alphas).fit(X, y)
    el_pred = elastic_net.predict(T)

    glm = GeneralizedLinearRegressorCV(l1_ratio=l1_ratio, n_alphas=n_alphas).fit(X, y)
    glm_pred = glm.predict(T)

    np.testing.assert_allclose(glm.l1_ratio_, elastic_net.l1_ratio_)
    np.testing.assert_allclose(glm.alphas_, elastic_net.alphas_)
    np.testing.assert_allclose(glm.alpha_, elastic_net.alpha_)
    np.testing.assert_allclose(glm.intercept_, elastic_net.intercept_, rtol=5e-4)
    np.testing.assert_allclose(glm.coef_, elastic_net.coef_, rtol=4e-3)
    np.testing.assert_allclose(glm_pred, el_pred, rtol=9e-5)
