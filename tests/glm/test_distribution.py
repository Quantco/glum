from typing import Union

import numpy as np
import pytest
import quantcore.matrix as mx
import scipy as sp

from quantcore.glm._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)
from quantcore.glm._link import IdentityLink, LogitLink, LogLink, TweedieLink
from quantcore.glm._util import _safe_sandwich_dot


@pytest.mark.parametrize(
    "distribution, expected",
    [
        (NormalDistribution(), -np.inf),
        (PoissonDistribution(), 0),
        (TweedieDistribution(power=-0.5), -np.inf),
        (GammaDistribution(), 0),
        (InverseGaussianDistribution(), 0),
        (TweedieDistribution(power=1.5), 0),
    ],
)
def test_lower_bounds(
    distribution: ExponentialDispersionModel, expected: Union[float, int]
):
    assert distribution.lower_bound == expected


@pytest.mark.parametrize(
    "family, expected",
    [
        (NormalDistribution(), [True, True, True]),
        (PoissonDistribution(), [False, True, True]),
        (TweedieDistribution(power=1.5), [False, True, True]),
        (GammaDistribution(), [False, False, True]),
        (InverseGaussianDistribution(), [False, False, True]),
        (TweedieDistribution(power=4.5), [False, False, True]),
    ],
)
def test_family_bounds(family, expected):
    """Test the valid range of distributions at -1, 0, 1."""
    result = family.in_y_range([-1, 0, 1])
    np.testing.assert_array_equal(result, expected)


def test_tweedie_distribution_power():
    with pytest.raises(ValueError, match="no distribution exists"):
        TweedieDistribution(power=0.5)
    with pytest.raises(TypeError, match="must be an int or float"):
        TweedieDistribution(power=1j)
    with pytest.raises(TypeError, match="must be an int or float"):
        dist = TweedieDistribution()
        dist.power = 1j

    dist = TweedieDistribution()
    assert dist.include_lower_bound is False
    dist.power = 1
    assert dist.include_lower_bound is True


@pytest.mark.parametrize(
    "family, chk_values",
    [
        (NormalDistribution(), [-1.5, -0.1, 0.1, 2.5]),
        (PoissonDistribution(), [0.1, 1.5]),
        (GammaDistribution(), [0.1, 1.5]),
        (InverseGaussianDistribution(), [0.1, 1.5]),
        (TweedieDistribution(power=-2.5), [0.1, 1.5]),
        (TweedieDistribution(power=-1), [0.1, 1.5]),
        (TweedieDistribution(power=1.5), [0.1, 1.5]),
        (TweedieDistribution(power=2.5), [0.1, 1.5]),
        (TweedieDistribution(power=-4), [0.1, 1.5]),
        (GeneralizedHyperbolicSecant(), [0.1, 1.5]),
    ],
)
def test_deviance_zero(family, chk_values):
    """Test deviance(y,y) = 0 for different families."""
    for x in chk_values:
        np.testing.assert_allclose(family.deviance(x, x), 0, atol=1e-9)


@pytest.mark.parametrize(
    "family, link",
    [
        (NormalDistribution(), IdentityLink()),
        (PoissonDistribution(), LogLink()),
        (GammaDistribution(), LogLink()),
        (InverseGaussianDistribution(), LogLink()),
        (TweedieDistribution(power=1.5), LogLink()),
        (TweedieDistribution(power=2.5), LogLink()),
        (BinomialDistribution(), LogitLink()),
        (TweedieDistribution(power=1.5), TweedieLink(1.5)),
        (TweedieDistribution(power=2.5), TweedieLink(2.5)),
    ],
    ids=lambda args: args.__class__.__name__,
)
def test_gradients(family, link):
    np.random.seed(1001)
    for _ in range(5):
        nrows = 100
        ncols = 10
        X = np.random.rand(nrows, ncols)
        coef = np.random.rand(ncols)
        y = np.random.rand(nrows)
        weights = np.ones(nrows)

        eta, mu, _ = family.eta_mu_deviance(
            link, 1.0, np.zeros(nrows), X.dot(coef), y, weights
        )
        gradient_rows, _ = family.rowwise_gradient_hessian(
            link=link,
            coef=coef,
            phi=1.0,
            X=X,
            y=y,
            weights=weights,
            eta=eta,
            mu=mu,
        )
        score_analytic = gradient_rows @ X

        def f(coef2):
            _, _, ll = family.eta_mu_deviance(
                link, 1.0, np.zeros(nrows), X.dot(coef2), y, weights
            )
            return -0.5 * ll

        score_numeric = np.empty_like(score_analytic)
        epsilon = 1e-7
        for k in range(score_numeric.shape[0]):
            L = coef.copy()
            L[k] -= epsilon
            R = coef.copy()
            R[k] += epsilon
            score_numeric[k] = (f(R) - f(L)) / (2 * epsilon)
        np.testing.assert_allclose(score_numeric, score_analytic, rtol=5e-5)


@pytest.mark.parametrize(
    "family, link, true_hessian",
    [
        (NormalDistribution(), IdentityLink(), False),
        (PoissonDistribution(), LogLink(), False),
        (GammaDistribution(), LogLink(), True),
        (InverseGaussianDistribution(), LogLink(), False),
        (TweedieDistribution(power=1.5), LogLink(), True),
        (TweedieDistribution(power=4.5), LogLink(), False),
    ],
    ids=lambda args: args.__class__.__name__,
)
def test_hessian_matrix(family, link, true_hessian):
    """Test the Hessian matrix numerically.

    Trick: For the FIM, use numerical differentiation with y = mu
    """
    coef = np.array([-2, 1, 0, 1, 2.5])
    phi = 0.5
    rng = np.random.RandomState(42)
    X = mx.DenseMatrix(rng.randn(10, 5))
    lin_pred = np.dot(X, coef)
    mu = link.inverse(lin_pred)
    weights = rng.randn(10) ** 2 + 1
    _, hessian_rows = family.rowwise_gradient_hessian(
        link=link,
        coef=coef,
        phi=phi,
        X=X,
        y=weights,
        weights=weights,
        eta=lin_pred,
        mu=mu,
    )
    hessian = _safe_sandwich_dot(X, hessian_rows)
    # check that the Hessian matrix is square and positive definite
    assert hessian.ndim == 2
    assert hessian.shape[0] == hessian.shape[1]
    assert np.all(np.linalg.eigvals(hessian) >= 0)

    approx = np.array([]).reshape(0, coef.shape[0])
    for i in range(coef.shape[0]):

        def f(coef):
            this_eta = X.dot(coef)
            this_mu = link.inverse(this_eta)
            yv = mu
            if true_hessian:
                # If we're using the true hessian, use the true y
                yv = weights
            else:
                # If we're using the FIM, use y = mu
                yv = mu
            gradient_rows, _ = family.rowwise_gradient_hessian(
                link=link,
                coef=coef,
                phi=phi,
                X=X,
                y=yv,
                weights=weights,
                eta=this_eta,
                mu=this_mu,
            )
            score = gradient_rows @ X
            return -score[i]

        approx = np.vstack(
            [approx, sp.optimize.approx_fprime(xk=coef, f=f, epsilon=1e-5)]
        )
    np.testing.assert_allclose(hessian, approx, rtol=1e-3)
