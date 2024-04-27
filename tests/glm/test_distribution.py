import numpy as np
import pytest
import scipy as sp
import tabmat as tm

from glum._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)
from glum._glm import GeneralizedLinearRegressor, get_family
from glum._link import IdentityLink, LogitLink, LogLink, TweedieLink
from glum._util import _safe_sandwich_dot


@pytest.mark.parametrize(
    "distribution, expected",
    [
        (NormalDistribution(), -np.inf),
        (PoissonDistribution(), 0),
        (TweedieDistribution(power=-0.5), -np.inf),
        (GammaDistribution(), 0),
        (InverseGaussianDistribution(), 0),
        (TweedieDistribution(power=1.5), 0),
        (NegativeBinomialDistribution(theta=1.5), 0),
    ],
)
def test_lower_bounds(distribution: ExponentialDispersionModel, expected: float):
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
        (NegativeBinomialDistribution(theta=1.0), [False, True, True]),
    ],
)
def test_family_bounds(family, expected):
    """Test the valid range of distributions at -1, 0, 1."""
    result = family.in_y_range([-1, 0, 1])
    np.testing.assert_array_equal(result, expected)


def test_tweedie_distribution_power():
    with pytest.raises(ValueError, match="no distribution exists"):
        TweedieDistribution(power=0.5)
    with pytest.raises(TypeError, match="must be numeric"):
        TweedieDistribution(power=1j)
    with pytest.raises(TypeError, match="must be numeric"):
        dist = TweedieDistribution()
        dist.power = 1j

    dist = TweedieDistribution()
    assert dist.include_lower_bound is False
    dist.power = 1
    assert dist.include_lower_bound is True


def test_tweedie_distribution_parsing():
    dist = get_family("tweedie")

    assert isinstance(dist, TweedieDistribution)
    assert dist.power == 1.5

    dist = get_family("tweedie (1.25)")

    assert isinstance(dist, TweedieDistribution)
    assert dist.power == 1.25

    dist = get_family("tweedie(1.25)")

    assert isinstance(dist, TweedieDistribution)
    assert dist.power == 1.25

    with pytest.raises(ValueError):
        get_family("tweedie (a)")


def test_negative_binomial_distribution_alpha():
    with pytest.raises(ValueError, match="must be strictly positive"):
        NegativeBinomialDistribution(theta=-0.5)
    with pytest.raises(TypeError, match="must be numeric"):
        NegativeBinomialDistribution(theta=1j)
    with pytest.raises(TypeError, match="must be numeric"):
        dist = NegativeBinomialDistribution()
        dist.theta = 1j


def test_negative_binomial_distribution_parsing():
    dist = get_family("negative.binomial")

    assert isinstance(dist, NegativeBinomialDistribution)
    assert dist.theta == 1.0

    dist = get_family("negative.binomial (1.25)")

    assert isinstance(dist, NegativeBinomialDistribution)
    assert dist.theta == 1.25

    dist = get_family("negative.binomial(1.25)")

    assert isinstance(dist, NegativeBinomialDistribution)
    assert dist.theta == 1.25

    with pytest.raises(ValueError):
        get_family("negative.binomial (a)")


def test_equality():
    assert BinomialDistribution() == BinomialDistribution()
    assert GammaDistribution() == GammaDistribution()
    assert NegativeBinomialDistribution(1) != BinomialDistribution()
    assert NegativeBinomialDistribution(1) != NegativeBinomialDistribution(1.5)
    assert NegativeBinomialDistribution(1) == NegativeBinomialDistribution(1)
    assert NormalDistribution() == NormalDistribution()
    assert PoissonDistribution() == PoissonDistribution()
    assert TweedieDistribution(0) != NormalDistribution()
    assert TweedieDistribution(0) == NormalDistribution().to_tweedie()
    assert TweedieDistribution(1) != BinomialDistribution()
    assert TweedieDistribution(1) != PoissonDistribution()
    assert TweedieDistribution(1) == PoissonDistribution().to_tweedie()
    assert TweedieDistribution(1) != TweedieDistribution(1.5)
    assert TweedieDistribution(1) == TweedieDistribution(1)
    assert TweedieDistribution(2) != GammaDistribution()
    assert TweedieDistribution(2) == GammaDistribution().to_tweedie()
    assert TweedieDistribution(3) != InverseGaussianDistribution()
    assert TweedieDistribution(3) == InverseGaussianDistribution().to_tweedie()


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
        (NegativeBinomialDistribution(theta=1.0), [0.1, 1.5]),
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
        (NegativeBinomialDistribution(theta=1.0), LogLink()),
    ],
    ids=lambda args: args.__class__.__name__,
)
def test_gradients(family, link):
    np.random.seed(1001)

    nrows = 100
    ncols = 10
    X = np.random.rand(nrows, ncols)
    coef = np.random.rand(ncols)
    y = np.random.rand(nrows)
    sample_weight = np.ones(nrows)

    for _ in range(5):
        eta, mu, _ = family.eta_mu_deviance(
            link, 1.0, np.zeros(nrows), X.dot(coef), y, sample_weight
        )
        gradient_rows, _ = family.rowwise_gradient_hessian(
            link=link,
            coef=coef,
            dispersion=1.0,
            X=X,
            y=y,
            sample_weight=sample_weight,
            eta=eta,
            mu=mu,
        )
        score_analytic = gradient_rows @ X

        def f(coef2):
            _, _, ll = family.eta_mu_deviance(
                link, 1.0, np.zeros(nrows), X.dot(coef2), y, sample_weight
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
        (NegativeBinomialDistribution(theta=1.0), LogLink(), False),
    ],
    ids=lambda args: args.__class__.__name__,
)
def test_hessian_matrix(family, link, true_hessian):
    """Test the Hessian matrix numerically.

    Trick: For the FIM, use numerical differentiation with y = mu
    """
    coef = np.array([-2, 1, 0, 1, 2.5])
    dispersion = 0.5
    rng = np.random.RandomState(42)
    X = tm.DenseMatrix(rng.randn(10, 5))
    lin_pred = X.matvec(coef)
    mu = link.inverse(lin_pred)
    sample_weight = rng.randn(10) ** 2 + 1
    _, hessian_rows = family.rowwise_gradient_hessian(
        link=link,
        coef=coef,
        dispersion=dispersion,
        X=X,
        y=sample_weight,
        sample_weight=sample_weight,
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
            this_eta = X.matvec(coef)
            this_mu = link.inverse(this_eta)
            yv = mu
            if true_hessian:
                # If we're using the true hessian, use the true y
                yv = sample_weight
            else:
                # If we're using the FIM, use y = mu
                yv = mu
            gradient_rows, _ = family.rowwise_gradient_hessian(
                link=link,
                coef=coef,
                dispersion=dispersion,
                X=X,
                y=yv,
                sample_weight=sample_weight,
                eta=this_eta,
                mu=this_mu,
            )
            score = gradient_rows @ X
            return -score[i]  # noqa B023

        approx = np.vstack(
            [approx, sp.optimize.approx_fprime(xk=coef, f=f, epsilon=1e-5)]
        )
    np.testing.assert_allclose(hessian, approx, rtol=1e-3)


@pytest.mark.parametrize("weighted", [False, True])
def test_poisson_deviance_dispersion_loglihood(weighted):
    # y <- c(0, 0, 1, 2, 3)
    # glm_model = glm(y ~ 1, family = poisson)

    # glm_model$coefficients  # 0.1823216
    # sum(glm_model$weights * glm_model$residuals^2)/4  # 1.416679
    # glm_model$deviance  # 7.176404
    # logLik(glm_model)  # -7.390977 (df=1)

    regressor = GeneralizedLinearRegressor(
        family="poisson",
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([0, 0, 1, 2, 3])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    x = np.ones((len(y), 1))
    mu = regressor.fit(x, y, sample_weight=wgts).predict(x)
    family = regressor._family_instance

    ll = family.log_likelihood(
        y,
        mu,
        sample_weight=wgts,
        # R bases dispersion on the deviance for log_likelihood
        dispersion=family.dispersion(
            y, mu, sample_weight=wgts, method="deviance", ddof=0
        ),
    )

    np.testing.assert_approx_equal(regressor.coef_[0], 0.1823216)
    np.testing.assert_approx_equal(family.deviance(y, mu, sample_weight=wgts), 7.176404)
    np.testing.assert_approx_equal(ll, -7.390977)

    # higher tolerance for the dispersion parameter because of numerical precision
    np.testing.assert_approx_equal(
        family.dispersion(y, mu, sample_weight=wgts), 1.416679, significant=5
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_gamma_deviance_dispersion_loglihood(weighted):
    # y <- c(1, 2, 2, 3, 4)
    # glm_model = glm(y ~ 1, family = Gamma(link = "log"))

    # glm_model$coefficients  # 0.8754687
    # sum(glm_model$weights * glm_model$residuals^2)/4  # 0.2256944
    # glm_model$deviance  # 1.012285
    # logLik(glm_model)  # -7.057068 (df=2)

    regressor = GeneralizedLinearRegressor(
        family="gamma",
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([1, 2, 2, 3, 4])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    x = np.ones((len(y), 1))
    mu = regressor.fit(x, y, sample_weight=wgts).predict(x)
    family = regressor._family_instance

    ll = family.log_likelihood(
        y,
        mu,
        sample_weight=wgts,
        # R bases dispersion on the deviance for log_likelihood
        dispersion=family.dispersion(
            y, mu, sample_weight=wgts, method="deviance", ddof=0
        ),
    )

    np.testing.assert_approx_equal(regressor.coef_[0], 0.8754687)
    np.testing.assert_approx_equal(
        family.dispersion(y, mu, sample_weight=wgts), 0.2256944
    )
    np.testing.assert_approx_equal(family.deviance(y, mu, sample_weight=wgts), 1.012285)
    np.testing.assert_approx_equal(ll, -7.057068)


@pytest.mark.parametrize("family", ["gaussian", "normal"])
@pytest.mark.parametrize("weighted", [False, True])
def test_gaussian_deviance_dispersion_loglihood(family, weighted):
    # y <- c(-1, -1, 0, 1, 2)
    # glm_model = glm(y ~ 1, family = gaussian)

    # glm_model$coefficients  # 0.2
    # sum(glm_model$weights * glm_model$residuals^2)/4  # 1.7
    # glm_model$deviance  # 6.8
    # logLik(glm_model)  # -7.863404 (df=2)

    regressor = GeneralizedLinearRegressor(
        family=family,
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([-1, -1, 0, 1, 2])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    x = np.ones((len(y), 1))
    mu = regressor.fit(x, y, sample_weight=wgts).predict(x)
    family = regressor._family_instance

    ll = family.log_likelihood(
        y,
        mu,
        sample_weight=wgts,
        # R bases dispersion on the deviance for log_likelihood
        dispersion=family.dispersion(
            y, mu, sample_weight=wgts, method="deviance", ddof=0
        ),
    )

    np.testing.assert_approx_equal(regressor.coef_[0], 0.2)
    np.testing.assert_approx_equal(family.dispersion(y, mu, sample_weight=wgts), 1.7)
    np.testing.assert_approx_equal(family.deviance(y, mu, sample_weight=wgts), 6.8)
    np.testing.assert_approx_equal(ll, -7.863404)


@pytest.mark.parametrize("weighted", [False, True])
def test_tweedie_deviance_dispersion_loglihood(weighted):
    # library(statmod)  # Tweedie GLMs
    # library(tweedie)  # Tweedie log likelihood

    # y <- c(0, 0, 1, 2, 3)
    # glm_model = glm(y ~ 1, family = tweedie(var.power = 1.5, link.power = 0))

    # glm_model$coefficients  # 0.1823216
    # sum(glm_model$weights * glm_model$residuals^2)/4  # 1.293318
    # glm_model$deviance  # 10.64769
    # logLiktweedie(glm_model)  # -8.35485

    regressor = GeneralizedLinearRegressor(
        family=TweedieDistribution(1.5),
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([0, 0, 1, 2, 3])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    x = np.ones((len(y), 1))
    mu = regressor.fit(x, y, sample_weight=wgts).predict(x)
    family = regressor._family_instance

    ll = family.log_likelihood(
        y,
        mu,
        sample_weight=wgts,
        # R bases dispersion on the deviance for log_likelihood
        dispersion=family.dispersion(
            y, mu, sample_weight=wgts, method="deviance", ddof=0
        ),
    )

    np.testing.assert_approx_equal(regressor.coef_[0], 0.1823216)
    np.testing.assert_approx_equal(family.deviance(y, mu, sample_weight=wgts), 10.64769)
    np.testing.assert_approx_equal(ll, -8.35485)

    # higher tolerance for the dispersion parameter because of numerical precision
    np.testing.assert_approx_equal(
        family.dispersion(y, mu, sample_weight=wgts), 1.293318, significant=5
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_binomial_deviance_dispersion_loglihood(weighted):
    # y <- c(0, 1, 0, 1, 0)
    # glm_model = glm(y ~ 1, family = binomial)

    # glm_model$coefficients  # -0.4054651
    # sum(glm_model$weights * glm_model$residuals^2)/4  # 1.25
    # glm_model$deviance  # 6.730117
    # logLik(glm_model)  # -3.365058 (df=1)

    regressor = GeneralizedLinearRegressor(
        family="binomial",
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([0, 1, 0, 1, 0])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    x = np.ones((len(y), 1))
    mu = regressor.fit(x, y, sample_weight=wgts).predict(x)
    family = regressor._family_instance

    # R bases dispersion on the deviance for log_likelihood
    ll = family.log_likelihood(
        y,
        mu,
        sample_weight=wgts,
        dispersion=family.dispersion(
            y, mu, sample_weight=wgts, method="deviance", ddof=0
        ),
    )

    np.testing.assert_approx_equal(regressor.coef_[0], -0.4054651)
    np.testing.assert_approx_equal(family.dispersion(y, mu, sample_weight=wgts), 1.25)
    np.testing.assert_approx_equal(family.deviance(y, mu, sample_weight=wgts), 6.730117)
    np.testing.assert_approx_equal(ll, -3.365058)


@pytest.mark.parametrize("weighted", [False, True])
def test_negative_binomial_deviance_dispersion_loglihood(weighted):
    # y <- c(0, 1, 0, 1, 0)
    # glm_model = glm(y~1, family=MASS::negative.binomial(theta=1))

    # glm_model$coefficients  # -0.9162907
    # sum(glm_model$weights * glm_model$residuals^2)/4  # 0.535716
    # glm_model$deviance  # 2.830597
    # logLik(glm_model)  # -4.187887 (df=1)

    regressor = GeneralizedLinearRegressor(
        family="negative.binomial",
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([0, 1, 0, 1, 0])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    x = np.ones((len(y), 1))
    mu = regressor.fit(x, y, sample_weight=wgts).predict(x)
    family = regressor._family_instance

    # R bases dispersion on the deviance for log_likelihood
    ll = family.log_likelihood(
        y,
        mu,
        sample_weight=wgts,
        dispersion=family.dispersion(
            y, mu, sample_weight=wgts, method="deviance", ddof=0
        ),
    )

    np.testing.assert_approx_equal(regressor.coef_[0], -0.9162907)
    np.testing.assert_approx_equal(
        family.dispersion(y, mu, sample_weight=wgts), 0.53571, significant=5
    )
    np.testing.assert_approx_equal(family.deviance(y, mu, sample_weight=wgts), 2.830597)
    np.testing.assert_approx_equal(ll, -4.187887)


@pytest.mark.parametrize("dispersion", [1, 5, 10, 25])
@pytest.mark.parametrize("power", [1.1, 1.5, 1.9, 1.99])
def test_tweedie_normalization(dispersion, power):
    def scipy_based_normalization(y, power, dispersion):
        alpha = (2 - power) / (1 - power)
        x = (((power - 1) / y) ** alpha) / ((2 - power) * (dispersion ** (1 - alpha)))
        return np.log(sp.special.wright_bessel(-alpha, 0, x)) - np.log(y)

    def scipy_based_loglihood(y, mu, power, dispersion):
        ll = np.zeros_like(y)
        ix = y > 0

        kappa = (mu[ix] ** (2 - power)) / (2 - power)
        theta = (mu[ix] ** (1 - power)) / (1 - power)
        normalization = scipy_based_normalization(y[ix], power, dispersion)

        ll[~ix] = -(mu[~ix] ** (2 - power)) / (dispersion * (2 - power))
        ll[ix] = (theta * y[ix] - kappa) / dispersion + normalization

        return ll.sum()

    y = np.arange(0, 100, step=0.5, dtype="float")
    mu = np.full_like(y, y.mean())

    candidate = TweedieDistribution(power).log_likelihood(y, mu, dispersion=dispersion)
    target = scipy_based_loglihood(y, mu, power, dispersion)

    np.testing.assert_allclose(candidate, target, rtol=1e-6)
