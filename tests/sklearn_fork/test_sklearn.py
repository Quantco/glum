# Authors: Christian Lorentzen <lorentzen.ch@gmail.com>
#
# License: BSD 3 clause
import copy
from typing import Tuple, Union

import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize, sparse
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.utils.estimator_checks import check_estimator

import quantcore.glm.matrix as mx
from quantcore.glm.sklearn_fork import GeneralizedLinearRegressorCV
from quantcore.glm.sklearn_fork._distribution import guess_intercept
from quantcore.glm.sklearn_fork._glm import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    GeneralizedLinearRegressor,
    IdentityLink,
    InverseGaussianDistribution,
    Link,
    LogitLink,
    LogLink,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
    _unstandardize,
    is_pos_semidef,
)
from quantcore.glm.sklearn_fork._util import _safe_sandwich_dot

GLM_SOLVERS = ["irls-ls", "lbfgs", "irls-cd"]

estimators = [
    (GeneralizedLinearRegressor, {}),
    (GeneralizedLinearRegressorCV, {"n_alphas": 2}),
]


def get_small_x_y(
    estimator: Union[GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(estimator, GeneralizedLinearRegressor):
        n_rows = 1
    else:
        n_rows = 10
    x = np.ones((n_rows, 1), dtype=int)
    y = np.ones(n_rows) * 0.5
    return x, y


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=107, n_features=10, n_informative=80, noise=0.5, random_state=2
    )
    return X, y


@pytest.fixture
def y():
    """In range of all distributions"""
    return np.array([0.1, 0.5])


@pytest.fixture
def X():
    return np.array([[1], [2]])


@pytest.mark.parametrize("link", Link.__subclasses__())
def test_link_properties(link):
    """Test link inverse and derivative."""
    rng = np.random.RandomState(42)
    x = rng.rand(100) * 100
    link = link()  # instantiate object
    if isinstance(link, LogitLink):
        # careful for large x, note expit(36) = 1
        # limit max eta to 15
        x = x / 100 * 15
    assert_allclose(link.link(link.inverse(x)), x)
    # if f(g(x)) = x, then f'(g(x)) = 1/g'(x)
    assert_allclose(link.derivative(link.inverse(x)), 1.0 / link.inverse_derivative(x))

    assert link.inverse_derivative2(x).shape == link.inverse_derivative(x).shape


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
    assert_array_equal(result, expected)


def test_tweedie_distribution_power():
    with pytest.raises(ValueError, match="no distribution exists"):
        TweedieDistribution(power=0.5)

    with pytest.raises(TypeError, match="must be a real number"):
        TweedieDistribution(power=1j)

    with pytest.raises(TypeError, match="must be a real number"):
        dist = TweedieDistribution()
        dist.power = 1j

    dist = TweedieDistribution()
    assert dist._include_lower_bound is False
    dist.power = 1
    assert dist._include_lower_bound is True


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
        assert_allclose(family.deviance(x, x), 0, atol=1e-9)


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
    ],
    ids=lambda args: args.__class__.__name__,
)
def test_gradients(family, link):
    np.random.seed(1001)
    for i in range(5):
        nrows = 100
        ncols = 10
        X = np.random.rand(nrows, ncols)
        coef = np.random.rand(ncols)
        y = np.random.rand(nrows)
        weights = np.ones(nrows)

        eta, mu, _ = family.eta_mu_loglikelihood(
            link, 1.0, np.zeros(nrows), X.dot(coef), y, weights
        )
        gradient_rows, _ = family.rowwise_gradient_hessian(
            link=link, coef=coef, phi=1.0, X=X, y=y, weights=weights, eta=eta, mu=mu,
        )
        score_analytic = gradient_rows @ X

        def f(coef2):
            _, _, ll = family.eta_mu_loglikelihood(
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
        assert_allclose(score_numeric, score_analytic, rtol=5e-5)


@pytest.mark.parametrize(
    "family, link",
    [
        (NormalDistribution(), IdentityLink()),
        (PoissonDistribution(), LogLink()),
        (GammaDistribution(), LogLink()),
        (InverseGaussianDistribution(), LogLink()),
        (TweedieDistribution(power=1.5), LogLink()),
        (TweedieDistribution(power=4.5), LogLink()),
    ],
    ids=lambda args: args.__class__.__name__,
)
def test_hessian_matrix(family, link):
    """Test the Hessian matrix numerically.
    Trick: Use numerical differentiation with y = mu"""
    coef = np.array([-2, 1, 0, 1, 2.5])
    phi = 0.5
    rng = np.random.RandomState(42)
    X = mx.DenseGLMDataMatrix(rng.randn(10, 5))
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
            gradient_rows, _ = family.rowwise_gradient_hessian(
                link=link,
                coef=coef,
                phi=phi,
                X=X,
                y=mu,
                weights=weights,
                eta=this_eta,
                mu=this_mu,
            )
            score = gradient_rows @ X
            return -score[i]

        approx = np.vstack(
            [approx, sp.optimize.approx_fprime(xk=coef, f=f, epsilon=1e-5)]
        )
    assert_allclose(hessian, approx, rtol=1e-3)


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_sample_weights_validation(estimator, kwargs):
    """Test the raised errors in the validation of sample_weight."""
    # scalar value but not positive
    X, y = get_small_x_y(estimator)
    weights = 0
    glm = estimator(fit_intercept=False, **kwargs)
    with pytest.raises(ValueError, match="weights must be non-negative"):
        glm.fit(X, y, weights)

    # Positive weights are accepted
    glm.fit(X, y, sample_weight=1)

    # 2d array
    weights = [[0]]
    with pytest.raises(ValueError, match="must be 1D array or scalar"):
        glm.fit(X, y, weights)

    # 1d but wrong length
    weights = [1, 0]
    with pytest.raises(ValueError, match="weights must have the same length as y"):
        glm.fit(X, y, weights)

    # 1d but only zeros (sum not greater than 0)
    weights = [0, 0]
    X = [[0], [1]]
    y = [1, 2]
    with pytest.raises(ValueError, match="must have at least one positive element"):
        glm.fit(X, y, weights)

    # 5. 1d but with a negative value
    weights = [2, -1]
    with pytest.raises(ValueError, match="weights must be non-negative"):
        glm.fit(X, y, weights)


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_offset_validation(estimator, kwargs):
    X, y = get_small_x_y(estimator)
    glm = estimator(fit_intercept=False, **kwargs)

    # Negatives are accepted (makes sense for log link)
    glm.fit(X, y, offset=-1)

    # Arrays of the right shape are accepted
    glm.fit(X, y, offset=y.copy())

    # 2d array
    with pytest.raises(ValueError, match="must be 1D array or scalar"):
        glm.fit(X, y, offset=np.zeros_like(X))

    # 1d but wrong length
    with pytest.raises(ValueError, match="must have the same length as y"):
        glm.fit(X, y, offset=[1, 0])


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
def test_tol_validation_errors(estimator):
    X, y = get_small_x_y(estimator)

    glm = estimator(gradient_tol=None, step_size_tol=None)
    with pytest.raises(ValueError, match="cannot both be None"):
        glm.fit(X, y)

    glm = estimator(gradient_tol=-0.1)
    with pytest.raises(ValueError, match="Tolerance for stopping"):
        glm.fit(X, y)

    glm = estimator(step_size_tol=-0.1)
    with pytest.raises(ValueError, match="Tolerance for stopping"):
        glm.fit(X, y)


@pytest.mark.parametrize("estimator, kwargs", estimators)
@pytest.mark.parametrize(
    "tol_kws",
    [
        {},
        {"step_size_tol": 1},
        {"step_size_tol": None},
        {"gradient_tol": 1},
        {"gradient_tol": None, "step_size_tol": 1},
        {"gradient_tol": 1, "step_size_tol": 1},
    ],
)
def test_tol_validation_no_error(estimator, kwargs, tol_kws):
    X, y = get_small_x_y(estimator)
    glm = estimator(**tol_kws, **kwargs)
    glm.fit(X, y)


# TODO: something for CV regressor
@pytest.mark.parametrize(
    "f, fam",
    [
        ("normal", NormalDistribution()),
        ("poisson", PoissonDistribution()),
        ("gamma", GammaDistribution()),
        ("inverse.gaussian", InverseGaussianDistribution()),
        ("binomial", BinomialDistribution()),
    ],
)
def test_glm_family_argument(f, fam, y, X):
    """Test GLM family argument set as string."""
    glm = GeneralizedLinearRegressor(family=f, alpha=0).fit(X, y)
    assert isinstance(glm._family_instance, fam.__class__)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
def test_glm_family_argument_invalid_input(estimator):
    X, y = get_small_x_y(estimator)
    glm = estimator(family="not a family", fit_intercept=False)
    with pytest.raises(ValueError, match="family must be"):
        glm.fit(X, y)


@pytest.mark.parametrize("estimator, kwargs", estimators)
@pytest.mark.parametrize("family", ExponentialDispersionModel.__subclasses__())
def test_glm_family_argument_as_exponential_dispersion_model(estimator, kwargs, family):
    X, y = get_small_x_y(estimator)
    glm = estimator(family=family(), **kwargs)
    glm.fit(X, y)


@pytest.mark.parametrize(
    "l, link",
    [("identity", IdentityLink()), ("log", LogLink()), ("logit", LogitLink())],
)
def test_glm_link_argument(l, link, y, X):
    """Test GLM link argument set as string."""
    glm = GeneralizedLinearRegressor(family="normal", link=l).fit(X, y)
    assert isinstance(glm._link_instance, link.__class__)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
def test_glm_link_argument_invalid_input(estimator):
    X, y = get_small_x_y(estimator)
    glm = estimator(family="normal", link="not a link")
    with pytest.raises(ValueError, match="link must be"):
        glm.fit(X, y)


@pytest.mark.parametrize("alpha", ["not a number", -4.2])
def test_glm_alpha_argument(alpha, y, X):
    """Test GLM for invalid alpha argument."""
    glm = GeneralizedLinearRegressor(family="normal", alpha=alpha)
    with pytest.raises(ValueError, match="Penalty term must be a non-negative"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("l1_ratio", ["not a number", -4.2, 1.1])
def test_glm_l1_ratio_argument(estimator, l1_ratio):
    """Test GLM for invalid l1_ratio argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(family="normal", l1_ratio=l1_ratio)
    with pytest.raises(ValueError, match="l1_ratio must be a number in interval.*0, 1"):
        glm.fit(X, y)


def test_glm_ratio_argument_array():
    X, y = get_small_x_y(GeneralizedLinearRegressor)
    glm = GeneralizedLinearRegressor(family="normal", l1_ratio=[1])
    with pytest.raises(ValueError, match="l1_ratio must be a number in interval.*0, 1"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("P1", [["a string", "a string"], [1, [2]], [1, 2, 3], [-1]])
def test_glm_P1_argument(estimator, P1, y, X):
    """Test GLM for invalid P1 argument."""
    glm = estimator(P1=P1, l1_ratio=0.5, check_input=True)
    with pytest.raises((ValueError, TypeError)):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize(
    "P2", ["a string", [1, 2, 3], [[2, 3]], sparse.csr_matrix([1, 2, 3]), [-1]]
)
def test_glm_P2_argument(estimator, P2, y, X):
    """Test GLM for invalid P2 argument."""
    glm = estimator(P2=P2, check_input=True)
    with pytest.raises(ValueError):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
def test_glm_P2_positive_semidefinite(estimator):
    """Test GLM for a positive semi-definite P2 argument."""
    n_samples, n_features = 10, 2
    y = np.arange(n_samples)
    X = np.zeros((n_samples, n_features))

    # negative definite matrix
    P2 = np.array([[1, 2], [2, 1]])
    glm = estimator(P2=P2, fit_intercept=False, check_input=True)
    with pytest.raises(ValueError, match="P2 must be positive semi-definite"):
        glm.fit(X, y)

    P2 = sparse.csr_matrix(P2)
    glm = estimator(P2=P2, fit_intercept=False, check_input=True)
    with pytest.raises(ValueError, match="P2 must be positive semi-definite"):
        glm.fit(X, y)


def test_positive_semidefinite():
    """Test GLM for a positive semi-definite P2 argument."""
    # negative definite matrix
    P2 = np.array([[1, 2], [2, 1]])
    assert not is_pos_semidef(P2)

    P2 = sparse.csr_matrix(P2)
    assert not is_pos_semidef(P2)

    assert is_pos_semidef(np.eye(2))
    assert is_pos_semidef(sparse.eye(2))


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("fit_intercept", ["not bool", 1, 0, [True]])
def test_glm_fit_intercept_argument(estimator, fit_intercept):
    """Test GLM for invalid fit_intercept argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="fit_intercept must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize(
    "solver, l1_ratio",
    [("not a solver", 0), (1, 0), ([1], 0), ("irls-ls", 0.5), ("lbfgs", 0.5)],
)
def test_glm_solver_argument(estimator, solver, l1_ratio, y, X):
    """Test GLM for invalid solver argument."""
    glm = estimator(solver=solver, l1_ratio=l1_ratio)
    with pytest.raises(ValueError):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("max_iter", ["not a number", 0, -1, 5.5, [1]])
def test_glm_max_iter_argument(estimator, max_iter):
    """Test GLM for invalid max_iter argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(max_iter=max_iter)
    with pytest.raises(ValueError, match="must be a positive integer"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("tol_param", ["gradient_tol", "step_size_tol"])
@pytest.mark.parametrize("tol", ["not a number", 0, -1.0, [1e-3]])
def test_glm_tol_argument(estimator, tol_param, tol):
    """Test GLM for invalid tol argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(**{tol_param: tol})
    with pytest.raises(ValueError, match="stopping criteria must be positive"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("warm_start", ["not bool", 1, 0, [True]])
def test_glm_warm_start_argument(estimator, warm_start):
    """Test GLM for invalid warm_start argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(warm_start=warm_start)
    with pytest.raises(ValueError, match="warm_start must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize(
    "start_params", ["not a start_params", ["zero"], [0, 0, 0], [[0, 0]], ["a", "b"]]
)
def test_glm_start_params_argument(estimator, start_params, y, X):
    """Test GLM for invalid start_params argument."""
    glm = estimator(start_params=start_params)
    with pytest.raises(ValueError):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("selection", ["not a selection", 1, 0, ["cyclic"]])
def test_glm_selection_argument(estimator, selection):
    """Test GLM for invalid selection argument"""
    X, y = get_small_x_y(estimator)
    glm = estimator(selection=selection)
    with pytest.raises(ValueError, match="argument selection must be"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("random_state", ["a string", 0.5, [0]])
def test_glm_random_state_argument(estimator, random_state):
    """Test GLM for invalid random_state argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(random_state=random_state)
    with pytest.raises(ValueError, match="cannot be used to seed"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("copy_X", ["not bool", 1, 0, [True]])
def test_glm_copy_X_argument(estimator, copy_X):
    """Test GLM for invalid copy_X arguments."""
    X, y = get_small_x_y(estimator)
    glm = estimator(copy_X=copy_X)
    with pytest.raises(ValueError, match="copy_X must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("check_input", ["not bool", 1, 0, [True]])
def test_glm_check_input_argument(estimator, check_input):
    """Test GLM for invalid check_input argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(check_input=check_input)
    with pytest.raises(ValueError, match="check_input must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("offset", [None, np.array([-0.1, 0, 0.1, 0, -0.2]), 0.1])
@pytest.mark.parametrize(
    "convert_x_fn",
    [
        np.asarray,
        sparse.csc_matrix,
        sparse.csr_matrix,
        mx.DenseGLMDataMatrix,
        lambda x: mx.MKLSparseMatrix(sparse.csc_matrix(x)),
        lambda x: mx.SplitMatrix(sparse.csc_matrix(x)),
    ],
)
def test_glm_identity_regression(solver, fit_intercept, offset, convert_x_fn):
    """Test GLM regression with identity link on a simple dataset."""
    coef = [1.0, 2.0]
    X = np.array([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4]]).T
    y = np.dot(X, coef) + (0 if offset is None else offset)
    glm = GeneralizedLinearRegressor(
        alpha=0,
        family="normal",
        link="identity",
        fit_intercept=fit_intercept,
        solver=solver,
        gradient_tol=1e-7,
    )
    if fit_intercept:
        X = X[:, 1:]

    X = convert_x_fn(X.astype(float))
    res = glm.fit(X, y, offset=offset)
    if fit_intercept:
        fit_coef = np.concatenate([[res.intercept_], res.coef_])
    else:
        fit_coef = res.coef_
    assert fit_coef.dtype.itemsize == X.dtype.itemsize
    assert_allclose(fit_coef, coef, rtol=1e-6)


@pytest.mark.parametrize(
    "family",
    [
        NormalDistribution(),
        PoissonDistribution(),
        GammaDistribution(),
        InverseGaussianDistribution(),
        TweedieDistribution(power=1.5),
        TweedieDistribution(power=4.5),
        GeneralizedHyperbolicSecant(),
    ],
)
@pytest.mark.parametrize(
    "solver, tol", [("irls-ls", 1e-6), ("lbfgs", 1e-7), ("irls-cd", 1e-7)]
)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("offset", [None, np.array([-0.1, 0, 0.1, 0, -0.2]), 0.1])
def test_glm_log_regression(family, solver, tol, fit_intercept, offset):
    """Test GLM regression with log link on a simple dataset."""
    coef = [0.2, -0.1]
    X = np.array([[1, 1, 1, 1, 1], [0, 1, 2, 3, 4]]).T
    y = np.exp(np.dot(X, coef) + (0 if offset is None else offset))
    glm = GeneralizedLinearRegressor(
        alpha=0,
        family=family,
        link="log",
        fit_intercept=fit_intercept,
        solver=solver,
        gradient_tol=tol,
    )
    if fit_intercept:
        X = X[:, 1:]
    res = glm.fit(X, y, offset=offset)
    if fit_intercept:
        fit_coef = np.concatenate([np.atleast_1d(res.intercept_), res.coef_])
    else:
        fit_coef = res.coef_
    assert_allclose(fit_coef, coef, rtol=8e-6)


@pytest.mark.filterwarnings("ignore:The line search algorithm")
@pytest.mark.filterwarnings("ignore:Line Search failed")
@pytest.mark.parametrize("n_samples, n_features", [(100, 10), (10, 100)])
@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("use_offset", [False, True])
def test_normal_ridge_comparison(n_samples, n_features, solver, use_offset):
    """Test ridge regression for Normal distributions.

    Case n_samples >> n_features

    Compare to test_ridge in test_ridge.py.
    """
    alpha = 1.0
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
    if use_offset:
        np.random.seed(0)
        offset = np.random.randn(n_samples)
        y += offset
    else:
        offset = None

    if n_samples > n_features:
        ridge_params = {"solver": "svd"}
    else:
        ridge_params = {"solver": "sag", "max_iter": 10000, "tol": 1e-9}

    # GLM has 1/(2*n) * Loss + 1/2*L2, Ridge has Loss + L2
    ridge = Ridge(
        alpha=alpha * n_samples, normalize=False, random_state=42, **ridge_params
    )
    ridge.fit(X, y if offset is None else y - offset)

    glm = GeneralizedLinearRegressor(
        alpha=1.0,
        l1_ratio=0,
        family="normal",
        fit_intercept=True,
        max_iter=300,
        solver=solver,
        gradient_tol=1e-6,
        check_input=False,
        random_state=42,
    )
    glm.fit(X, y, offset=offset)
    assert glm.coef_.shape == (X.shape[1],)
    assert_allclose(glm.coef_, ridge.coef_, rtol=5e-5)
    assert_allclose(glm.intercept_, ridge.intercept_, rtol=1e-5)
    assert_allclose(glm.predict(T), ridge.predict(T), rtol=1e-4)


@pytest.mark.parametrize(
    "solver, tol", [("irls-ls", 1e-7), ("lbfgs", 1e-7), ("irls-cd", 1e-7)]
)
@pytest.mark.parametrize("scale_predictors", [True, False])
@pytest.mark.parametrize("use_sparse", [True, False])
def test_poisson_ridge(solver, tol, scale_predictors, use_sparse):
    """Test ridge regression with poisson family and LogLink.

    Compare to R's glmnet"""
    # library("glmnet")
    # options(digits=10)
    # df <- data.frame(a=c(-2,-1,1,2), b=c(0,0,1,1), y=c(0,1,1,2))
    # x <- data.matrix(df[,c("a", "b")])
    # y <- df$y
    # fit <- glmnet(x=x, y=y, alpha=0, intercept=T, family="poisson",
    #               standardize=F, thresh=1e-10, nlambda=10000)
    # coef(fit, s=1)
    # (Intercept) -0.12889386979
    # a            0.29019207995
    # b            0.03741173122
    #
    # fit <- glmnet(x=x, y=y, alpha=0, intercept=T, family="poisson",
    #               standardize=T, thresh=1e-10, nlambda=10000)
    # coef(fit, s=1)
    # (Intercept) -0.21002571120839675
    # a            0.16472093,
    # b            0.27051971

    # Alternately, for running from Python:
    # from glmnet_python import glmnet
    # model = glmnet(x=X_dense, y=y, alpha=0, family="poisson",
    #               standardize=scale_predictors, thresh=1e-10, lambdau=np.array([1.0]))
    # true_intercept = model["a0"][0]
    # true_beta = model["beta"][:, 0]
    # print(true_intercept, true_beta)

    X_dense = np.array([[-2, -1, 1, 2], [0, 0, 1, 1]], dtype=np.float).T
    if use_sparse:
        X = sparse.csc_matrix(X_dense)
    else:
        X = X_dense
    y = np.array([0, 1, 1, 2], dtype=np.float)
    rng = np.random.RandomState(42)
    glm = GeneralizedLinearRegressor(
        alpha=1,
        l1_ratio=0,
        fit_intercept=True,
        family="poisson",
        link="log",
        gradient_tol=1e-7,
        solver=solver,
        max_iter=300,
        random_state=rng,
        copy_X=True,
        scale_predictors=scale_predictors,
    )

    glm2 = copy.deepcopy(glm)

    def check(G):
        G.fit(X, y)
        if scale_predictors:
            assert_allclose(G.intercept_, -0.21002571120839675, rtol=1e-5)
            assert_allclose(G.coef_, [0.16472093, 0.27051971], rtol=1e-5)
        else:
            assert_allclose(G.intercept_, -0.12889386979, rtol=1e-5)
            assert_allclose(G.coef_, [0.29019207995, 0.03741173122], rtol=1e-5)

    check(glm)

    # Test warm starting a re-fit model.
    glm.warm_start = True
    check(glm)
    assert glm.n_iter_ <= 1

    # Test warm starting with start_params.
    glm2.warm_start = True
    glm2.start_params = np.concatenate(([glm.intercept_], glm.coef_))
    check(glm2)
    assert glm2.n_iter_ <= 1


def test_normal_enet():
    """Test elastic net regression with normal/gaussian family."""
    alpha, l1_ratio = 0.3, 0.7
    n_samples, n_features = 20, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).copy(order="F")
    beta = rng.randn(n_features)
    y = 2 + np.dot(X, beta) + rng.randn(n_samples)

    # 1. test normal enet on dense data
    glm = GeneralizedLinearRegressor(
        alpha=alpha,
        l1_ratio=l1_ratio,
        family="normal",
        link="identity",
        fit_intercept=True,
        gradient_tol=1e-8,
        max_iter=100,
        selection="cyclic",
        solver="irls-cd",
        check_input=False,
    )
    glm.fit(X, y)

    enet = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        normalize=False,
        tol=1e-8,
        copy_X=True,
    )
    enet.fit(X, y)

    assert_allclose(glm.intercept_, enet.intercept_, rtol=2e-7)
    assert_allclose(glm.coef_, enet.coef_, rtol=5e-5)

    # 2. test normal enet on sparse data
    X = sparse.csc_matrix(X)
    glm.fit(X, y)
    assert_allclose(glm.intercept_, enet.intercept_, rtol=2e-7)
    assert_allclose(glm.coef_, enet.coef_, rtol=5e-5)


def test_poisson_enet():
    """Test elastic net regression with poisson family and LogLink.

    Compare to R's glmnet"""
    # library("glmnet")
    # options(digits=10)
    # df <- data.frame(a=c(-2,-1,1,2), b=c(0,0,1,1), y=c(0,1,1,2))
    # x <- data.matrix(df[,c("a", "b")])
    # y <- df$y
    # fit <- glmnet(x=x, y=y, alpha=0.5, intercept=T, family="poisson",
    #               standardize=F, thresh=1e-10, nlambda=10000)
    # coef(fit, s=1)
    # (Intercept) -0.03550978409
    # a            0.16936423283
    # b            .
    glmnet_intercept = -0.03550978409
    glmnet_coef = [0.16936423283, 0.0]
    X = np.array([[-2, -1, 1, 2], [0, 0, 1, 1]]).T
    y = np.array([0, 1, 1, 2])
    rng = np.random.RandomState(42)
    glm = GeneralizedLinearRegressor(
        alpha=1,
        l1_ratio=0.5,
        family="poisson",
        link="log",
        solver="irls-cd",
        gradient_tol=1e-8,
        selection="random",
        random_state=rng,
    )
    glm.fit(X, y)
    assert_allclose(glm.intercept_, glmnet_intercept, rtol=2e-6)
    assert_allclose(glm.coef_, glmnet_coef, rtol=2e-7)

    # test results with general optimization procedure
    def obj(coef):
        pd = PoissonDistribution()
        link = LogLink()
        N = y.shape[0]
        mu = link.inverse(X @ coef[1:] + coef[0])
        alpha, l1_ratio = (1, 0.5)
        return (
            1.0 / (2.0 * N) * pd.deviance(y, mu)
            + 0.5 * alpha * (1 - l1_ratio) * (coef[1:] ** 2).sum()
            + alpha * l1_ratio * np.sum(np.abs(coef[1:]))
        )

    res = optimize.minimize(
        obj,
        [0, 0, 0],
        method="nelder-mead",
        tol=1e-10,
        options={"maxiter": 1000, "disp": False},
    )
    assert_allclose(glm.intercept_, res.x[0], rtol=5e-5)
    assert_allclose(glm.coef_, res.x[1:], rtol=1e-5, atol=1e-9)
    assert_allclose(
        obj(np.concatenate(([glm.intercept_], glm.coef_))), res.fun, rtol=1e-8
    )

    # same for start_params='zero' and selection='cyclic'
    # with reduced precision
    glm = GeneralizedLinearRegressor(
        alpha=1,
        l1_ratio=0.5,
        family="poisson",
        link="log",
        solver="irls-cd",
        gradient_tol=1e-5,
        selection="cyclic",
    )
    glm.fit(X, y)
    assert_allclose(glm.intercept_, glmnet_intercept, rtol=1e-4)
    assert_allclose(glm.coef_, glmnet_coef, rtol=1e-4)

    # check warm_start, therefore start with different alpha
    glm = GeneralizedLinearRegressor(
        alpha=0.005,
        l1_ratio=0.5,
        family="poisson",
        max_iter=300,
        link="log",
        solver="irls-cd",
        gradient_tol=1e-5,
        selection="cyclic",
    )
    glm.fit(X, y)
    # warm start with original alpha and use of sparse matrices
    glm.warm_start = True
    glm.alpha = 1
    X = sparse.csr_matrix(X)
    glm.fit(X, y)
    assert_allclose(glm.intercept_, glmnet_intercept, rtol=1e-4)
    assert_allclose(glm.coef_, glmnet_coef, rtol=1e-4)


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1, 10])
def test_binomial_enet(alpha):
    """Test elastic net regression with binomial family and LogitLink.

    Compare to LogisticRegression.
    """
    l1_ratio = 0.5
    n_samples = 500
    rng = np.random.RandomState(42)
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        random_state=rng,
    )
    log = LogisticRegression(
        penalty="elasticnet",
        random_state=rng,
        fit_intercept=False,
        tol=1e-7,
        max_iter=1000,
        l1_ratio=l1_ratio,
        C=1.0 / (n_samples * alpha),
        solver="saga",
    )
    log.fit(X, y)

    glm = GeneralizedLinearRegressor(
        family=BinomialDistribution(),
        link=LogitLink(),
        fit_intercept=False,
        alpha=alpha,
        l1_ratio=l1_ratio,
        solver="irls-cd",
        selection="cyclic",
        gradient_tol=1e-7,
    )
    glm.fit(X, y)
    assert_allclose(log.intercept_[0], glm.intercept_, rtol=1e-6)
    assert_allclose(log.coef_[0, :], glm.coef_, rtol=5.1e-6)


@pytest.mark.parametrize(
    "params",
    [
        {"solver": "irls-ls"},
        {"solver": "lbfgs"},
        {"solver": "irls-cd", "selection": "cyclic"},
        {"solver": "irls-cd", "selection": "random"},
    ],
    ids=lambda params: ", ".join(
        "{}={}".format(key, val) for key, val in params.items()
    ),
)
@pytest.mark.parametrize("use_offset", [False, True])
def test_solver_equivalence(params, use_offset, regression_data):
    X, y = regression_data
    if use_offset:
        np.random.seed(0)
        offset = np.random.random(len(y))
    else:
        offset = None
    est_ref = GeneralizedLinearRegressor(random_state=2)
    est_ref.fit(X, y, offset=offset)

    est_2 = GeneralizedLinearRegressor(**params)
    est_2.set_params(random_state=2)

    est_2.fit(X, y, offset=offset)

    assert_allclose(est_2.intercept_, est_ref.intercept_, rtol=1e-4)
    assert_allclose(est_2.coef_, est_ref.coef_, rtol=1e-4)
    assert_allclose(
        mean_absolute_error(est_2.predict(X), y),
        mean_absolute_error(est_ref.predict(X), y),
        rtol=1e-4,
    )


# TODO: different distributions
# Specify rtol since some are more accurate than others
@pytest.mark.parametrize(
    "params",
    [
        {"solver": "irls-ls", "rtol": 1e-6},
        {"solver": "lbfgs", "rtol": 2e-4},
        {"solver": "irls-cd", "selection": "cyclic", "rtol": 2e-5},
        {"solver": "irls-cd", "selection": "random", "rtol": 6e-5},
    ],
    ids=lambda params: ", ".join(
        "{}={}".format(key, val) for key, val in params.items()
    ),
)
@pytest.mark.parametrize("use_offset", [False, True])
def test_solver_equivalence_cv(params, use_offset):
    n_alphas = 3
    n_samples = 100
    n_features = 10
    gradient_tol = 1e-5

    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=2)
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
        return assert_allclose(x, y, rtol=params["rtol"], atol=1e-7)

    _assert_all_close(est_2.alphas_, est_ref.alphas_)
    _assert_all_close(est_2.alpha_, est_ref.alpha_)
    _assert_all_close(est_2.l1_ratio_, est_ref.l1_ratio_)
    _assert_all_close(est_2.coef_path_, est_ref.coef_path_)
    _assert_all_close(est_2.deviance_path_, est_ref.deviance_path_)
    _assert_all_close(est_2.intercept_, est_ref.intercept_)
    _assert_all_close(est_2.coef_, est_ref.coef_)
    _assert_all_close(
        mean_absolute_error(est_2.predict(X), y),
        mean_absolute_error(est_ref.predict(X), y),
    )


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_fit_dispersion(estimator, kwargs, regression_data):
    X, y = regression_data

    est1 = estimator(random_state=2, **kwargs)
    est1.fit(X, y)
    assert not hasattr(est1, "dispersion_")

    est2 = estimator(random_state=2, fit_dispersion="chisqr", **kwargs)
    est2.fit(X, y)
    assert isinstance(est2.dispersion_, float)

    est3 = estimator(random_state=2, fit_dispersion="deviance", **kwargs)
    est3.fit(X, y)
    assert isinstance(est3.dispersion_, float)

    assert_allclose(est2.dispersion_, est3.dispersion_)


@pytest.mark.parametrize("solver", GLM_SOLVERS)
def test_convergence_warning(solver, regression_data):
    X, y = regression_data

    est = GeneralizedLinearRegressor(
        solver=solver, random_state=2, max_iter=1, gradient_tol=1e-20
    )
    with pytest.warns(ConvergenceWarning):
        est.fit(X, y)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("scale_predictors", [False, True])
def test_standardize(use_sparse, scale_predictors):
    def _arrays_share_data(arr1: np.ndarray, arr2: np.ndarray) -> bool:
        return arr1.__array_interface__["data"] == arr2.__array_interface__["data"]

    NR = 101
    NC = 10
    col_mults = np.arange(1, NC + 1)
    row_mults = np.linspace(0, 2, NR)
    M = row_mults[:, None] * col_mults[None, :]

    if use_sparse:
        M = mx.MKLSparseMatrix(sparse.csc_matrix(M))
    else:
        M = mx.DenseGLMDataMatrix(M)
    MC = copy.deepcopy(M)

    X, col_means, col_stds = M.standardize(np.ones(NR) / NR, scale_predictors)
    if use_sparse:
        assert _arrays_share_data(X.mat.data, M.data)
        assert _arrays_share_data(X.mat.indices, M.indices)
        assert _arrays_share_data(X.mat.indptr, M.indptr)
    else:
        # Check that the underlying data pointer is the same
        assert _arrays_share_data(X.mat, M)
    np.testing.assert_almost_equal(col_means, col_mults)

    # After standardization, all the columns will have the same values.
    # To check that, just convert to dense first.
    if use_sparse:
        Xdense = X.A
    else:
        Xdense = X
    for i in range(1, NC):
        if scale_predictors:
            if isinstance(Xdense, mx.ColScaledMat):
                one, two = Xdense.A[:, 0], Xdense.A[:, i]
            else:
                one, two = Xdense[:, 0], Xdense[:, i]
        else:
            if isinstance(Xdense, mx.ColScaledMat):
                one, two = (i + 1) * Xdense.A[:, 0], Xdense.A[:, i]
            else:
                one, two = (i + 1) * Xdense[:, 0], Xdense[:, i]
        np.testing.assert_almost_equal(one, two)

    if scale_predictors:
        # The sample variance of row_mults is 0.34. This is scaled up by the col_mults
        true_std = np.sqrt(0.34)
        np.testing.assert_almost_equal(col_stds, true_std * col_mults)

    intercept_standardized = 0.0
    coef_standardized = (
        np.ones_like(col_means) if col_stds is None else copy.copy(col_stds)
    )
    X2, intercept, coef = _unstandardize(
        X, col_means, col_stds, intercept_standardized, coef_standardized,
    )
    if use_sparse:
        assert _arrays_share_data(X2.data, X.mat.data)
    else:
        assert _arrays_share_data(X2, X.mat)
    np.testing.assert_almost_equal(intercept, -(NC + 1) * NC / 2)
    if scale_predictors:
        np.testing.assert_almost_equal(coef, 1.0)

    if use_sparse:
        assert type(X.mat) in [sparse.csc_matrix, mx.MKLSparseMatrix]
        assert type(X2) in [sparse.csc_matrix, mx.MKLSparseMatrix]
        np.testing.assert_almost_equal(MC.toarray(), X2.toarray())
    else:
        np.testing.assert_almost_equal(MC, X2)


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_check_estimator(estimator, kwargs):
    check_estimator(estimator(**kwargs))


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV],
)
def test_clonable(estimator):
    clone(estimator())


@pytest.mark.parametrize(
    "link, distribution, tol",
    [
        (IdentityLink(), NormalDistribution(), 1e-4),
        (LogLink(), PoissonDistribution(), 1e-4),
        (LogLink(), GammaDistribution(), 1e-4),
        (LogLink(), TweedieDistribution(1.5), 1e-4),
        (LogLink(), TweedieDistribution(4.5), 1e-4),
        (LogLink(), NormalDistribution(), 1e-4),
        (LogLink(), InverseGaussianDistribution(), 1e-4),
        (LogitLink(), BinomialDistribution(), 1e-2),
        (IdentityLink(), GeneralizedHyperbolicSecant(), 1e-1),
    ],
)
@pytest.mark.parametrize("offset", [None, np.array([0.3, -0.1, 0, 0.1]), 0.1])
def test_get_best_intercept(
    link: Link, distribution: ExponentialDispersionModel, tol: float, offset
):
    y = np.array([1, 1, 1, 2], dtype=np.float)
    if isinstance(distribution, BinomialDistribution):
        y -= 1

    weights = np.array([0.1, 0.2, 5, 1])
    best_intercept = guess_intercept(y, weights, link, distribution, offset)
    assert np.isfinite(best_intercept)

    def _get_dev(intercept):
        eta = intercept if offset is None else offset + intercept
        mu = link.inverse(eta)
        assert np.isfinite(mu).all()
        return distribution.deviance(y, mu, weights)

    obj = _get_dev(best_intercept)
    obj_low = _get_dev(best_intercept - tol)
    obj_high = _get_dev(best_intercept + tol)
    assert obj < obj_low
    assert obj < obj_high


@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6])
def test_step_size_tolerance(tol):
    X, y = make_regression(n_samples=100, n_features=5, noise=0.5, random_state=42,)
    y[y < 0] = 0

    def build_glm(step_size_tol):
        glm = GeneralizedLinearRegressor(
            alpha=1,
            l1_ratio=0.5,
            family="poisson",
            solver="irls-cd",
            gradient_tol=1e-10,
            step_size_tol=step_size_tol,
            selection="cyclic",
        )
        glm.fit(X, y)
        return glm

    baseline = build_glm(1e-10)
    glm = build_glm(tol)
    assert_allclose(baseline.intercept_, glm.intercept_, atol=tol)
    assert_allclose(baseline.coef_, glm.coef_, atol=tol)


def test_alpha_search(regression_data):
    X, y = regression_data
    mdl_no_path = GeneralizedLinearRegressor(
        alpha=0.001, l1_ratio=1, family="normal", link="identity", gradient_tol=1e-10,
    )
    mdl_no_path.fit(X=X, y=y)

    mdl_path = GeneralizedLinearRegressor(
        alpha_search=True,
        min_alpha=0.001,
        n_alphas=5,
        l1_ratio=1,
        family="normal",
        link="identity",
        gradient_tol=1e-10,
    )
    mdl_path.fit(X=X, y=y)

    assert_allclose(mdl_path.coef_, mdl_no_path.coef_)
    assert_allclose(mdl_path.intercept_, mdl_no_path.intercept_)
