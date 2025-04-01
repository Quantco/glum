from typing import Any, Union

import numpy as np
import pandas as pd
import pytest
import sklearn as skl
import sklearn.utils.estimator_checks
from scipy import sparse

from glum._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    InverseGaussianDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
)
from glum._glm import GeneralizedLinearRegressor
from glum._glm_cv import GeneralizedLinearRegressorCV
from glum._linalg import is_pos_semidef
from glum._link import IdentityLink, LogitLink, LogLink

GLM_SOLVERS = ["irls-ls", "lbfgs", "irls-cd", "trust-constr"]

estimators = [
    (GeneralizedLinearRegressor, {"alpha": 1.0}),
    (GeneralizedLinearRegressorCV, {"n_alphas": 2}),
]


def get_small_x_y(
    estimator: Union[GeneralizedLinearRegressor, GeneralizedLinearRegressorCV],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(estimator, GeneralizedLinearRegressor):
        n_rows = 2
    else:
        n_rows = 10
    x = np.ones((n_rows, 1), dtype=int)
    y = np.array([0, 1] * (n_rows // 2)) * 0.5
    return x, y


@pytest.fixture(scope="module")
def regression_data():
    X, y = skl.datasets.make_regression(
        n_samples=107, n_features=10, n_informative=80, noise=0.5, random_state=2
    )
    return X, y


@pytest.fixture
def y():
    """Get values for y that are in range of all distributions."""
    return np.array([0.1, 0.5])


@pytest.fixture
def X():
    return np.array([[1], [2]])


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_sample_weights_validation(estimator, kwargs):
    """Test the raised errors in the validation of sample_weight."""
    # scalar value but not positive
    X, y = get_small_x_y(estimator)
    sample_weight: Any = 0
    glm = estimator(fit_intercept=False, **kwargs)
    with pytest.raises(ValueError, match="weights must be non-negative"):
        glm.fit(X, y, sample_weight)

    # Positive weights are accepted
    glm.fit(X, y, sample_weight=1)

    # 2d array
    sample_weight = [[0]]
    with pytest.raises(ValueError, match="must be 1D array or scalar"):
        glm.fit(X, y, sample_weight)

    # 1d but wrong length
    sample_weight = [1, 0]
    with pytest.raises(ValueError, match="weights must have the same length as y"):
        glm.fit(X, y, sample_weight)

    # 1d but only zeros (sum not greater than 0)
    sample_weight = [0, 0]
    X = [[0], [1]]
    y = [1, 2]
    with pytest.raises(ValueError, match="must have at least one positive element"):
        glm.fit(X, y, sample_weight)

    # 5. 1d but with a negative value
    sample_weight = [2, -1]
    with pytest.raises(ValueError, match="weights must be non-negative"):
        glm.fit(X, y, sample_weight)


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

    glm = estimator(gradient_tol=-0.1)
    with pytest.raises(ValueError, match="Tolerance for the gradient stopping"):
        glm.fit(X, y)

    glm = estimator(step_size_tol=-0.1)
    with pytest.raises(ValueError, match="Tolerance for the step-size stopping"):
        glm.fit(X, y)


@pytest.mark.parametrize("estimator, kwargs", estimators)
@pytest.mark.parametrize(
    "tol_kws",
    [
        {},
        {"step_size_tol": 1},
        {"step_size_tol": None},
        {"gradient_tol": 1},
        {"gradient_tol": 1, "step_size_tol": 1},
    ],
)
def test_tol_validation_no_error(estimator, kwargs, tol_kws):
    X, y = get_small_x_y(estimator)
    glm = estimator(**tol_kws, **kwargs)
    glm.fit(X, y)


@pytest.mark.parametrize("estimator, kwargs", estimators)
@pytest.mark.parametrize("solver", ["auto", "irls-cd", "trust-constr"])
@pytest.mark.parametrize("gradient_tol", [None, 1])
def test_gradient_tol_setting(estimator, kwargs, solver, gradient_tol):
    X, y = get_small_x_y(estimator)
    glm = estimator(solver=solver, gradient_tol=gradient_tol, **kwargs)
    glm.fit(X, y)

    if gradient_tol is None:
        if solver == "trust-constr":
            gradient_tol = 1e-8
        else:
            gradient_tol = 1e-4

    np.testing.assert_allclose(gradient_tol, glm._gradient_tol)


# TODO: something for CV regressor
@pytest.mark.parametrize(
    "f, fam",
    [
        ("gaussian", NormalDistribution()),
        ("normal", NormalDistribution()),
        ("poisson", PoissonDistribution()),
        ("gamma", GammaDistribution()),
        ("inverse.gaussian", InverseGaussianDistribution()),
        ("binomial", BinomialDistribution()),
        ("negative.binomial", NegativeBinomialDistribution()),
    ],
)
def test_glm_family_argument(f, fam, y, X):
    """Test GLM family argument set as string."""
    glm = GeneralizedLinearRegressor(family=f).fit(X, y)
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
    glm.fit(X, np.where(y > family().lower_bound, y, y.max() / 2))


@pytest.mark.parametrize(
    "link_func, link",
    [("identity", IdentityLink()), ("log", LogLink()), ("logit", LogitLink())],
)
def test_glm_link_argument(link_func, link, y, X):
    """Test GLM link argument set as string."""
    glm = GeneralizedLinearRegressor(family="normal", link=link_func).fit(X, y)
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


def test_P1_P2_expansion_with_categoricals():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        data={
            "dense": np.linspace(0, 10, 60),
            "cat": pd.Categorical(rng.integers(5, size=60)),
        }
    )
    y = rng.normal(size=60)

    mdl1 = GeneralizedLinearRegressor(
        l1_ratio=0.01,
        P1=[1, 2, 2, 2, 2, 2],
        P2=[2, 1, 1, 1, 1, 1],
    )
    mdl1.fit(X, y)

    mdl2 = GeneralizedLinearRegressor(
        l1_ratio=0.01,
        P1=[1, 2],
        P2=[2, 1],
    )
    mdl2.fit(X, y)
    np.testing.assert_allclose(mdl1.coef_, mdl2.coef_)

    mdl2 = GeneralizedLinearRegressor(
        l1_ratio=0.01, P1=[1, 2], P2=sparse.diags([2, 1, 1, 1, 1, 1])
    )
    mdl2.fit(X, y)
    np.testing.assert_allclose(mdl1.coef_, mdl2.coef_)


def test_P1_P2_expansion_with_categoricals_missings():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        data={
            "dense": np.linspace(0, 10, 60),
            "cat": pd.Categorical(rng.integers(5, size=60)).remove_categories(0),
        }
    )
    y = rng.normal(size=60)

    mdl1 = GeneralizedLinearRegressor(
        alpha=1.0,
        l1_ratio=0.01,
        P1=[1, 2, 2, 2, 2, 2],
        P2=[2, 1, 1, 1, 1, 1],
        cat_missing_method="convert",
    )
    mdl1.fit(X, y)

    mdl2 = GeneralizedLinearRegressor(
        alpha=1.0,
        l1_ratio=0.01,
        P1=[1, 2],
        P2=[2, 1],
        cat_missing_method="convert",
    )
    mdl2.fit(X, y)
    np.testing.assert_allclose(mdl1.coef_, mdl2.coef_)

    mdl3 = GeneralizedLinearRegressor(
        alpha=1.0,
        l1_ratio=0.01,
        P1=[1, 2],
        P2=sparse.diags([2, 1, 1, 1, 1, 1]),
        cat_missing_method="convert",
    )
    mdl3.fit(X, y)
    np.testing.assert_allclose(mdl1.coef_, mdl3.coef_)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("fit_intercept", ["not bool", 1, 0, [True]])
def test_glm_fit_intercept_argument(estimator, fit_intercept):
    """Test GLM for invalid fit_intercept argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(fit_intercept=fit_intercept)
    with pytest.raises(TypeError, match="fit_intercept must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize(
    "solver, l1_ratio",
    [
        ("not a solver", 0),
        (1, 0),
        ([1], 0),
        ("irls-ls", 0.5),
        ("lbfgs", 0.5),
        ("trust-constr", 0.5),
    ],
)
def test_glm_solver_argument(estimator, solver, l1_ratio, y, X):
    """Test GLM for invalid solver argument."""
    kwargs = {"solver": solver, "l1_ratio": l1_ratio}
    if estimator == GeneralizedLinearRegressor:
        kwargs["alpha"] = 1.0
    glm = estimator(**kwargs)
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
    with pytest.raises(TypeError, match="warm_start must be bool"):
        glm.fit(X, y)


# https://github.com/Quantco/glum/issues/645
@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
def test_glm_warm_start_with_constant_column(estimator):
    X, y = skl.datasets.make_regression()
    X[:, 0] = 0
    kwargs = {"warm_start": True}
    if estimator == GeneralizedLinearRegressor:
        kwargs["alpha"] = 1.0
    glm = estimator(**kwargs)
    glm.fit(X, y)
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
    """Test GLM for invalid selection argument."""
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
def test_glm_copy_X_argument_invalid(estimator, copy_X):
    """Test GLM for invalid copy_X arguments."""
    X, y = get_small_x_y(estimator)
    glm = estimator(copy_X=copy_X)
    with pytest.raises(TypeError, match="copy_X must be None or bool"):
        glm.fit(X, y)


def test_glm_copy_X_input_needs_conversion():
    y = np.array([1.0])
    # If X is of int dtype, it needs to be copied
    X = np.array([[1]])
    glm = GeneralizedLinearRegressor(copy_X=False)
    # should raise an error
    with pytest.raises(ValueError, match="copy_X"):
        glm.fit(X, y)
    # should be OK with copy_X = None or copy_X = True
    GeneralizedLinearRegressor(copy_X=None).fit(X, y)
    GeneralizedLinearRegressor(copy_X=True).fit(X, y)


@pytest.mark.parametrize(
    "estimator", [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV]
)
@pytest.mark.parametrize("check_input", ["not bool", 1, 0, [True]])
def test_glm_check_input_argument(estimator, check_input):
    """Test GLM for invalid check_input argument."""
    X, y = get_small_x_y(estimator)
    glm = estimator(check_input=check_input)
    with pytest.raises(TypeError, match="check_input must be bool"):
        glm.fit(X, y)


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_check_estimator(estimator, kwargs):
    sklearn.utils.estimator_checks.check_estimator(estimator(**kwargs))


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV],
)
def test_clonable(estimator):
    skl.base.clone(estimator())
