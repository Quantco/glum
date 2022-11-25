# Authors: Christian Lorentzen <lorentzen.ch@gmail.com>
#
# License: BSD 3 clause
import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import tabmat as tm
from numpy.testing import assert_allclose
from scipy import optimize, sparse
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.utils.estimator_checks import check_estimator
from statsmodels.tools import eval_measures

from glum import GeneralizedLinearRegressorCV
from glum._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
    guess_intercept,
)
from glum._glm import GeneralizedLinearRegressor, _unstandardize, is_pos_semidef
from glum._link import IdentityLink, Link, LogitLink, LogLink

GLM_SOLVERS = ["irls-ls", "lbfgs", "irls-cd", "trust-constr"]

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
    with pytest.raises(TypeError, match="warm_start must be bool"):
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


@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("offset", [None, np.array([-0.1, 0, 0.1, 0, -0.2]), 0.1])
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


@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("full_report", [False, True])
@pytest.mark.parametrize("custom_columns", [None, ["objective_fct"]])
def test_get_diagnostics(
    solver, fit_intercept: bool, full_report: bool, custom_columns: Optional[List[str]]
):
    """Test GLM regression with identity link on a simple dataset."""
    X, y = get_small_x_y(GeneralizedLinearRegressor)

    glm = GeneralizedLinearRegressor(fit_intercept=fit_intercept, solver=solver)
    res = glm.fit(X, y)

    diagnostics = res.get_formatted_diagnostics(full_report, custom_columns)
    if solver in ("lbfgs", "trust-constr"):
        assert diagnostics == "solver does not report diagnostics"
    else:
        assert diagnostics.index.name == "n_iter"
        if custom_columns is not None:
            expected_columns = custom_columns
        elif full_report:
            expected_columns = [
                "convergence",
                "objective_fct",
                "L1(coef)",
                "L2(coef)",
                "L2(step)",
                "first_coef",
                "n_coef_updated",
                "n_active_cols",
                "n_active_rows",
                "n_cycles",
                "n_line_search",
                "iteration_runtime",
                "build_hessian_runtime",
                "inner_solver_runtime",
                "line_search_runtime",
                "quadratic_update_runtime",
                "intercept",
            ]
        else:
            expected_columns = [
                "convergence",
                "n_cycles",
                "iteration_runtime",
                "intercept",
            ]
        assert (diagnostics.columns == expected_columns).all()
        if "intercept" in expected_columns:
            null_intercept = diagnostics["intercept"].isnull()
            if fit_intercept:
                assert not null_intercept.any()
            else:
                assert null_intercept.all()


@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("full_report", [False, True])
@pytest.mark.parametrize("custom_columns", [None, ["objective_fct"]])
def test_report_diagnostics(
    solver, fit_intercept: bool, full_report: bool, custom_columns: Optional[List[str]]
):
    """Test GLM regression with identity link on a simple dataset."""
    X, y = get_small_x_y(GeneralizedLinearRegressor)

    glm = GeneralizedLinearRegressor(fit_intercept=fit_intercept, solver=solver)
    res = glm.fit(X, y)

    # Make sure something prints
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        res.report_diagnostics(full_report, custom_columns)
    printed = f.getvalue()
    # Something should be printed
    assert len(printed) > 0


@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("offset", [None, np.array([-0.1, 0, 0.1, 0, -0.2]), 0.1])
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
def test_x_not_modified_inplace(solver, fit_intercept, offset, convert_x_fn):
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

    X_before = copy.deepcopy(X)
    glm.fit(X, y, offset=offset)
    if isinstance(X, np.ndarray):
        np.testing.assert_almost_equal(X, X_before)
    else:
        np.testing.assert_almost_equal(X.A, X_before.A)


@pytest.mark.parametrize("solver", GLM_SOLVERS)
@pytest.mark.parametrize("offset", [None, np.array([-0.1, 0, 0.1, 0, -0.2]), 0.1])
@pytest.mark.parametrize(
    "convert_x_fn",
    [
        np.asarray,
        sparse.csc_matrix,
        sparse.csr_matrix,
        lambda x: tm.DenseMatrix(x.astype(float)),
        lambda x: tm.SparseMatrix(sparse.csc_matrix(x)),
        lambda x: tm.from_csc(sparse.csc_matrix(x).astype(float)),
        lambda x: tm.CategoricalMatrix(x.dot([0, 1])),
    ],
)
def test_glm_identity_regression_categorical_data(solver, offset, convert_x_fn):
    """Test GLM regression with identity link on a simple dataset."""
    coef = [1.0, 2.0]
    x_vec = np.array([1, 0, 0, 1, 0])
    x_mat = np.stack([x_vec, 1 - x_vec]).T
    y = np.dot(x_mat, coef) + (0 if offset is None else offset)

    glm = GeneralizedLinearRegressor(
        alpha=0,
        family="normal",
        link="identity",
        fit_intercept=False,
        solver=solver,
        gradient_tol=1e-7,
    )
    X = convert_x_fn(x_mat)
    np.testing.assert_almost_equal(X.A if hasattr(X, "A") else X, x_mat)
    res = glm.fit(X, y, offset=offset)

    assert_allclose(res.coef_, coef, rtol=1e-6)


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
    "solver, tol",
    [("irls-ls", 1e-6), ("lbfgs", 1e-7), ("irls-cd", 1e-7), ("trust-constr", 1e-7)],
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
    X, y, _ = make_regression(
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
        ridge_params: Dict[str, Any] = {"solver": "svd"}
    else:
        ridge_params = {"solver": "sag", "max_iter": 10000, "tol": 1e-9}

    # GLM has 1/(2*n) * Loss + 1/2*L2, Ridge has Loss + L2
    ridge = Ridge(alpha=alpha * n_samples, random_state=42, **ridge_params)
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
    "solver, tol",
    [("irls-ls", 1e-7), ("lbfgs", 1e-7), ("irls-cd", 1e-7), ("trust-constr", 1e-8)],
)
@pytest.mark.parametrize("scale_predictors", [True, False])
@pytest.mark.parametrize("use_sparse", [True, False])
def test_poisson_ridge(solver, tol, scale_predictors, use_sparse):
    """Test ridge regression with poisson family and LogLink.

    Compare to R's glmnet
    """
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

    X_dense = np.array([[-2, -1, 1, 2], [0, 0, 1, 1]], dtype=np.float_).T
    if use_sparse:
        X = sparse.csc_matrix(X_dense)
    else:
        X = X_dense
    y = np.array([0, 1, 1, 2], dtype=np.float_)
    model_args = dict(
        alpha=1,
        l1_ratio=0,
        fit_intercept=True,
        family="poisson",
        link="log",
        gradient_tol=tol,
        solver=solver,
        max_iter=300,
        random_state=np.random.RandomState(42),
        copy_X=True,
        scale_predictors=scale_predictors,
    )
    glm = GeneralizedLinearRegressor(**model_args)
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


@pytest.mark.parametrize("scale_predictors", [True, False])
def test_poisson_ridge_bounded(scale_predictors):
    X = np.array([[-1, 1, 1, 2], [0, 0, 1, 1]], dtype=np.float_).T
    y = np.array([0, 1, 1, 2], dtype=np.float_)
    lb = np.array([-0.1, -0.1])
    ub = np.array([0.1, 0.1])

    # For comparison, this is the source of truth for the assert_allclose below.
    # from glmnet_python import glmnet
    # model = glmnet(x=X.copy(), y=y.copy(), alpha=0, family="poisson",
    #               standardize=scale_predictors, thresh=1e-10, lambdau=np.array([1.0]),
    #               cl = np.array([lb, ub])
    #               )
    # true_intercept = model["a0"][0]
    # true_beta = model["beta"][:, 0]
    # print(true_intercept, true_beta)

    glm = GeneralizedLinearRegressor(
        alpha=1,
        l1_ratio=0,
        fit_intercept=True,
        family="poisson",
        link="log",
        gradient_tol=1e-7,
        solver="irls-cd",
        max_iter=300,
        random_state=np.random.RandomState(42),
        copy_X=True,
        scale_predictors=scale_predictors,
        lower_bounds=lb,
        upper_bounds=ub,
    )
    glm.fit(X, y)

    # These correct values come from glmnet.
    assert_allclose(glm.intercept_, -0.13568186971946633, rtol=1e-5)
    assert_allclose(glm.coef_, [0.1, 0.1], rtol=1e-5)


@pytest.mark.parametrize("scale_predictors", [True, False])
def test_poisson_ridge_ineq_constrained(scale_predictors):
    X = np.array([[-1, 1, 1, 2], [0, 0, 1, 1]], dtype=np.float_).T
    y = np.array([0, 1, 1, 2], dtype=np.float_)
    A_ineq = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b_ineq = 0.1 * np.ones(shape=(4))

    # For comparison, this is the source of truth for the assert_allclose below.
    # from glmnet_python import glmnet
    # model = glmnet(x=X.copy(), y=y.copy(), alpha=0, family="poisson",
    #               standardize=scale_predictors, thresh=1e-10, lambdau=np.array([1.0]),
    #               cl = np.array([lb, ub])
    #               )
    # true_intercept = model["a0"][0]
    # true_beta = model["beta"][:, 0]
    # print(true_intercept, true_beta)

    glm = GeneralizedLinearRegressor(
        alpha=1,
        l1_ratio=0,
        fit_intercept=True,
        family="poisson",
        link="log",
        gradient_tol=1e-12,  # 1e-8 not sufficient
        random_state=np.random.RandomState(42),
        copy_X=True,
        scale_predictors=scale_predictors,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
    )
    glm.fit(X, y)

    # These correct values come from glmnet.
    assert_allclose(glm.intercept_, -0.13568186971946633, rtol=1e-5)
    assert_allclose(glm.coef_, [0.1, 0.1], rtol=1e-5)


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

    Compare to R's glmnet
    """
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
        {"solver": "trust-constr"},
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
        {"solver": "trust-constr", "rtol": 2e-4},
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
        M = tm.SparseMatrix(sparse.csc_matrix(M))
    else:
        M = tm.DenseMatrix(M)

    X, col_means, col_stds = M.standardize(np.ones(NR) / NR, True, scale_predictors)
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
            if isinstance(Xdense, tm.StandardizedMatrix):
                one, two = Xdense.A[:, 0], Xdense.A[:, i]
            else:
                one, two = Xdense[:, 0], Xdense[:, i]
        else:
            if isinstance(Xdense, tm.StandardizedMatrix):
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
    intercept, coef = _unstandardize(
        col_means,
        col_stds,
        intercept_standardized,
        coef_standardized,
    )
    np.testing.assert_almost_equal(intercept, -(NC + 1) * NC / 2)
    if scale_predictors:
        np.testing.assert_almost_equal(coef, 1.0)


@pytest.mark.parametrize("estimator, kwargs", estimators)
def test_check_estimator(estimator, kwargs):
    check_estimator(estimator(**kwargs))


@pytest.mark.parametrize(
    "estimator",
    [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV],
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
    y = np.array([1, 1, 1, 2], dtype=np.float_)
    if isinstance(distribution, BinomialDistribution):
        y -= 1

    sample_weight = np.array([0.1, 0.2, 5, 1])
    best_intercept = guess_intercept(y, sample_weight, link, distribution, offset)
    assert np.isfinite(best_intercept)

    def _get_dev(intercept):
        eta = intercept if offset is None else offset + intercept
        mu = link.inverse(eta)
        assert np.isfinite(mu).all()
        return distribution.deviance(y, mu, sample_weight)

    obj = _get_dev(best_intercept)
    obj_low = _get_dev(best_intercept - tol)
    obj_high = _get_dev(best_intercept + tol)
    assert obj < obj_low
    assert obj < obj_high


@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-6])
def test_step_size_tolerance(tol):
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        noise=0.5,
        random_state=42,
    )
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
        alpha=0.001,
        l1_ratio=1,
        family="normal",
        link="identity",
        gradient_tol=1e-10,
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


@pytest.mark.parametrize("alpha, alpha_index", [(0.5, 0), (0.75, 1), (None, 1)])
def test_predict_scalar(regression_data, alpha, alpha_index):

    X, y = regression_data
    offset = np.ones_like(y)

    estimator = GeneralizedLinearRegressor(alpha=[0.5, 0.75], alpha_search=True)
    estimator.fit(X, y)

    target = estimator.predict(X, alpha_index=alpha_index)

    candidate = estimator.predict(X, alpha=alpha, offset=offset)
    np.testing.assert_allclose(candidate, target + 1)


@pytest.mark.parametrize(
    "alpha, alpha_index",
    [([0.5, 0.75], [0, 1]), ([0.75, 0.5], [1, 0]), ([0.5, 0.5], [0, 0])],
)
def test_predict_list(regression_data, alpha, alpha_index):

    X, y = regression_data
    offset = np.ones_like(y)

    estimator = GeneralizedLinearRegressor(alpha=[0.5, 0.75], alpha_search=True)
    estimator.fit(X, y)

    target = np.stack(
        [
            estimator.predict(X, alpha_index=alpha_index[0]),
            estimator.predict(X, alpha_index=alpha_index[1]),
        ],
        axis=1,
    )

    candidate = estimator.predict(X, alpha=alpha, offset=offset)
    np.testing.assert_allclose(candidate, target + 1)

    candidate = estimator.predict(X, alpha_index=alpha_index, offset=offset)
    np.testing.assert_allclose(candidate, target + 1)


def test_predict_error(regression_data):

    X, y = regression_data

    estimator = GeneralizedLinearRegressor(alpha=0.5, alpha_search=False).fit(X, y)

    with pytest.raises(ValueError):
        estimator.predict(X, alpha=0.5)
    with pytest.raises(ValueError):
        estimator.predict(X, alpha=[0.5])
    with pytest.raises(AttributeError):
        estimator.predict(X, alpha_index=0)
    with pytest.raises(AttributeError):
        estimator.predict(X, alpha_index=[0])

    estimator.set_params(alpha=[0.5, 0.75], alpha_search=True).fit(X, y)

    with pytest.raises(IndexError):
        estimator.predict(X, y, alpha=0.25)
    with pytest.raises(IndexError):
        estimator.predict(X, y, alpha=[0.25, 0.5])
    with pytest.raises(IndexError):
        estimator.predict(X, y, alpha_index=2)
    with pytest.raises(IndexError):
        estimator.predict(X, y, alpha_index=[2, 0])
    with pytest.raises(ValueError):
        estimator.predict(X, y, alpha_index=0, alpha=0.5)


def test_very_large_initial_gradient():
    # this is a problem where 0 is a starting value that produces
    # a very large gradient initially
    np.random.seed(1234)
    y = np.exp(np.random.gamma(5, size=100))
    X = np.ones([len(y), 1])

    model_0 = GeneralizedLinearRegressor(
        link="log", family="gamma", fit_intercept=False, alpha=0, start_params=[0.0]
    ).fit(X, y)

    model_5 = GeneralizedLinearRegressor(
        link="log", family="gamma", fit_intercept=False, alpha=0, start_params=[5.0]
    ).fit(X, y)

    np.testing.assert_allclose(model_0.coef_, model_5.coef_, rtol=1e-5)


def test_fit_has_no_side_effects():
    y = np.array([0, 1, 2])
    w = np.array([0.5, 0.5, 0.5])
    X = np.array([[1, 1, 1]]).reshape(-1, 1)
    win = w.copy()
    yin = y.copy()
    Xin = X.copy()
    GeneralizedLinearRegressor(family="poisson").fit(Xin, yin, sample_weight=win)
    np.testing.assert_almost_equal(Xin, X)
    np.testing.assert_almost_equal(yin, y)
    np.testing.assert_almost_equal(win, w)
    GeneralizedLinearRegressor(family="poisson").fit(Xin, yin, offset=win)
    np.testing.assert_almost_equal(win, w)
    lb = np.array([-1.2])
    ub = np.array([1.2])
    lbin = lb.copy()
    ubin = ub.copy()
    GeneralizedLinearRegressor(
        family="poisson", scale_predictors=True, lower_bounds=lbin, upper_bounds=ubin
    ).fit(Xin, yin)
    np.testing.assert_almost_equal(lbin, lb)
    np.testing.assert_almost_equal(ubin, ub)


def test_column_with_stddev_zero():
    np.random.seed(1234)
    y = np.random.choice([1, 2, 3, 4], size=1000)
    X = np.ones([len(y), 1])

    model = GeneralizedLinearRegressor(
        family="poisson", fit_intercept=False, scale_predictors=False
    ).fit(
        X, y
    )  # noqa: F841
    model = GeneralizedLinearRegressor(family="poisson").fit(X, y)  # noqa: F841


@pytest.mark.parametrize("scale_predictors", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("P1", ["identity", np.array([2, 1, 0.5, 0.1, 0.01])])
def test_alpha_path(scale_predictors, fit_intercept, P1):
    """Test regularization path."""
    if scale_predictors and not fit_intercept:
        return
    np.random.seed(1234)
    y = np.random.choice([1, 2, 3, 4], size=100)
    X = np.random.randn(100, 5) * np.array([1, 5, 10, 25, 100])

    model = GeneralizedLinearRegressor(
        family="poisson",
        alpha_search=True,
        l1_ratio=1,
        n_alphas=10,
        scale_predictors=scale_predictors,
        fit_intercept=fit_intercept,
        P1=P1,
    )
    model.fit(X=X, y=y)

    # maximum alpha result in all zero coefficients
    np.testing.assert_almost_equal(model.coef_path_[0], 0)
    # next alpha gives at least one non-zero coefficient
    assert np.any(model.coef_path_[1] > 0)


def test_passing_noncontiguous_as_X():
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    baseline = GeneralizedLinearRegressor(family="normal", fit_intercept=False).fit(
        X[:, :2].copy(), y
    )
    np_view = GeneralizedLinearRegressor(family="normal", fit_intercept=False).fit(
        X[:, :2], y
    )
    pd_view = GeneralizedLinearRegressor(family="normal", fit_intercept=False).fit(
        pd.DataFrame(X).iloc[:, :2], y
    )
    np.testing.assert_almost_equal(baseline.coef_, np_view.coef_)
    np.testing.assert_almost_equal(baseline.coef_, pd_view.coef_)


@pytest.mark.parametrize(
    "X, feature_names",
    [
        (pd.DataFrame({"x1": np.arange(5), "x2": 2}), np.array(["x1", "x2"])),
        (pd.DataFrame({"x1": np.arange(5), "x2": 2}).to_numpy(), None),
        (
            pd.DataFrame({"x1": pd.Categorical(np.arange(5)), "x2": 2}),
            np.array(["x1__0", "x1__1", "x1__2", "x1__3", "x1__4", "x2"]),
        ),
        (
            pd.DataFrame(
                {
                    "x1": pd.Categorical(np.arange(5)),
                    "x2": pd.Categorical([2, 2, 2, 2, 2]),
                }
            ),
            np.array(["x1__0", "x1__1", "x1__2", "x1__3", "x1__4", "x2__2"]),
        ),
    ],
)
def test_feature_names(X, feature_names):
    model = GeneralizedLinearRegressor(family="poisson").fit(X, np.arange(5))
    np.testing.assert_array_equal(getattr(model, "feature_names_", None), feature_names)


@pytest.mark.parametrize(
    "X, dtypes",
    [
        (pd.DataFrame({"x1": np.arange(5)}, dtype="int64"), {"x1": np.int64}),
        (pd.DataFrame({"x1": np.arange(5)}).to_numpy(), None),
        (
            pd.DataFrame({"x1": pd.Categorical(np.arange(5))}),
            {"x1": pd.CategoricalDtype(list(range(5)), ordered=False)},
        ),
    ],
)
def test_feature_dtypes(X, dtypes):
    model = GeneralizedLinearRegressor(family="poisson").fit(X, np.arange(5))
    np.testing.assert_array_equal(getattr(model, "feature_dtypes_", None), dtypes)


@pytest.mark.parametrize(
    "k, n",
    [
        (5, 5),
        (10, 5),
        (100, 5),
        (500, 50),
        (500, 100),
        (500, 500),
    ],
)
def test_categorical_types(k, n):
    np.random.seed(12345)
    categories = np.arange(k)
    group = np.random.choice(categories, size=n)
    y = group / k + np.random.uniform(size=n)

    # use categorical types
    X_cat = pd.DataFrame({"group": pd.Categorical(group, categories=categories)})
    model_cat = GeneralizedLinearRegressor(family="poisson").fit(X_cat, y)
    pred_cat = model_cat.predict(X_cat)

    # use one-hot encoding
    X_oh = pd.get_dummies(X_cat, dtype=float)
    model_oh = GeneralizedLinearRegressor(family="poisson").fit(X_oh, y)
    pred_oh = model_oh.predict(X_oh)

    # check predictions
    np.testing.assert_allclose(pred_cat, pred_oh)
    np.testing.assert_allclose(model_cat.intercept_, model_oh.intercept_)
    np.testing.assert_allclose(model_cat.coef_, model_oh.coef_)

    # compare across models/data types
    pred_cat_oh = model_cat.predict(X_oh)
    pred_oh_cat = model_oh.predict(X_cat)
    np.testing.assert_allclose(pred_cat_oh, pred_oh_cat)


@pytest.mark.parametrize(
    "kwargs", [{"alpha_search": True, "alpha": [1, 0.5, 0.1, 0.01]}, {"alpha": 0.1}]
)
def test_alpha_parametrization(kwargs, regression_data):
    X, y = regression_data
    model = GeneralizedLinearRegressor(**kwargs)
    model.fit(X=X, y=y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_search": True, "alpha": 0.1},
        {"alpha_search": False, "alpha": [1, 0.5, 0.1, 0.01]},
        {"alpha_search": True, "alpha": "abc"},
        {"alpha": "abc"},
        {"alpha": -0.1},
        {"alpha_search": True, "alpha": [1, 0.5, -0.1]},
    ],
)
def test_alpha_parametrization_fail(kwargs, regression_data):
    X, y = regression_data
    with pytest.raises((ValueError, TypeError)):
        model = GeneralizedLinearRegressor(**kwargs)
        model.fit(X=X, y=y)


def test_verbose(regression_data, capsys):
    X, y = regression_data
    mdl = GeneralizedLinearRegressor(verbose=1)
    mdl.fit(X=X, y=y)
    captured = capsys.readouterr()
    assert "Iteration" in captured.err


def test_ols_std_errors(regression_data):
    X, y = regression_data
    mdl = GeneralizedLinearRegressor(alpha=0, family="normal")
    mdl.fit(X=X, y=y)

    mdl_sm = sm.OLS(endog=y, exog=sm.add_constant(X))

    # nonrobust
    # Here, statsmodels does not do a degree of freedom adjustment,
    # so we manually add it.
    ourse = mdl.std_errors(X=X, y=y, robust=False)
    corr = len(y) / (len(y) - X.shape[1] - 1)
    smse = mdl_sm.fit(cov_type="nonrobust").bse * np.sqrt(corr)
    np.testing.assert_allclose(ourse, smse, rtol=1e-8)

    # robust
    ourse = mdl.std_errors(X=X, y=y, robust=True)
    smse = mdl_sm.fit(cov_type="HC1").bse
    np.testing.assert_allclose(ourse, smse, rtol=1e-8)

    # clustered
    rng = np.random.default_rng(42)
    clu = rng.integers(5, size=len(y))
    ourse = mdl.std_errors(X=X, y=y, clusters=clu)
    smse = mdl_sm.fit(cov_type="cluster", cov_kwds={"groups": clu}).bse
    np.testing.assert_allclose(ourse, smse, rtol=1e-8)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("family", ["poisson", "normal", "binomial"])
def test_array_std_errors(regression_data, family, fit_intercept):
    X, y = regression_data
    if family == "poisson":
        y = np.round(abs(y))
        sm_family = sm.families.Poisson()
        dispersion = 1
    elif family == "binomial":
        rng = np.random.default_rng(42)
        y = rng.choice([0, 1], len(y))
        sm_family = sm.families.Binomial()
        dispersion = 1
    else:
        sm_family = sm.families.Gaussian()
        dispersion = None

    mdl = GeneralizedLinearRegressor(
        alpha=0, family=family, fit_intercept=fit_intercept
    ).fit(X=X, y=y)

    if fit_intercept:
        mdl_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm_family)
    else:
        mdl_sm = sm.GLM(endog=y, exog=X, family=sm_family)

    # Here, statsmodels does not do a degree of freedom adjustment,
    # so we manually add it for nonrobust and robust errors

    # nonrobust
    ourse = mdl.std_errors(X=X, y=y, dispersion=dispersion, robust=False)
    corr = len(y) / (len(y) - X.shape[1] - int(fit_intercept))
    smse = mdl_sm.fit(cov_type="nonrobust").bse * np.sqrt(corr)
    np.testing.assert_allclose(ourse, smse, rtol=1e-4)

    # robust
    ourse = mdl.std_errors(X=X, y=y, robust=True)
    smse = mdl_sm.fit(cov_type="HC1").bse * np.sqrt(corr)
    np.testing.assert_allclose(ourse, smse, rtol=1e-4)

    # clustered
    rng = np.random.default_rng(42)
    clu = rng.integers(5, size=len(y))
    ourse = mdl.std_errors(X=X, y=y, clusters=clu)
    smse = mdl_sm.fit(cov_type="cluster", cov_kwds={"groups": clu}).bse
    np.testing.assert_allclose(ourse, smse, rtol=1e-4)


def test_sparse_std_errors(regression_data):
    X, y = regression_data
    sp_X = sparse.csc_matrix(X)
    mdl = GeneralizedLinearRegressor(alpha=0, family="normal")
    mdl.fit(X=X, y=y)

    actual1 = mdl.std_errors(X=sp_X, y=y, robust=False)
    expected1 = mdl.std_errors(X=X, y=y, robust=False)
    np.testing.assert_allclose(actual1, expected1)
    actual2 = mdl.std_errors(X=sp_X, y=y, robust=True)
    expected2 = mdl.std_errors(X=X, y=y, robust=True)
    np.testing.assert_allclose(actual2, expected2)
    rng = np.random.default_rng(42)
    clu = rng.integers(5, size=len(y))
    actual3 = mdl.std_errors(X=X, y=y, clusters=clu)
    expected3 = mdl.std_errors(X=X, y=y, clusters=clu)
    np.testing.assert_allclose(actual3, expected3)


# TODO add intercepts for models with categorical variables when glum allows drop_first
@pytest.mark.parametrize(
    "categorical, split, fit_intercept",
    [
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (False, False, True),
    ],
)
def test_dataframe_std_errors(regression_data, categorical, split, fit_intercept):
    X, y = regression_data
    X = pd.DataFrame(X)
    if categorical or split:
        rng = np.random.default_rng(42)
        categories = np.arange(4)
        group = rng.choice(categories, size=len(X))
        if categorical:
            X = pd.DataFrame(pd.Categorical(group, categories=categories))
        if split:
            X = X.assign(col_cat=pd.Categorical(group, categories=categories))
    mdl = GeneralizedLinearRegressor(
        alpha=0, family="normal", fit_intercept=fit_intercept
    )
    mdl.fit(X=X, y=y)
    X_sm = pd.get_dummies(X, dtype=float)
    if fit_intercept:
        mdl_sm = sm.OLS(endog=y, exog=sm.add_constant(X_sm))
    else:
        mdl_sm = sm.OLS(endog=y, exog=X_sm)

    # nonrobust
    # manually add dof adjustment in statsmodels
    ourse = mdl.std_errors(X=X, y=y, robust=False)
    corr = len(y) / (len(y) - X_sm.shape[1] - fit_intercept)
    smse = mdl_sm.fit(cov_type="nonrobust").bse * np.sqrt(corr)
    np.testing.assert_allclose(ourse, smse, rtol=1e-8)

    # robust
    ourse = mdl.std_errors(X=X, y=y, robust=True)
    smse = mdl_sm.fit(cov_type="HC1").bse
    np.testing.assert_allclose(ourse, smse, rtol=1e-8)

    # clustered
    rng = np.random.default_rng(42)
    clu = rng.integers(5, size=len(y))
    ourse = mdl.std_errors(X=X, y=y, clusters=clu)
    smse = mdl_sm.fit(cov_type="cluster", cov_kwds={"groups": clu}).bse
    np.testing.assert_allclose(ourse, smse, rtol=1e-8)


@pytest.mark.parametrize("as_data_frame", [False, True])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_score_method(as_data_frame, offset, weighted):

    regressor = GeneralizedLinearRegressor(
        alpha=0,
        family="normal",
        fit_intercept=False,
        gradient_tol=1e-8,
        check_input=False,
    )

    y = np.array([-1, -1, 0, 1, 2])

    if weighted:
        y, wgts = np.unique(y, return_counts=True)
    else:
        wgts = None

    if as_data_frame:
        x = pd.DataFrame({"x": np.ones(len(y))})
    else:
        x = np.ones((len(y), 1))

    if offset:
        offset = y
    else:
        offset = None

    score = regressor.fit(x, y, offset=offset, sample_weight=wgts).score(
        x, y, offset=offset, sample_weight=wgts
    )

    # use pytest because NumPy used to always reject comparisons against zero
    assert pytest.approx(score, 1e-8) == int(offset is not None)


def test_information_criteria(regression_data):
    X, y = regression_data
    regressor = GeneralizedLinearRegressor(family="gaussian", alpha=0)
    regressor.fit(X, y)

    llf = regressor.family_instance.log_likelihood(y, regressor.predict(X))
    nobs, df = X.shape[0], X.shape[1] + 1
    sm_aic = eval_measures.aic(llf, nobs, df)
    sm_bic = eval_measures.bic(llf, nobs, df)
    sm_aicc = eval_measures.aicc(llf, nobs, df)

    assert np.allclose(
        [sm_aic, sm_aicc, sm_bic],
        [regressor.aic(X, y), regressor.aicc(X, y), regressor.bic(X, y)],
        atol=1e-8,
    )


@pytest.mark.filterwarnings("ignore: There is no")
def test_information_criteria_raises_correct_warnings_and_errors(regression_data):
    X, y = regression_data

    # test no warnings are raised for L1 regularisation
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        regressor = GeneralizedLinearRegressor(family="normal", l1_ratio=1.0)
        regressor.fit(X, y)
        regressor.aic(X, y), regressor.aicc(X, y), regressor.bic(X, y)

    # test warnings are raised for L2 regularisation
    with pytest.warns(match="There is no") as records:
        regressor = GeneralizedLinearRegressor(family="normal", l1_ratio=0.0)
        regressor.fit(X, y)
        regressor.aic(X, y), regressor.aicc(X, y), regressor.bic(X, y)
    assert len(records) == 3

    # test exceptions are raised when information criteria called but model not fitted
    regressor = GeneralizedLinearRegressor()
    with pytest.raises(Exception):
        regressor.aic(X, y)
    with pytest.raises(Exception):
        regressor.aicc(X, y)
    with pytest.raises(Exception):
        regressor.bic(X, y)

    # test exception is raised when not train set is used
    regressor.fit(X, y)
    X_not_train = np.ones((10, 2))
    y_not_train = np.ones(10)
    with pytest.raises(Exception):
        regressor.aic(X_not_train, y_not_train)


def test_drop_first_allows_alpha_equals_0():
    rng = np.random.default_rng(42)
    y = np.random.normal(size=10)
    X = pd.DataFrame(data={"cat": pd.Categorical(rng.integers(2, size=10))})
    regressor = GeneralizedLinearRegressor(alpha=0, drop_first=True)
    regressor.fit(X, y)

    regressor = GeneralizedLinearRegressor(alpha=0)  # default is False
    with pytest.raises(np.linalg.LinAlgError):
        regressor.fit(X, y)


def test_error_on_distinct_categorical_column():
    y = np.random.normal(size=10)
    X = pd.DataFrame(data={"cat": pd.Categorical(np.ones(10))})
    regressor = GeneralizedLinearRegressor(alpha=0, drop_first=True)
    with pytest.raises(ValueError):
        regressor.fit(X, y)

    regressor = GeneralizedLinearRegressor(alpha=0)
    regressor.fit(X, y)


def test_P1_P2_with_drop_first():
    rng = np.random.default_rng(42)
    y = np.random.normal(size=50)
    X = pd.DataFrame(data={"cat": pd.Categorical(rng.integers(2, size=50))})
    P_2 = np.ones(1)
    P_1 = np.ones(1)
    regressor = GeneralizedLinearRegressor(
        alpha=0.1, l1_ratio=0.5, P1=P_1, P2=P_2, drop_first=True
    )
    regressor.fit(X, y)
    regressor = GeneralizedLinearRegressor(alpha=0.1, l1_ratio=0.5, P1=P_1, P2=P_2)
    regressor.fit(X, y)
