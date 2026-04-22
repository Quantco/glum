import warnings
from typing import Literal

import formulaic
import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import sparse as sp
from sklearn.preprocessing import SplineTransformer

from glum._formula import _build_monotonic_constraints, parse_formula
from glum._glm import GeneralizedLinearRegressor
from glum._glm_cv import GeneralizedLinearRegressorCV


@pytest.fixture
def get_mixed_data(namespace: Literal["pandas", "polars"]):
    nrow = 10
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "y": np.random.rand(nrow),
            "x1": np.random.rand(nrow),
            "x2": np.random.rand(nrow),
            "c1": pd.Categorical(np.random.choice(["a", "b", "c"], nrow)),
            "c2": pd.Categorical(np.random.choice(["d", "e"], nrow)),
        }
    )
    if namespace == "pandas":
        return df
    elif namespace == "polars":
        return pl.from_pandas(df).with_columns(
            # We need this otherwise category order is not preserved by formulaic
            pl.col("c1").cast(pl.Enum(["a", "b", "c"])),
            pl.col("c2").cast(pl.Enum(["d", "e"])),
        )
    else:
        raise ValueError(f"Unknown namespace: {namespace}")


@pytest.mark.parametrize(
    "input, expected",
    [
        pytest.param(
            "y ~ x1 + x2",
            (["y"], ["1", "x1", "x2"]),
            id="implicit_intercept",
        ),
        pytest.param(
            "y ~ x1 + x2 + 1",
            (["y"], ["1", "x1", "x2"]),
            id="explicit_intercept",
        ),
        pytest.param(
            "y ~ x1 + x2 - 1",
            (["y"], ["x1", "x2"]),
            id="no_intercept",
        ),
        pytest.param(
            "y ~ ",
            (["y"], ["1"]),
            id="empty_rhs",
        ),
    ],
)
def test_parse_formula(input, expected):
    lhs_exp, rhs_exp = expected
    lhs, rhs = parse_formula(input)
    assert list(lhs) == lhs_exp
    assert list(rhs) == rhs_exp

    formula = formulaic.Formula(input)
    lhs, rhs = parse_formula(formula)
    assert list(lhs) == lhs_exp
    assert list(rhs) == rhs_exp


@pytest.mark.parametrize(
    "input, error",
    [
        pytest.param("y1 + y2 ~ x1 + x2", ValueError, id="multiple_lhs"),
        pytest.param([["y"], ["x1", "x2"]], TypeError, id="wrong_type"),
    ],
)
def test_parse_formula_invalid(input, error):
    with pytest.raises(error):
        parse_formula(input)


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("y ~ x1 + x2", id="numeric"),
        pytest.param("y ~ c1", id="categorical"),
        pytest.param("y ~ c1 * c2", id="categorical_interaction"),
        pytest.param("y ~ x1 + x2 + c1 + c2", id="numeric_categorical"),
        pytest.param("y ~ x1 * c1 * c2", id="numeric_categorical_interaction"),
    ],
)
@pytest.mark.parametrize(
    "drop_first", [True, False], ids=["drop_first", "no_drop_first"]
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
@pytest.mark.parametrize("namespace", ["pandas", "polars"])
def test_formula(get_mixed_data, formula, drop_first, fit_intercept, namespace):
    """Model with formula and model with externally constructed model matrix should
    match.
    """
    data = get_mixed_data

    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=drop_first,
        formula=formula,
        fit_intercept=fit_intercept,
        categorical_format="{name}[T.{category}]",
        alpha=1.0,
    ).fit(data)

    if fit_intercept:
        # full rank check must consider presence of intercept
        y_ext, X_ext = formulaic.model_matrix(
            formula,
            data,
            ensure_full_rank=drop_first,
        )
        if namespace == "pandas":
            X_ext = X_ext.drop(columns="Intercept")
        elif namespace == "polars":
            X_ext = X_ext.drop("Intercept")
    else:
        y_ext, X_ext = formulaic.model_matrix(
            formula + "-1",
            data,
            ensure_full_rank=drop_first,
        )

    if namespace == "pandas":
        y_ext = y_ext.iloc[:, 0]
    elif namespace == "polars":
        y_ext = y_ext.to_series()

    model_ext = GeneralizedLinearRegressor(
        family="normal",
        drop_first=drop_first,
        fit_intercept=fit_intercept,
        categorical_format="{name}[T.{category}]",
        alpha=1.0,
    ).fit(X_ext, y_ext)

    np.testing.assert_almost_equal(model_ext.coef_, model_formula.coef_)


@pytest.mark.parametrize("namespace", ["pandas", "polars"])
def test_formula_explicit_intercept(get_mixed_data, namespace):
    data = get_mixed_data

    with pytest.raises(ValueError, match="The formula sets the intercept to False"):
        GeneralizedLinearRegressor(
            family="normal",
            formula="y ~ x1 - 1",
            fit_intercept=True,
        ).fit(data)


@pytest.mark.parametrize(
    "formula, feature_names, term_names",
    [
        pytest.param("y ~ x1 + x2", ["x1", "x2"], ["x1", "x2"], id="numeric"),
        pytest.param(
            "y ~ c1", ["c1[T.a]", "c1[T.b]", "c1[T.c]"], 3 * ["c1"], id="categorical"
        ),
        pytest.param(
            "y ~ x1 : c1",
            ["x1:c1[T.a]", "x1:c1[T.b]", "x1:c1[T.c]"],
            3 * ["x1:c1"],
            id="interaction",
        ),
        pytest.param(
            "y ~ poly(x1, 3)",
            ["poly(x1, 3)[1]", "poly(x1, 3)[2]", "poly(x1, 3)[3]"],
            3 * ["poly(x1, 3)"],
            id="function",
        ),
    ],
)
@pytest.mark.parametrize("namespace", ["pandas", "polars"])
def test_formula_names_formulaic_style(
    get_mixed_data, formula, feature_names, term_names, namespace
):
    data = get_mixed_data
    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=False,
        formula=formula,
        categorical_format="{name}[T.{category}]",
        interaction_separator=":",
        alpha=1.0,
    ).fit(data)

    np.testing.assert_array_equal(model_formula.feature_names_, feature_names)
    np.testing.assert_array_equal(model_formula.term_names_, term_names)


@pytest.mark.parametrize(
    "formula, feature_names, term_names",
    [
        pytest.param("y ~ x1 + x2", ["x1", "x2"], ["x1", "x2"], id="numeric"),
        pytest.param(
            "y ~ c1", ["c1__a", "c1__b", "c1__c"], 3 * ["c1"], id="categorical"
        ),
        pytest.param(
            "y ~ x1 : c1",
            ["x1__x__c1__a", "x1__x__c1__b", "x1__x__c1__c"],
            3 * ["x1:c1"],
            id="interaction",
        ),
        pytest.param(
            "y ~ poly(x1, 3)",
            ["poly(x1, 3)[1]", "poly(x1, 3)[2]", "poly(x1, 3)[3]"],
            3 * ["poly(x1, 3)"],
            id="function",
        ),
    ],
)
@pytest.mark.parametrize("namespace", ["pandas", "polars"])
def test_formula_names_old_glum_style(
    get_mixed_data, formula, feature_names, term_names, namespace
):
    data = get_mixed_data
    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=False,
        formula=formula,
        categorical_format="{name}__{category}",
        interaction_separator="__x__",
        alpha=1.0,
    ).fit(data)

    np.testing.assert_array_equal(model_formula.feature_names_, feature_names)
    np.testing.assert_array_equal(model_formula.term_names_, term_names)


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("y ~ x1 + x2", id="numeric"),
        pytest.param("y ~ c1", id="categorical"),
        pytest.param("y ~ c1 * c2", id="categorical_interaction"),
    ],
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
@pytest.mark.parametrize("namespace", ["pandas"])
def test_formula_against_smf(get_mixed_data, formula, fit_intercept, namespace):
    # Only test with pandas since statsmodels doesn't support polars
    data = get_mixed_data
    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=True,
        formula=formula,
        fit_intercept=fit_intercept,
    ).fit(data)

    if fit_intercept:
        beta_formula = np.concatenate([[model_formula.intercept_], model_formula.coef_])
    else:
        beta_formula = model_formula.coef_

    formula_smf = formula + "- 1" if not fit_intercept else formula
    model_smf = smf.glm(formula_smf, data, family=sm.families.Gaussian()).fit()

    np.testing.assert_almost_equal(beta_formula, model_smf.params)


@pytest.mark.parametrize("namespace", ["pandas"])
def test_formula_context(get_mixed_data, namespace):
    # Only test with pandas since statsmodels doesn't support polars
    data = get_mixed_data
    x_context = np.arange(len(data), dtype=float)  # noqa: F841
    formula = "y ~ x1 + x2 + x_context"

    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=True,
        formula=formula,
        fit_intercept=True,
    )
    # default is to add nothing to context
    with pytest.raises(formulaic.errors.FactorEvaluationError):
        model_formula.fit(data)

    # set context to 0 to capture calling scope
    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=True,
        formula=formula,
        fit_intercept=True,
    ).fit(data, context=0)

    model_smf = smf.glm(formula, data, family=sm.families.Gaussian()).fit()

    np.testing.assert_almost_equal(
        np.concatenate([[model_formula.intercept_], model_formula.coef_]),
        model_smf.params,
    )
    np.testing.assert_almost_equal(
        model_formula.predict(data, context=0), model_smf.predict(data)
    )


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("y ~ x1 + x2", id="numeric"),
        pytest.param("y ~ c1", id="categorical"),
        pytest.param("y ~ c1 * c2", id="categorical_interaction"),
    ],
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
@pytest.mark.parametrize("namespace", ["pandas"])
def test_formula_predict(get_mixed_data, formula, fit_intercept, namespace):
    # Only test with pandas since statsmodels doesn't support polars
    data = get_mixed_data
    data_unseen = data.copy()
    data_unseen.loc[data_unseen["c1"] == "b", "c1"] = "c"
    model_formula = GeneralizedLinearRegressor(
        family="normal",
        drop_first=True,
        formula=formula,
        fit_intercept=fit_intercept,
    ).fit(data)

    formula_smf = formula + "- 1" if not fit_intercept else formula
    model_smf = smf.glm(formula_smf, data, family=sm.families.Gaussian()).fit()

    yhat_formula = model_formula.predict(data_unseen)
    yhat_smf = model_smf.predict(data_unseen)

    np.testing.assert_almost_equal(yhat_formula, yhat_smf)


def test_build_monotonic_constraints_sorts_indices():
    """Consecutive-pair constraints are correct even when columns are out of order."""
    names = ["sp[10]", "sp[2]", "sp[1]", "sp[3]"]
    A, b = _build_monotonic_constraints(names, {"sp": "increasing"})
    assert b.tolist() == [0.0, 0.0, 0.0]
    # Sorted order: sp[1]=idx2, sp[2]=idx1, sp[3]=idx3, sp[10]=idx0
    # Pairs: (2,1), (1,3), (3,0) with sign=+1 for increasing: row[i]=1, row[j]=-1
    expected = np.zeros((3, 4))
    expected[0, 2] = 1.0  # sp[1]
    expected[0, 1] = -1.0  # sp[2]
    expected[1, 1] = 1.0  # sp[2]
    expected[1, 3] = -1.0  # sp[3]
    expected[2, 3] = 1.0  # sp[3]
    expected[2, 0] = -1.0  # sp[10]
    np.testing.assert_array_equal(A, expected)


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
@pytest.mark.parametrize(
    "estimator_cls",
    [GeneralizedLinearRegressor, GeneralizedLinearRegressorCV],
    ids=["GLM", "GLMCV"],
)
def test_monotonic_constraints_spline(direction, estimator_cls):
    """Spline coefficients are ordered."""
    sign = 1 if direction == "increasing" else -1
    rng = np.random.default_rng(42)
    n = 500
    x = np.sort(rng.uniform(0, 1, n))
    y = (np.sin(2 * np.pi * x) + rng.standard_normal(n) * 0.3).clip(0.01)
    df = pd.DataFrame({"x": x, "y": y})

    kwargs = dict(
        formula="y ~ bs(x, df=6) - 1",
        monotonic_constraints={"x": direction},
        alpha=0.1,
        l1_ratio=0,
        fit_intercept=False,
        gradient_tol=1e-8,
    )
    if estimator_cls is GeneralizedLinearRegressorCV:
        del kwargs["alpha"]
        kwargs["n_alphas"] = 5

    model = estimator_cls(**kwargs)
    model.fit(df)

    diffs = np.diff(model.coef_)
    assert all(sign * diffs >= -1e-8)


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_monotonic_constraints_spline_interaction(direction):
    """Spline x categorical: coefficients ordered within each group level."""
    sign = 1 if direction == "increasing" else -1
    rng = np.random.default_rng(42)
    n = 400
    x = rng.uniform(0, 1, n)
    g = rng.choice(["a", "b"], n)
    y = (x + (g == "b").astype(float) + rng.standard_normal(n) * 0.3).clip(0.01)
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "g": pd.Categorical(g),
        }
    )

    model = GeneralizedLinearRegressor(
        formula="y ~ bs(x, df=5):g - 1",
        monotonic_constraints={"x": direction, "g": direction},
        alpha=0.1,
        l1_ratio=0,
        fit_intercept=False,
        gradient_tol=1e-8,
    )
    model.fit(df)

    names = list(model.feature_names_)
    for lvl in ["a", "b"]:
        indices = [i for i, n in enumerate(names) if f"g[{lvl}]" in n]
        diffs = np.diff(model.coef_[indices])
        assert all(sign * diffs >= -1e-8)

    for bs_idx in sorted({n.split(":")[0] for n in names}):
        idx_a = [i for i, n in enumerate(names) if n.startswith(bs_idx) and "g[a]" in n]
        idx_b = [i for i, n in enumerate(names) if n.startswith(bs_idx) and "g[b]" in n]
        if idx_a and idx_b:
            assert sign * (model.coef_[idx_b[0]] - model.coef_[idx_a[0]]) >= -1e-8


def _make_spline_constraint_data(n=500, seed=42):
    """Build spline design matrix with increasing-constraint matrices."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = (np.sin(2 * np.pi * x) + rng.standard_normal(n) * 0.3).clip(0.01)
    X = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit_transform(
        x.reshape(-1, 1)
    )
    p = X.shape[1]
    A_ineq = np.zeros((p - 1, p))
    for i in range(p - 1):
        A_ineq[i, i] = 1.0
        A_ineq[i, i + 1] = -1.0
    b_ineq = np.zeros(p - 1)
    return x, X, y, A_ineq, b_ineq


def test_monotonic_constraint_paths_agree():
    """IRLS-formula, IRLS-explicit A_ineq/b_ineq, and trust-constr agree."""
    x, _, y, _, _ = _make_spline_constraint_data()
    df = pd.DataFrame({"x": x, "y": y})

    common = dict(alpha=0.1, l1_ratio=0, fit_intercept=False, gradient_tol=1e-8)

    formula_model = GeneralizedLinearRegressor(
        formula="y ~ bs(x, df=6) - 1",
        monotonic_constraints={"x": "increasing"},
        solver="irls-ls-monotonic",
        **common,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        formula_model.fit(df)

    X = formula_model.X_model_spec_.get_model_matrix(df).toarray()
    p = X.shape[1]
    A_ineq = np.zeros((p - 1, p))
    for i in range(p - 1):
        A_ineq[i, i] = 1.0
        A_ineq[i, i + 1] = -1.0
    b_ineq = np.zeros(p - 1)

    irls_model = GeneralizedLinearRegressor(
        solver="irls-ls-monotonic", A_ineq=A_ineq, b_ineq=b_ineq, **common
    )
    tc_model = GeneralizedLinearRegressor(
        solver="trust-constr", A_ineq=A_ineq, b_ineq=b_ineq, max_iter=1000, **common
    )
    unconstrained_model = GeneralizedLinearRegressor(solver="irls-ls", **common)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        irls_model.fit(X, y)
        tc_model.fit(X, y)
        unconstrained_model.fit(X, y)

    np.testing.assert_allclose(formula_model.coef_, irls_model.coef_, atol=1e-8)
    np.testing.assert_allclose(irls_model.coef_, tc_model.coef_, atol=1e-3)

    assert np.all(A_ineq @ formula_model.coef_ <= 1e-6)
    assert np.all(A_ineq @ irls_model.coef_ <= 1e-6)
    assert np.all(A_ineq @ tc_model.coef_ <= 1e-6)
    assert np.any(A_ineq @ unconstrained_model.coef_ > 1e-6)


def test_monotonic_with_matrix_P2():
    """Dense and sparse matrix P2 produce identical monotonic-constrained fits."""
    _, X, y, A_ineq, b_ineq = _make_spline_constraint_data(n=300)
    P2 = np.eye(X.shape[1]) * 0.1

    common = dict(
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        l1_ratio=0,
        fit_intercept=False,
        gradient_tol=1e-8,
        solver="irls-ls-monotonic",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_dense = GeneralizedLinearRegressor(P2=P2, **common).fit(X, y)
        m_sparse = GeneralizedLinearRegressor(P2=sp.csc_matrix(P2), **common).fit(X, y)

    np.testing.assert_allclose(m_dense.coef_, m_sparse.coef_, atol=1e-6)
    assert np.all(A_ineq @ m_dense.coef_ <= 1e-6)
