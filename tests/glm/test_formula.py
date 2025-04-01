import formulaic
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from glum._formula import parse_formula
from glum._glm import GeneralizedLinearRegressor


@pytest.fixture
def get_mixed_data():
    nrow = 10
    np.random.seed(0)
    return pd.DataFrame(
        {
            "y": np.random.rand(nrow),
            "x1": np.random.rand(nrow),
            "x2": np.random.rand(nrow),
            "c1": np.random.choice(["a", "b", "c"], nrow),
            "c2": np.random.choice(["d", "e"], nrow),
        }
    )


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
def test_formula(get_mixed_data, formula, drop_first, fit_intercept):
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
            materializer=formulaic.materializers.PandasMaterializer,
        )
        X_ext = X_ext.drop(columns="Intercept")
    else:
        y_ext, X_ext = formulaic.model_matrix(
            formula + "-1",
            data,
            ensure_full_rank=drop_first,
            materializer=formulaic.materializers.PandasMaterializer,
        )
    y_ext = y_ext.iloc[:, 0]

    model_ext = GeneralizedLinearRegressor(
        family="normal",
        drop_first=drop_first,
        fit_intercept=fit_intercept,
        categorical_format="{name}[T.{category}]",
        alpha=1.0,
    ).fit(X_ext, y_ext)

    np.testing.assert_almost_equal(model_ext.coef_, model_formula.coef_)


def test_formula_explicit_intercept(get_mixed_data):
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
def test_formula_names_formulaic_style(
    get_mixed_data, formula, feature_names, term_names
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
def test_formula_names_old_glum_style(
    get_mixed_data, formula, feature_names, term_names
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
def test_formula_against_smf(get_mixed_data, formula, fit_intercept):
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


def test_formula_context(get_mixed_data):
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
def test_formula_predict(get_mixed_data, formula, fit_intercept):
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
