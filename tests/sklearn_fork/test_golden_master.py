import pickle

import numpy as np
import pytest
from git_root import git_root

from glm_benchmarks.sklearn_fork._glm import GeneralizedLinearRegressor


def create_poisson_reg_data(n_rows=10000, n_features=10):
    rand = np.random.default_rng(42)
    X = rand.standard_normal(size=(n_rows, n_features))
    coefs = np.array([1.0, 0.5, 0.1, -0.1, -0.5, -1.0, 0, 0, 0, 0])
    intercept = 0.2
    y = rand.poisson(np.exp(intercept + X @ coefs))
    weights = rand.uniform(size=n_rows)
    data = {"intercept": intercept, "X": X, "b": coefs, "y": y, "weights": weights}
    return data


@pytest.fixture
def poisson_reg_data():
    return create_poisson_reg_data()


def _make_P2():
    rand = np.random.default_rng(42)
    a = rand.uniform(size=(10, 10)) - 0.5  # centered uniform distribution
    P2 = a.T @ a  # make sure P2 is positive semi-definite
    return P2


gm_model_parameters = {
    "default": {},  # default params
    "no-regularization": {"alpha": 0},  # no-regularization
    "half-regularization": {"alpha": 0.5},  # regularization (other than alpha = 1)
    "elastic-net": {"l1_ratio": 0.5},  # elastic-net
    "lasso": {"l1_ratio": 1},  # lasso
    "variable_p1": {"l1_ratio": 1, "P1": np.arange(10)},  # lasso with variable penalty
    "variable_p2": {
        "l1_ratio": 0,
        "P2": _make_P2(),
    },  # ridge with Tikhonov regularization
    "variable_p1_p2": {
        "l1_ratio": 0.5,
        "P1": np.arange(10),
        "P2": _make_P2(),
    },  # elastic net with P1 and P2 variable penalty
    "fit_intercept": {"fit_intercept": False},  # do not fit the intercept
}


@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
@pytest.mark.parametrize(
    ["run_name", "model_parameters"],
    gm_model_parameters.items(),
    ids=gm_model_parameters.keys(),
)
def test_poisson_golden_master(
    model_parameters, run_name, use_weights, poisson_reg_data
):
    model = GeneralizedLinearRegressor(family="poisson", **model_parameters)

    fit_params = {
        "X": poisson_reg_data["X"],
        "y": poisson_reg_data["y"],
    }

    if use_weights:
        fit_params.update({"sample_weight": poisson_reg_data["weights"]})
        run_name = f"{run_name}_weights"

    model.fit(**fit_params)

    with open(git_root(f"golden_master/gm_{run_name}.pkl"), "rb") as fh:
        expected = pickle.load(fh)

    np.testing.assert_allclose(model.coef_, expected.coef_, rtol=1e-5, atol=0)
    np.testing.assert_allclose(model.intercept_, expected.intercept_, rtol=1e-5, atol=0)


def run_and_store_golden_master(model_parameters, run_name, use_weights, data):
    model = GeneralizedLinearRegressor(family="poisson", **model_parameters)

    fit_params = {
        "X": data["X"],
        "y": data["y"],
    }

    if use_weights:
        fit_params.update({"sample_weight": data["weights"]})
        run_name = f"{run_name}_weights"

    model.fit(**fit_params)

    with open(git_root(f"golden_master/gm_{run_name}.pkl"), "wb") as fh:
        pickle.dump(model, fh)


if __name__ == "__main__":
    data = create_poisson_reg_data()

    for mdl_param in gm_model_parameters.items():
        for use_weights in [True, False]:
            run_and_store_golden_master(mdl_param[1], mdl_param[0], use_weights, data)
