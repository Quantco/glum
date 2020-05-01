import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from git_root import git_root
from scipy import sparse

from glm_benchmarks.scaled_spmat.mkl_sparse_matrix import MKLSparseMatrix
from glm_benchmarks.sklearn_fork._glm import GeneralizedLinearRegressor
from glm_benchmarks.sklearn_fork.dense_glm_matrix import DenseGLMDataMatrix


def create_reg_data(
    distribution="poisson", n_rows=10000, n_features_dense=10, n_features_ohe=2
):
    rand = np.random.default_rng(42)
    X = rand.standard_normal(size=(n_rows, n_features_dense))
    coefs = np.array([1.0, 0.5, 0.1, -0.1, -0.5, -1.0, 0, 0, 0, 0])

    for i in range(n_features_ohe):
        X = np.concatenate(
            [X, pd.get_dummies(rand.integers(0, 10, size=(n_rows)), drop_first=True)],
            axis=1,
        )
        coefs = np.concatenate([coefs, rand.uniform(size=9)])

    intercept = 0.2
    if distribution == "poisson":
        y = rand.poisson(np.exp(intercept + X @ coefs))
    elif distribution == "normal":
        y = intercept + X @ coefs + rand.standard_normal(size=n_rows)
    else:
        raise ValueError("Only poisson and normal are allowed currently")

    weights = rand.uniform(size=n_rows)
    data = {"intercept": intercept, "X": X, "b": coefs, "y": y, "weights": weights}
    return data


def _make_P2():
    rand = np.random.default_rng(42)
    a = rand.uniform(size=(28, 28)) - 0.5  # centered uniform distribution
    P2 = a.T @ a  # make sure P2 is positive semi-definite
    return P2


@pytest.fixture(params=["sparse", "dense"])
def poisson_data(request):
    data = create_reg_data(distribution="poisson")

    if request.param == "dense":
        data["X"] = DenseGLMDataMatrix(data["X"])
    else:
        data["X"] = MKLSparseMatrix(sparse.csc_matrix(data["X"]))

    return data


@pytest.fixture(params=["sparse", "dense"])
def normal_data(request):
    data = create_reg_data(distribution="normal")

    if request.param == "dense":
        data["X"] = DenseGLMDataMatrix(data["X"])
    else:
        data["X"] = MKLSparseMatrix(sparse.csc_matrix(data["X"]))

    return data


gm_model_parameters = {
    "default": {},  # default params
    "no-regularization": {"alpha": 0},  # no-regularization
    "half-regularization": {"alpha": 0.5},  # regularization (other than alpha = 1)
    "elastic-net": {"l1_ratio": 0.5},  # elastic-net
    "lasso": {"l1_ratio": 1},  # lasso
    "variable_p1": {
        "l1_ratio": 1,
        "P1": np.arange(28) / 10,
    },  # lasso with variable penalty
    "variable_p2": {
        "l1_ratio": 0,
        "P2": _make_P2(),
    },  # ridge with Tikhonov regularization
    "variable_p1_p2": {
        "l1_ratio": 0.5,
        "P1": np.arange(28) / 10,
        "P2": _make_P2(),
    },  # elastic net with P1 and P2 variable penalty
    "fit_intercept": {"fit_intercept": False},  # do not fit the intercept
}


def fit_model(family, model_parameters, use_weights, data):
    model = GeneralizedLinearRegressor(family=family, **model_parameters)

    fit_params = {
        "X": data["X"],
        "y": data["y"],
    }
    if use_weights:
        fit_params.update({"sample_weight": data["weights"]})

    model.fit(**fit_params)
    return model


@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
@pytest.mark.parametrize(
    ["run_name", "model_parameters"],
    gm_model_parameters.items(),
    ids=gm_model_parameters.keys(),
)
def test_poisson_golden_master(model_parameters, run_name, use_weights, poisson_data):
    model = fit_model("poisson", model_parameters, use_weights, poisson_data)

    if use_weights:
        run_name = f"{run_name}_weights"

    with open(git_root(f"golden_master/poisson/gm_{run_name}.pkl"), "rb") as fh:
        expected = pickle.load(fh)

    np.testing.assert_allclose(model.coef_, expected.coef_, rtol=1e-5, atol=0)
    np.testing.assert_allclose(model.intercept_, expected.intercept_, rtol=1e-5, atol=0)


@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
@pytest.mark.parametrize(
    ["run_name", "model_parameters"],
    gm_model_parameters.items(),
    ids=gm_model_parameters.keys(),
)
def test_gaussian_golden_master(model_parameters, run_name, use_weights, normal_data):
    model = fit_model("normal", model_parameters, use_weights, normal_data)

    if use_weights:
        run_name = f"{run_name}_weights"

    with open(git_root(f"golden_master/normal/gm_{run_name}.pkl"), "rb") as fh:
        expected = pickle.load(fh)

    np.testing.assert_allclose(model.coef_, expected.coef_, rtol=1e-5, atol=0)
    np.testing.assert_allclose(model.intercept_, expected.intercept_, rtol=1e-5, atol=0)


def run_and_store_golden_master(
    distribution, model_parameters, run_name, use_weights, data, overwrite=False
):
    if use_weights:
        run_name = f"{run_name}_weights"

    if os.path.exists(git_root(f"golden_master/{distribution}/gm_{run_name}.pkl")):
        if not overwrite:
            warnings.warn("File exists and cannot overwrite. Skipping")
            return
        else:
            warnings.warn("Overwriting existing file")

    model = fit_model(distribution, model_parameters, use_weights, data)

    with open(git_root(f"golden_master/{distribution}/gm_{run_name}.pkl"), "wb") as fh:
        pickle.dump(model, fh)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing golden master"
    )
    args = parser.parse_args()

    # Poisson
    poisson_dat = create_reg_data("poisson")
    for mdl_param in gm_model_parameters.items():
        for use_weights in [True, False]:
            run_and_store_golden_master(
                "poisson",
                mdl_param[1],
                mdl_param[0],
                use_weights,
                poisson_dat,
                args.overwrite,
            )

    # Gaussian
    normal_dat = create_reg_data("normal")
    for mdl_param in gm_model_parameters.items():
        for use_weights in [True, False]:
            run_and_store_golden_master(
                "normal",
                mdl_param[1],
                mdl_param[0],
                use_weights,
                normal_dat,
                args.overwrite,
            )
