import argparse
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from git_root import git_root
from scipy import sparse

from glm_benchmarks.scaled_spmat.mkl_sparse_matrix import MKLSparseMatrix
from glm_benchmarks.sklearn_fork._glm import (
    GeneralizedLinearRegressor,
    TweedieDistribution,
)
from glm_benchmarks.sklearn_fork.dense_glm_matrix import DenseGLMDataMatrix

distributions_to_test = ["normal", "poisson", "gamma", "tweedie_p=1.5"]

# Do not create a golden master for the following because the problem does not converge
problems_with_issue = [
    ("gamma", "fit_intercept"),
]


def tweedie_rv(p, mu, sigma2=1):
    """Generates draws from a tweedie distribution with power p.

    mu is the location parameter and sigma2 is the dispersion coefficient.
    """
    n = len(mu)
    rand = np.random.default_rng(1)

    # transform tweedie parameters into poisson and gamma
    lambda_ = (mu ** (2 - p)) / ((2 - p) * sigma2)
    alpha_ = (2 - p) / (p - 1)
    beta_ = (mu ** (1 - p)) / ((p - 1) * sigma2)

    arr_N = rand.poisson(lambda_)
    out = np.empty(n, dtype=np.float64)
    for i, N in enumerate(arr_N):
        out[i] = np.sum(rand.gamma(alpha_, 1 / beta_[i], size=N))

    return out


def create_reg_data(
    distribution="poisson", n_rows=5000, n_features_dense=10, n_features_ohe=2
):
    rand = np.random.default_rng(1)
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
    elif distribution == "gamma":
        y = rand.gamma(np.exp(intercept + X @ coefs))
    elif "tweedie" in distribution:
        p = float(distribution.split("=")[1])
        y = tweedie_rv(p, np.exp(intercept + X @ coefs))
    else:
        raise ValueError(f"{distribution} not supported as distribution")

    weights = rand.uniform(size=n_rows)
    data = {"intercept": intercept, "X": X, "b": coefs, "y": y, "weights": weights}
    return data


def _make_P2():
    rand = np.random.default_rng(1)
    a = rand.uniform(size=(28, 28)) - 0.5  # centered uniform distribution
    P2 = a.T @ a  # make sure P2 is positive semi-definite
    return P2


@pytest.fixture(params=["sparse", "dense"])
def data_all(request):
    data = dict()
    for dist in distributions_to_test:
        data_dist = create_reg_data(distribution=dist)

        if request.param == "dense":
            data_dist["X"] = DenseGLMDataMatrix(data_dist["X"])
        else:
            data_dist["X"] = MKLSparseMatrix(sparse.csc_matrix(data_dist["X"]))

        data[dist] = data_dist

    return data


@pytest.fixture
def expected_all():
    with open(git_root("golden_master/simulation_gm.json"), "r") as fh:
        return json.load(fh)


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
    if "tweedie" in family:
        p = float(family.split("=")[1])
        family = TweedieDistribution(power=p)
    model = GeneralizedLinearRegressor(family=family, **model_parameters)

    fit_params = {
        "X": data["X"],
        "y": data["y"],
    }
    if use_weights:
        fit_params.update({"sample_weight": data["weights"]})

    model.fit(**fit_params)
    return model


@pytest.mark.parametrize(
    "distribution",
    ["normal", "poisson", "gamma", "tweedie_p=1.5"],
    ids=["normal", "poisson", "gamma", "tweedie"],
)
@pytest.mark.parametrize(
    ["run_name", "model_parameters"],
    gm_model_parameters.items(),
    ids=gm_model_parameters.keys(),
)
@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
def test_golden_master(
    distribution, model_parameters, run_name, use_weights, data_all, expected_all
):
    if (distribution, run_name) in problems_with_issue:
        pytest.skip("Problem does not converge")

    data = data_all[distribution]
    model = fit_model(distribution, model_parameters, use_weights, data)

    if use_weights:
        run_name = f"{run_name}_weights"

    expected = expected_all[distribution][run_name]

    np.testing.assert_allclose(
        model.coef_, np.array(expected["coef_"]), rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(
        model.intercept_, expected["intercept_"], rtol=1e-5, atol=0
    )


def run_and_store_golden_master(
    distribution,
    model_parameters,
    run_name,
    use_weights,
    data,
    gm_dict,
    overwrite=False,
):
    print((distribution, run_name))
    if use_weights:
        run_name = f"{run_name}_weights"

    if distribution not in gm_dict.keys():
        gm_dict[distribution] = dict()

    if run_name in gm_dict[distribution].keys():
        if overwrite:
            warnings.warn("Overwriting existing result")
        else:
            warnings.warn("Result exists and cannot overwrite. Skipping")
            return gm_dict

    model = fit_model(distribution, model_parameters, use_weights, data)

    gm_dict[distribution][run_name] = dict(
        coef_=model.coef_.tolist(),
        intercept_=model.intercept_,
        n_iter_=model.n_iter_,  # not used right now but could in the future
    )
    return gm_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing golden master"
    )
    args = parser.parse_args()

    try:
        with open(git_root("golden_master/simulation_gm.json"), "r") as fh:
            gm_dict = json.load(fh)
    except FileNotFoundError:
        gm_dict = dict()

    for dist in distributions_to_test:
        data = create_reg_data(dist)
        for mdl_param in gm_model_parameters.items():
            for use_weights in [True, False]:
                if (dist, mdl_param[0]) in problems_with_issue:
                    continue
                gm_dict = run_and_store_golden_master(
                    dist,
                    mdl_param[1],
                    mdl_param[0],
                    use_weights,
                    data,
                    gm_dict,
                    args.overwrite,
                )

    with open(git_root("golden_master/simulation_gm.json"), "w") as fh:
        json.dump(gm_dict, fh, indent=2)
