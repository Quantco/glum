import argparse
import copy
import json
import warnings

import numpy as np
import pytest
import tabmat as mx
from git_root import git_root
from scipy import sparse

from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
from glum._glm import TweedieDistribution
from glum_benchmarks.data import simulate_glm_data

distributions_to_test = ["normal", "poisson", "gamma", "tweedie_p=1.5", "binomial"]
custom_family_link = [("normal", "log")]

# Not the canonical link for tweedie and gamma, but what the actuaries use.
link_map = {
    "normal": "identity",
    "poisson": "log",
    "gamma": "log",
    "tweedie_p=1.5": "log",
    "binomial": "logit",
}


def _make_P2():
    rand = np.random.default_rng(1)
    a = rand.uniform(size=(30, 30)) - 0.5  # centered uniform distribution
    P2 = a.T @ a  # make sure P2 is positive semi-definite
    return P2


@pytest.fixture(scope="module")
def data_all():
    return {
        dist: simulate_glm_data(
            family=dist,
            link=link_map[dist],
            n_rows=5000,
            dense_features=10,
            sparse_features=0,
            categorical_features=2,
            categorical_levels=10,
            ohe_categorical=True,
            drop_first=False,
        )
        for dist in distributions_to_test
    }


@pytest.fixture(
    params=["categorical"],
    scope="module",
)
def data_all_storage(request):
    data = {}
    for dist in distributions_to_test:
        data_config = {
            "family": dist,
            "link": link_map[dist],
            "n_rows": 5000,
            "dense_features": 10,
            "categorical_features": 2,
            "categorical_levels": 10,
        }

        if request.param == "dense":
            data_dist = simulate_glm_data(**data_config, ohe_categorical=True)
            data_dist["X"] = mx.DenseMatrix(data_dist["X"])
        elif request.param == "scipy-sparse":
            data_dist = simulate_glm_data(**data_config, ohe_categorical=True)
            data_dist["X"] = sparse.csc_matrix(data_dist["X"])
        elif request.param == "mkl-sparse":
            data_dist = simulate_glm_data(**data_config, ohe_categorical=True)
            data_dist["X"] = mx.SparseMatrix(sparse.csc_matrix(data_dist["X"]))
        elif request.param == "split":
            data_dist = simulate_glm_data(**data_config, ohe_categorical=True)
            data_dist["X"] = mx.csc_to_split(
                sparse.csc_matrix(data_dist["X"]), threshold=0.1
            )
        elif request.param == "categorical":
            data_dist = simulate_glm_data(**data_config, ohe_categorical=False)
            dense_X = mx.DenseMatrix(np.ascontiguousarray(data_dist["X"].iloc[:, :10]))
            cat0 = mx.CategoricalMatrix(data_dist["X"]["cat0"])
            cat1 = mx.CategoricalMatrix(data_dist["X"]["cat1"])
            data_dist["X"] = mx.SplitMatrix([dense_X, cat0, cat1])

        data[dist] = data_dist

    return data


@pytest.fixture(scope="module")
def expected_all():
    with open(git_root("tests/glm/golden_master/simulation_gm.json")) as fh:
        return json.load(fh)


gm_model_parameters = {
    "default": {},  # default params
    "half-regularization": {"alpha": 0.5},  # regularization (other than alpha = 1)
    "elastic-net": {"l1_ratio": 0.5},  # elastic-net
    "lasso": {"l1_ratio": 1},  # lasso
    "variable_p1": {
        "l1_ratio": 1,
        "P1": np.arange(30) / 10,
    },  # lasso with variable penalty
    "variable_p2": {
        "l1_ratio": 0,
        "P2": _make_P2(),
    },  # ridge with Tikhonov regularization
    "variable_p1_p2": {
        "l1_ratio": 0.5,
        "P1": np.arange(30) / 10,
        "P2": _make_P2(),
    },  # elastic net with P1 and P2 variable penalty
    "fit_intercept": {"fit_intercept": False},  # do not fit the intercept
    "bounds": {"lower_bounds": np.full(30, 0), "upper_bounds": np.full(30, 0.4)},
}


def assert_gm_allclose(model, expected, rtol=0, atol=1e-5):
    np.testing.assert_allclose(
        model.coef_, np.array(expected["coef_"]), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        model.intercept_, expected["intercept_"], rtol=rtol, atol=atol
    )


def fit_model(data, family, model_parameters, use_weights, use_offset, cv):
    if "tweedie" in family:
        p = float(family.split("=")[1])
        family = TweedieDistribution(power=p)
    if cv:
        model = GeneralizedLinearRegressorCV(
            family=family, gradient_tol=1e-6, **model_parameters
        )
    else:
        model = GeneralizedLinearRegressor(
            family=family, gradient_tol=1e-6, **model_parameters
        )

    fit_params = {
        "X": data["X"],
        "y": data["y"],
    }
    if use_weights:
        fit_params.update({"sample_weight": data["sample_weight"]})
    if use_offset:
        fit_params.update({"offset": data["offset"]})

    model.fit(**fit_params)
    return model


@pytest.mark.parametrize(
    "distribution",
    distributions_to_test,
    ids=distributions_to_test,
)
@pytest.mark.parametrize(
    ["run_name", "model_parameters"],
    gm_model_parameters.items(),
    ids=gm_model_parameters.keys(),
)
@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
@pytest.mark.parametrize("use_offset", [True, False], ids=["offset", "no_offset"])
def test_gm_features(
    distribution,
    model_parameters,
    run_name,
    use_weights,
    use_offset,
    data_all,
    expected_all,
):
    data = data_all[distribution]
    model = fit_model(
        data=data,
        family=distribution,
        model_parameters=model_parameters,
        use_weights=use_weights,
        use_offset=use_offset,
        cv=False,
    )

    if use_weights:
        run_name = f"{run_name}_weights"
    if use_offset:
        run_name = f"{run_name}_offset"
    expected = expected_all[distribution][run_name]

    assert_gm_allclose(model, expected)


@pytest.mark.parametrize(
    "distribution",
    distributions_to_test,
    ids=distributions_to_test,
)
def test_gm_storage(distribution, data_all_storage, expected_all):
    data = data_all_storage[distribution]
    model = fit_model(
        data=data,
        family=distribution,
        model_parameters={},
        use_weights=False,
        use_offset=False,
        cv=False,
    )

    run_name = "default"
    expected = expected_all[distribution][run_name]

    assert_gm_allclose(model, expected)


@pytest.mark.parametrize("family_link", custom_family_link)
@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
@pytest.mark.parametrize("use_offset", [True, False], ids=["offset", "no_offset"])
def test_gm_custom_link(family_link, use_weights, use_offset, data_all, expected_all):
    """Currently only testing log-linear model."""
    distribution, link = family_link
    data = data_all[distribution]
    model_parameters = {"link": link}
    model = fit_model(
        data=data,
        family=distribution,
        model_parameters=model_parameters,
        use_weights=use_weights,
        use_offset=use_offset,
        cv=False,
    )

    run_name = f"custom-{distribution}-{link}"
    if use_weights:
        run_name = f"{run_name}_weights"
    if use_offset:
        run_name = f"{run_name}_offset"
    expected = expected_all[distribution][run_name]

    assert_gm_allclose(model, expected)


@pytest.mark.parametrize(
    "distribution",
    distributions_to_test,
    ids=distributions_to_test,
)
@pytest.mark.parametrize("use_weights", [True, False], ids=["weights", "no_weights"])
@pytest.mark.parametrize("use_offset", [True, False], ids=["offset", "no_offset"])
def test_gm_approx_hessian(
    distribution, use_weights, use_offset, data_all, expected_all
):
    data = data_all[distribution]
    model_parameters = {
        "hessian_approx": 0.1,
    }
    model = fit_model(
        data=data,
        family=distribution,
        model_parameters=model_parameters,
        use_weights=use_weights,
        use_offset=use_offset,
        cv=False,
    )

    run_name = "default"
    if use_weights:
        run_name = f"{run_name}_weights"
    if use_offset:
        run_name = f"{run_name}_offset"

    expected = expected_all[distribution][run_name]
    assert_gm_allclose(model, expected)


@pytest.mark.parametrize(
    "distribution",
    distributions_to_test,
    ids=distributions_to_test,
)
def test_gm_cv(distribution, data_all, expected_all):
    data = data_all[distribution]
    model_parameters = {
        "alphas": [0.1, 0.05, 0.01],
        "l1_ratio": [0.2, 0.5, 0.9],
        "cv": 3,
    }
    model = fit_model(
        data=data,
        family=distribution,
        model_parameters=model_parameters,
        use_weights=False,
        use_offset=False,
        cv=True,
    )

    run_name = f"CV-{distribution}"

    expected = expected_all[distribution][run_name]
    assert_gm_allclose(model, expected)


@pytest.mark.parametrize(
    "dist_power",
    [("poisson", 1), ("gamma", 2), ("tweedie_p=1.5", 1.5)],
)
def test_weights_offset_equivalence(dist_power, data_all):
    distribution, power = dist_power

    weights_data = data_all[distribution]
    offset_data = copy.copy(weights_data)

    exposure = np.exp(weights_data["offset"])
    weights_data["sample_weight"] = exposure ** (2 - power)
    weights_data["y"] = weights_data["y"] / exposure

    weights_parameters = {"alpha": 0.1 / weights_data["sample_weight"].mean()}
    model_weights = fit_model(
        data=weights_data,
        family=distribution,
        model_parameters=weights_parameters,
        use_weights=True,
        use_offset=False,
        cv=False,
    )

    offset_parameters = {"alpha": 0.1}
    model_offset = fit_model(
        data=offset_data,
        family=distribution,
        model_parameters=offset_parameters,
        use_weights=False,
        use_offset=True,
        cv=False,
    )

    np.testing.assert_allclose(
        model_weights.coef_, model_offset.coef_, rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        model_weights.intercept_, model_offset.intercept_, rtol=0, atol=1e-5
    )


def run_and_store_golden_master(
    distribution,
    model_parameters,
    run_name,
    use_weights,
    use_offset,
    cv,
    data,
    gm_dict,
    overwrite=False,
):
    if use_weights:
        run_name = f"{run_name}_weights"
    if use_offset:
        run_name = f"{run_name}_offset"

    if distribution not in gm_dict.keys():
        gm_dict[distribution] = {}

    if run_name in gm_dict[distribution].keys():
        if overwrite:
            warnings.warn("Overwriting existing result")
        else:
            warnings.warn("Result exists and cannot overwrite. Skipping")
            return gm_dict

    model = fit_model(
        data=data,
        family=distribution,
        model_parameters=model_parameters,
        use_weights=use_weights,
        use_offset=use_offset,
        cv=cv,
    )

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
        with open(git_root("tests/glm/golden_master/simulation_gm.json")) as fh:
            gm_dict = json.load(fh)
    except FileNotFoundError:
        gm_dict = {}

    for dist in distributions_to_test:
        data = simulate_glm_data(family=dist, link=link_map[dist])
        for mdl_param in gm_model_parameters.items():
            for use_weights in [True, False]:
                for use_offset in [True, False]:
                    gm_dict = run_and_store_golden_master(
                        distribution=dist,
                        model_parameters=mdl_param[1],
                        run_name=mdl_param[0],
                        use_weights=use_weights,
                        use_offset=use_offset,
                        cv=False,
                        data=data,
                        gm_dict=gm_dict,
                        overwrite=args.overwrite,
                    )

    for dist in distributions_to_test:
        data = simulate_glm_data(family=dist, link=link_map[dist])
        gm_dict = run_and_store_golden_master(
            distribution=dist,
            model_parameters={
                "alphas": [0.1, 0.05, 0.01],
                "l1_ratio": [0.2, 0.5, 0.9],
                "cv": 3,
            },
            run_name=f"CV-{dist}",
            use_weights=use_weights,
            use_offset=use_offset,
            cv=True,
            data=data,
            gm_dict=gm_dict,
            overwrite=args.overwrite,
        )

    for dist, link in custom_family_link:
        data = simulate_glm_data(family=dist, link=link_map[dist])
        for use_weights in [True, False]:
            for use_offset in [True, False]:
                gm_dict = run_and_store_golden_master(
                    distribution=dist,
                    model_parameters={"link": link},
                    run_name=f"custom-{dist}-{link}",
                    use_weights=use_weights,
                    use_offset=use_offset,
                    cv=False,
                    data=data,
                    gm_dict=gm_dict,
                    overwrite=args.overwrite,
                )

    with open(git_root("tests/glm/golden_master/simulation_gm.json"), "w") as fh:
        json.dump(gm_dict, fh, indent=2)
