import json
import warnings

import click
import numpy as np
import pytest
from git_root import git_root

from glum_benchmarks.cli_run import execute_problem_library
from glum_benchmarks.problems import Problem, get_all_problems
from glum_benchmarks.util import BenchmarkParams, get_obj_val

bench_cfg = dict(num_rows=10000, regularization_strength=0.1, diagnostics_level="none")

all_test_problems = get_all_problems()


def is_weights_problem_with_offset_match(problem_name):
    return (
        "no-weights" not in problem_name
        and "weights" in problem_name
        and (
            "gamma" in problem_name
            or "poisson" in problem_name
            or "tweedie" in problem_name
        )
    )


@pytest.fixture(scope="module")
def expected_all():
    with open(git_root("tests/glm/golden_master/benchmark_gm.json")) as fh:
        return json.load(fh)


@pytest.mark.parametrize(
    ["Pn", "P"],
    [
        x if "wide" not in x[0] else pytest.param(x[0], x[1], marks=pytest.mark.slow)
        for x in all_test_problems.items()
    ],  # mark the "wide" problems as "slow" so that we can call pytest -m "not slow"
    ids=all_test_problems.keys(),
)
def test_gm_benchmarks(Pn: str, P: Problem, expected_all: dict):
    result, params = exec(Pn)

    if is_weights_problem_with_offset_match(Pn):
        expected = expected_all["offset".join(Pn.split("weights"))]
    else:
        expected = expected_all[Pn]

    all_result = np.concatenate(([result["intercept"]], result["coef"]))
    all_expected = np.concatenate(([expected["intercept"]], expected["coef"]))

    try:
        np.testing.assert_allclose(all_result, all_expected, rtol=2e-4, atol=2e-4)
    except AssertionError as e:
        dat = P.data_loader(
            num_rows=params.num_rows,
        )
        obj_result = get_obj_val(
            dat,
            P.distribution,
            P.regularization_strength,
            P.l1_ratio,
            all_result[0],
            all_result[1:],
        )
        expected_result = get_obj_val(
            dat,
            P.distribution,
            P.regularization_strength,
            P.l1_ratio,
            all_expected[0],
            all_expected[1:],
        )
        raise AssertionError(
            f"""Failed with error {e} on problem {Pn}.
            New objective function value is higher by {obj_result - expected_result}."""
        ) from e


@click.command()
@click.option("--overwrite", is_flag=True, help="overwrite existing golden master")
@click.option(
    "--problem_name", default=None, help="Only run and store a specific problem."
)
def run_and_store_golden_master(overwrite, problem_name):

    try:
        with open(git_root("tests/glm/golden_master/benchmark_gm.json")) as fh:
            gm_dict = json.load(fh)
    except FileNotFoundError:
        gm_dict = {}

    for Pn in get_all_problems().keys():
        if is_weights_problem_with_offset_match(Pn):
            continue
        if problem_name is not None:
            if Pn != problem_name:
                continue

        res, params = exec(Pn)

        if Pn in gm_dict.keys():
            if overwrite:
                warnings.warn("Overwriting existing result")
            else:
                warnings.warn("Result exists and cannot overwrite. Skipping")
                continue

        gm_dict[Pn] = dict(
            coef=res["coef"].tolist(),
            intercept=res["intercept"],
        )

    with open(git_root("tests/glm/golden_master/benchmark_gm.json"), "w") as fh:
        json.dump(gm_dict, fh, indent=2)


def exec(Pn):
    execute_args = ["diagnostics_level"]
    params = BenchmarkParams(
        problem_name=Pn,
        library_name="glum",
        **{k: v for k, v in bench_cfg.items() if k not in execute_args},
    )
    if bench_cfg["diagnostics_level"] != "none":
        print("Running", Pn)

    result, _ = execute_problem_library(
        params, **{k: v for k, v in bench_cfg.items() if k in execute_args}
    )
    return result, params


if __name__ == "__main__":
    run_and_store_golden_master()
