import json
import warnings

import click
import numpy as np
import pytest
from git_root import git_root
from sklearn.exceptions import ConvergenceWarning

from glm_benchmarks.cli_run import execute_problem_library
from glm_benchmarks.problems import Problem, get_all_problems
from glm_benchmarks.util import BenchmarkParams, get_obj_val

bench_cfg = dict(
    num_rows=10000,
    regularization_strength=0.1,
    storage="dense",
    print_diagnostics=False,
)

all_test_problems = get_all_problems()


@pytest.fixture(scope="module")
def bench_cfg_fix():
    return bench_cfg


@pytest.fixture(scope="module")
def expected_all():
    with open(git_root("golden_master/benchmark_gm.json"), "r") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def skipped_benchmark_gm():
    try:
        with open(git_root("golden_master/skipped_benchmark_gm.json"), "r") as fh:
            skipped_problems = json.load(fh)
    except FileNotFoundError:
        skipped_problems = []
    return skipped_problems


@pytest.mark.parametrize(
    ["Pn", "P"],
    [
        x if "wide" not in x[0] else pytest.param(x[0], x[1], marks=pytest.mark.slow)
        for x in all_test_problems.items()
    ],  # mark the "wide" problems as "slow" so that we can call pytest -m "not slow"
    ids=all_test_problems.keys(),
)
def test_gm_benchmarks(
    Pn: str,
    P: Problem,
    bench_cfg_fix: dict,
    expected_all: dict,
    skipped_benchmark_gm: list,
):
    if Pn in skipped_benchmark_gm:
        pytest.skip("Skipping problem with convergence issue.")

    execute_args = ["print_diagnostics"]
    params = BenchmarkParams(
        problem_name=Pn,
        library_name="sklearn-fork",
        **{k: v for k, v in bench_cfg_fix.items() if k not in execute_args},
    )
    if bench_cfg_fix["print_diagnostics"]:
        print(Pn)

    result, _ = execute_problem_library(
        params, **{k: v for k, v in bench_cfg_fix.items() if k in execute_args}
    )

    expected = expected_all[Pn]

    all_result = np.concatenate(([result["intercept"]], result["coef"]))
    all_expected = np.concatenate(([expected["intercept"]], expected["coef"]))

    try:
        np.testing.assert_allclose(all_result, all_expected, rtol=2e-4, atol=2e-4)
    except AssertionError as e:
        dat = P.data_loader(num_rows=params.num_rows,)
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
            f"""Failed with error {e}.
            New objective function value is higher by {obj_result - expected_result}."""
        )


@click.command()
@click.option("--overwrite", is_flag=True, help="overwrite existing golden master")
@click.option(
    "--problem_name", default=None, help="Only run and store a specific problem."
)
def run_and_store_golden_master(overwrite, problem_name):

    try:
        with open(git_root("golden_master/benchmark_gm.json"), "r") as fh:
            gm_dict = json.load(fh)
    except FileNotFoundError:
        gm_dict = dict()

    try:
        with open(git_root("golden_master/skipped_benchmark_gm.json"), "r") as fh:
            skipped_problems = json.load(fh)
    except FileNotFoundError:
        skipped_problems = []

    for Pn, P in get_all_problems().items():
        if problem_name is not None:
            if Pn != problem_name:
                continue

        if Pn in gm_dict.keys():
            if overwrite:
                warnings.warn("Overwriting existing result")
            else:
                warnings.warn("Result exists and cannot overwrite. Skipping")
                continue

        params = BenchmarkParams(
            problem_name=Pn,
            library_name="sklearn-fork",
            **{k: v for k, v in bench_cfg.items() if k != "print_diagnostics"},
        )
        warnings.simplefilter("error", ConvergenceWarning)
        skipped = False
        try:
            print(f"Running {Pn}")
            res, _ = execute_problem_library(params)
        except ConvergenceWarning:
            warnings.warn("Problem does not converge. Not storing result.")
            skipped = True

        if skipped:
            if Pn not in skipped_problems:
                skipped_problems.append(Pn)
        else:
            if Pn in skipped_problems:
                skipped_problems.remove(Pn)
            gm_dict[Pn] = dict(coef=res["coef"].tolist(), intercept=res["intercept"],)

    with open(git_root("golden_master/benchmark_gm.json"), "w") as fh:
        json.dump(gm_dict, fh, indent=2)

    if len(skipped_problems > 0):
        with open(git_root("golden_master/skipped_benchmark_gm.json"), "w") as fh:
            json.dump(skipped_problems, fh, indent=2)


if __name__ == "__main__":
    run_and_store_golden_master()
