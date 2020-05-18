import json
import warnings

import click
import numpy as np
import pytest
from git_root import git_root
from sklearn.exceptions import ConvergenceWarning

from glm_benchmarks.main import execute_problem_library
from glm_benchmarks.problems import get_all_problems
from glm_benchmarks.util import BenchmarkParams

bench_cfg = dict(
    num_rows=10000,
    regularization_strength=0.1,
    storage="dense",
    print_diagnostics=False,
)

all_test_problems = get_all_problems()


@pytest.fixture
def bench_cfg_fix():
    return bench_cfg


@pytest.fixture
def expected_all():
    with open(git_root("golden_master/benchmark_gm.json"), "r") as fh:
        return json.load(fh)


@pytest.mark.parametrize(
    ["Pn", "P"], all_test_problems.items(), ids=all_test_problems.keys()
)
def test_gm_benchmarks(Pn, P, bench_cfg_fix, expected_all):
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
    np.testing.assert_allclose(all_result, all_expected, rtol=2e-4, atol=2e-4)


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

    with open(git_root("golden_master/skipped_benchmark_gm.json"), "w") as fh:
        json.dump(skipped_problems, fh, indent=2)


if __name__ == "__main__":
    run_and_store_golden_master()
