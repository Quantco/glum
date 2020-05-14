import argparse
import json
import warnings

import numpy as np
import pytest
from git_root import git_root
from sklearn.exceptions import ConvergenceWarning

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import execute_problem_library
from glm_benchmarks.problems import get_all_problems

bench_cfg = dict(
    num_rows=10000,
    regularization_strength=0.1,
    storage="dense",
    print_diagnostics=False,
)

with open(git_root("golden_master/skipped_benchmark_gm.json"), "r") as fh:
    do_not_test = json.load(fh)

all_test_problems = {
    key: value for key, value in get_all_problems().items() if key not in do_not_test
}


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
    if "tweedie" not in Pn:
        return

    result, _ = execute_problem_library(P, sklearn_fork_bench, **bench_cfg_fix)

    expected = expected_all[Pn]

    all_result = np.concatenate(([result["intercept"]], result["coef"]))
    all_expected = np.concatenate(([expected["intercept"]], expected["coef"]))
    np.testing.assert_allclose(all_result, all_expected, rtol=1e-4, atol=1e-4)


def run_and_store_golden_master(overwrite=False):
    skipped_problems = []

    if overwrite:
        gm_dict = dict()
    else:
        try:
            with open(git_root("golden_master/benchmark_gm.json"), "r") as fh:
                gm_dict = json.load(fh)
        except FileNotFoundError:
            gm_dict = dict()

    for Pn, P in get_all_problems().items():
        if Pn in gm_dict.keys():
            if overwrite:
                warnings.warn("Overwriting existing result")
            else:
                warnings.warn("Result exists and cannot overwrite. Skipping")
                continue

        warnings.simplefilter("error", ConvergenceWarning)
        try:
            res = execute_problem_library(P, sklearn_fork_bench, **bench_cfg)
        except ConvergenceWarning:
            warnings.warn("Problem does not converge. Not storing result.")
            skipped_problems.append(Pn)
            continue

        gm_dict[Pn] = dict(coef=res["coef"].tolist(), intercept=res["intercept"],)

    with open(git_root("golden_master/benchmark_gm.json"), "w") as fh:
        json.dump(gm_dict, fh, indent=2)

    with open(git_root("golden_master/skipped_benchmark_gm.json"), "w") as fh:
        json.dump(skipped_problems, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing golden master"
    )
    args = parser.parse_args()

    run_and_store_golden_master(args.overwrite)
