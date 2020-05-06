import argparse
import os
import pickle
import warnings

import numpy as np
import pytest
from git_root import git_root

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import execute_problem_library
from glm_benchmarks.problems import get_all_problems

do_not_test = [
    "wide_insurance_weights_l2_gamma",
    "wide_insurance_no_weights_l2_gamma",
    "wide_insurance_offset_l2_gamma",
    "wide_insurance_weights_l2_tweedie_p=1.5",
    "wide_insurance_no_weights_l2_tweedie_p=1.5",
    "wide_insurance_offset_l2_tweedie_p=1.5",
    "wide_insurance_weights_net_poisson",
    "wide_insurance_no_weights_net_poisson",
    "wide_insurance_offset_net_poisson",
    "wide_insurance_weights_net_gamma",
    "wide_insurance_no_weights_net_gamma",
    "wide_insurance_offset_net_gamma",
    "wide_insurance_weights_net_tweedie_p=1.5",
    "wide_insurance_no_weights_net_tweedie_p=1.5",
    "wide_insurance_offset_net_tweedie_p=1.5",
    "wide_insurance_weights_lasso_gaussian",
    "wide_insurance_no_weights_lasso_gaussian",
    "wide_insurance_offset_lasso_gaussian",
    "wide_insurance_weights_lasso_poisson",
    "wide_insurance_no_weights_lasso_poisson",
    "wide_insurance_offset_lasso_poisson",
    "wide_insurance_weights_lasso_gamma",
    "wide_insurance_no_weights_lasso_gamma",
    "wide_insurance_offset_lasso_gamma",
    "wide_insurance_weights_lasso_tweedie_p=1.5",
    "wide_insurance_no_weights_lasso_tweedie_p=1.5",
    "wide_insurance_offset_lasso_tweedie_p=1.5",
]

all_test_problems = {
    key: value for key, value in get_all_problems().items() if key not in do_not_test
}


@pytest.mark.parametrize(
    ["Pn", "P"], all_test_problems.items(), ids=all_test_problems.keys()
)
def test_gm_benchmarks(Pn, P):
    result = execute_problem_library(
        P,
        sklearn_fork_bench,
        num_rows=10000,
        storage="dense",
        print_diagnostics=False,
        regularization_strength=0.1,
    )

    with open(git_root(f"golden_master/benchmarks_gm/{Pn}.pkl"), "rb") as fh:
        expected = pickle.load(fh)

    for thing_to_test in ["intercept", "coef"]:
        np.testing.assert_allclose(
            result[thing_to_test], expected[thing_to_test], rtol=1e-4, atol=1e-4
        )


def run_and_store_golden_master(overwrite=False):
    for Pn, P in get_all_problems().items():
        if Pn in do_not_test:
            warnings.warn(f"Skipping {Pn} because it's not converging.")
            continue

        if os.path.exists(git_root(f"golden_master/benchmarks_gm/{Pn}.pkl")):
            if not overwrite:
                warnings.warn("File exists and cannot overwrite. Skipping")
                continue
            else:
                warnings.warn("Overwriting existing file")

        res = execute_problem_library(
            P,
            sklearn_fork_bench,
            num_rows=10000,
            storage="dense",
            print_diagnostics=False,
            regularization_strength=0.1,
        )

        with open(git_root(f"golden_master/benchmarks_gm/{Pn}.pkl"), "wb") as fh:
            pickle.dump(res, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing golden master"
    )
    args = parser.parse_args()

    run_and_store_golden_master(args.overwrite)
