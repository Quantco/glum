import os
import pickle
from typing import Dict

import click
import numpy as np
import pandas as pd

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import execute_problem_library, get_limited_problems
from glm_benchmarks.util import get_obj_val


@click.command()
@click.option(
    "--num_rows",
    type=int,
    default=50000,
    help="Integer number of rows to run profiling on.",
)
@click.option(
    "--problem_names",
    default="simple_insurance_no_weights_lasso_poisson",
    help="Specify a comma-separated list of benchmark problems you want to run.",
)
@click.option(
    "--sparsify",
    is_flag=True,
    help="Convert an originally dense problem into a sparse one.",
)
@click.option(
    "--save_result",
    is_flag=True,
    help="Save the estimates for later golden master testing.",
)
@click.option(
    "--save_dir",
    default="golden_master",
    help="Where to find saved estimates for checking that estimates haven't changed.",
)
def main(num_rows, problem_names, sparsify, save_result, save_dir):
    problems = get_limited_problems(problem_names)
    for Pn in problems:
        print(f"benchmarking {Pn}")
        result = execute_problem_library(
            problems[Pn], sklearn_fork_bench, num_rows, sparsify
        )

        path = os.path.join(save_dir, Pn, str(num_rows) + ".pkl")
        if save_result:
            save_baseline(path, result)
        else:
            test_against_baseline(path, result, Pn, num_rows)
        print("")


def save_baseline(path, data):
    print("saving baseline estimates for later testing.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_against_baseline(path: str, data: Dict, prob_name: str, num_rows: int):
    print("loading baseline estimates for testing")
    with open(path, "rb") as f:
        baseline = pickle.load(f)

    problem = get_limited_problems(prob_name)[prob_name]

    def get_obj(intercept: float, coef: np.ndarray) -> float:
        return get_obj_val(
            problem.data_loader(num_rows),
            problem.distribution,
            problem.regularization_strength,
            problem.l1_ratio,
            intercept,
            coef,
        )

    obj_val_baseline = get_obj(baseline["intercept"], baseline["coef"])
    obj_val_new = get_obj(data["intercept"], data["coef"])
    results = pd.DataFrame(
        columns=["baseline", "new"],
        index=["obj", "intercept", "last_coef", "runtime"],
        data=[
            [obj_val_baseline, obj_val_new],
            [baseline["intercept"], data["intercept"]],
            [baseline["coef"][-1], data["coef"][-1]],
            [baseline["runtime"], data["runtime"]],
        ],
    )
    results["diff"] = results["new"] - results["baseline"]
    print(results)
    np.testing.assert_almost_equal(data["intercept"], baseline["intercept"])
    np.testing.assert_almost_equal(data["coef"], baseline["coef"])
    print("test passed")
    print(f"baseline runtime = {baseline['runtime']}")
    print(f"current runtime = {data['runtime']}")


if __name__ == "__main__":
    main()
