import os
import pickle
from typing import Dict

import click
import numpy as np

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
    default="narrow_insurance_no_weights_lasso_poisson",
    help="Specify a comma-separated list of benchmark problems you want to run.",
)
@click.option(
    "--storage",
    type=str,
    default="dense",
    help="Specify the storage format. Currently supported: dense, sparse. Leaving this black will default to dense.",
)
@click.option(
    "--save_result",
    is_flag=True,
    help="Save the estimates for later golden master testing.",
)
@click.option(
    "--no_test", is_flag=True, help="Skip the test against the baseline.",
)
@click.option(
    "--save_dir",
    default="golden_master",
    help="Where to find saved estimates for checking that estimates haven't changed.",
)
def main(num_rows, problem_names, storage, save_result, no_test, save_dir):
    problems = get_limited_problems(problem_names)
    for Pn in problems:
        print(f"benchmarking {Pn}")
        result = execute_problem_library(
            problems[Pn], sklearn_fork_bench, num_rows=num_rows, storage=storage
        )
        print(f"took {result['runtime']}")

        path = os.path.join(save_dir, Pn, str(num_rows) + ".pkl")
        if save_result:
            save_baseline(path, result)
        elif not no_test:
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
    print("")
    print(f"baseline objective: {obj_val_baseline}")
    print(f"new objective: {obj_val_new}")
    print(f"diff objective: {obj_val_new - obj_val_baseline}")
    print("")

    print(
        f'baseline intercept = {baseline["intercept"]}. new intercept = {data["intercept"]}.'
    )
    print(f'intercept difference = {baseline["intercept"] - data["intercept"]}')
    diff = data["coef"] - baseline["coef"]
    msd = np.sqrt(np.mean(diff ** 2))
    print("")
    print(f"root mean square difference between coef: {msd}")
    print(f"max difference between coef: {np.max(np.abs(diff))}")
    print(f"median difference between coef: {np.median(np.abs(diff))}")
    print("")
    print(f"baseline runtime = {baseline['runtime']}")
    print(f"current runtime = {data['runtime']}")


if __name__ == "__main__":
    main()
