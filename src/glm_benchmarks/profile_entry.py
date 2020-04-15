import os
import pickle

import click
import numpy as np

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import execute_problem_library, get_limited_problems


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
            test_against_baseline(path, result)
        print("")


def save_baseline(path, data):
    print("saving baseline estimates for later testing.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_against_baseline(path, data):
    print("loading baseline estimates for testing")
    with open(path, "rb") as f:
        baseline = pickle.load(f)

    print(
        f'baseline intercept = {baseline["intercept"]}. new intercept = {data["intercept"]}.'
    )
    diff = data["coef"] - baseline["coef"]
    msd = np.sqrt(np.mean(diff ** 2))
    print(f"root mean square difference between coef: {msd}")
    print(f"max difference between coef: {np.max(np.abs(diff))}")
    print(f"baseline runtime = {baseline['runtime']}")
    print(f"current runtime = {data['runtime']}")


if __name__ == "__main__":
    main()
