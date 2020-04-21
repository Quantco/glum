import pickle

import click
import scipy.sparse

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import get_limited_problems


@click.command()
@click.option(
    "--num_rows",
    type=int,
    default=int(1e6),
    help="Integer number of rows to run profiling on.",
)
@click.option(
    "--problem_name",
    default="simple_insurance_no_weights_lasso_poisson",
    help="Specify a single benchmark problem you want to run.",
)
@click.option(
    "--sparsify",
    is_flag=True,
    help="Convert an originally dense problem into a sparse one.",
)
@click.option(
    "--save", is_flag=True, help="Create the benchmark data, don't run the benchmark."
)
def main(num_rows, problem_name, sparsify, save):
    problems = get_limited_problems(problem_name)
    P = list(problems.values())[0]
    save_path = "memory_benchmark.pkl"
    if save:
        # Why do we save the data and re-load before profiling? It isolates the
        # GLM functionality rather than also including the data creation in the
        # benchmark.
        dat = P.data_loader(num_rows)
        if sparsify:
            dat["X"] = scipy.sparse.csc_matrix(dat["X"])
        with open(save_path, "wb") as f:
            pickle.dump(dat, f)
    else:
        with open(save_path, "rb") as f:
            dat = pickle.load(f)
        result = sklearn_fork_bench(
            dat, P.distribution, P.regularization_strength, P.l1_ratio
        )
        print(f"took {result['runtime']}")

        # import numpy as np
        # d = np.random.rand(num_rows)
        # from glm_benchmarks.scaled_spmat.standardize import zero_center
        # from glm_benchmarks.sklearn_fork._glm import _safe_sandwich_dot
        # dat['X'], means = zero_center(dat['X'])
        # _safe_sandwich_dot(dat['X'], d, intercept = True)


if __name__ == "__main__":
    main()
