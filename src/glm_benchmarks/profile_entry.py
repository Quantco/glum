import click
import numpy as np

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.main import execute_problem_library
from glm_benchmarks.problems import get_all_problems

intercept = -3.366495969747295
coef = np.array(
    [
        0.27608816,
        0.35549287,
        0.32686169,
        -0.26129063,
        0.05631971,
        1.66835335,
        0.0,
        0.0,
        -0.25640353,
        0.25240303,
        -0.04934933,
        0.00838973,
        -0.08514918,
        0.01641579,
        0.0,
        0.0,
        0.54944044,
        0.0,
        0.0,
        -0.03864504,
        -0.03402387,
        -0.05737517,
        -0.00961314,
        0.16551897,
        0.41539209,
        -0.08141213,
        0.03119013,
        -0.04565951,
        0.12508365,
        0.0,
        -0.05832944,
        0.0,
        -0.33902941,
        0.0,
        0.0,
        0.0,
        0.03173001,
        0.36435994,
        0.0,
        -0.33060842,
        0.13377216,
        0.0,
        0.09336626,
        -0.41353699,
        0.0,
        0.10139023,
        0.19088491,
        0.13317361,
    ]
)

# For line-by-line profiling, use line_profiler:
# kernprof -lbv src/glm_benchmarks/profile_entry.py
#
# For stack sampling profiling, use py-spy:
# py-spy record -o profile.svg -- python src/glm_benchmarks/profile_entry.py
# py-spy top -- python src/glm_benchmarks/profile_entry.py


@click.command()
@click.option(
    "--num_rows",
    type=int,
    default=50000,
    help="Integer number of rows to run profiling on.",
)
def main(num_rows):
    problems = get_all_problems()
    Pn = "sparse_insurance_no_weights_lasso_poisson"
    result = execute_problem_library(problems[Pn], sklearn_fork_bench, num_rows)
    if num_rows == 50000 and Pn == "simple_insurance_no_weights_lasso_poisson":
        np.testing.assert_almost_equal(result["intercept"], intercept)
        np.testing.assert_almost_equal(result["coef"], coef)


if __name__ == "__main__":
    main()
