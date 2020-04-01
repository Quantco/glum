import os
import pickle
import warnings
from typing import Dict, List, Tuple

import click
import numpy as np

from glm_benchmarks.bench_glmnet_python import glmnet_python_bench
from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.bench_tensorflow import tensorflow_bench
from glm_benchmarks.problems import get_all_problems


@click.command()
@click.option(
    "--problem_names",
    default="",
    help="Specify a comma-separated list of benchmark problems you want to run. Leaving this blank will default to running all problems.",
)
@click.option(
    "--library_names",
    default="",
    help="Specify a comma-separated list of libaries to benchmark. Leaving this blank will default to running all problems.",
)
@click.option(
    "--num_rows",
    type=int,
    help="Pass an integer number of rows. This is useful for testing and development. The default is to use the full dataset.",
)
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory to store benchmarking output.",
)
def cli_run(problem_names, library_names, num_rows, output_dir):
    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    for Pn, P in problems.items():
        for Ln, L in libraries.items():
            print(f"running problem={Pn} library={Ln}")
            dat = P.data_loader(num_rows=num_rows)
            result = L(dat, P.distribution, P.regularization_strength, P.l1_ratio)
            save_benchmark_results(output_dir, Pn, Ln, result)


@click.command()
@click.option(
    "--problem_names",
    default="",
    help="Specify a comma-separated list of benchmark problems you want to analyze. Leaving this blank will default to analyzing all problems.",
)
@click.option(
    "--library_names",
    default="",
    help="Specify a comma-separated list of libaries to analyze. Leaving this blank will default to analyzing all problems.",
)
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory where we load benchmarking output.",
)
def cli_analyze(problem_names: str, library_names: str, output_dir: str):
    np.set_printoptions(precision=4, suppress=True)
    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    # TODO: support more than a pair of libraries?
    # NOTE: might be better to leave this more ad-hoc until the dust settles a bit.

    for Pn in problems:
        print("")
        print(f"for {Pn}")

        results = dict()
        for Ln in libraries:
            res = load_benchmark_results(output_dir, Pn, Ln)
            if len(res) == 0:
                warnings.warn(f"Did not solve problem {Pn} in library {Ln}.")
            else:
                results[Ln] = res
                print(Ln, "number of iterations", results[Ln]["n_iter"])
                print(Ln, "runtime", results[Ln]["runtime"])
                print(
                    Ln,
                    "runtime per iter",
                    results[Ln]["runtime"] / results[Ln]["n_iter"],
                )

        if "glmnet_python" in results.keys() and "sklearn_fork" in results.keys():
            print("Difference in coefficients:")
            print(results["glmnet_python"]["coef"] - results["sklearn_fork"]["coef"])


def get_limited_problems_libraries(
    problem_names: str, library_names: str
) -> Tuple[Dict, Dict]:
    all_problems = get_all_problems()
    all_libraries = dict(
        sklearn_fork=sklearn_fork_bench,
        glmnet_python=glmnet_python_bench,
        tensorflow=tensorflow_bench,
    )

    if len(problem_names) > 0:
        problem_names_split = get_comma_sep_names(problem_names)
        problems = {k: all_problems[k] for k in problem_names_split}
    else:
        problems = all_problems

    if len(library_names) > 0:
        library_names_split = get_comma_sep_names(library_names)
        libraries = {k: all_libraries[k] for k in library_names_split}
    else:
        libraries = all_libraries
    return problems, libraries


def get_comma_sep_names(xs: str) -> List[str]:
    return [x.strip() for x in xs.split(",")]


def save_benchmark_results(output_dir, problem_name, library_name, result):
    problem_dir = os.path.join(output_dir, problem_name)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    with open(os.path.join(problem_dir, library_name + "-results.pkl"), "wb") as f:
        pickle.dump(result, f)


def load_benchmark_results(output_dir, problem_name, library_name):
    problem_dir = os.path.join(output_dir, problem_name)
    with open(os.path.join(problem_dir, library_name + "-results.pkl"), "rb") as f:
        return pickle.load(f)
