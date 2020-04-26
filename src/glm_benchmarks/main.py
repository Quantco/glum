import os
import pickle
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import scipy.sparse

from glm_benchmarks.bench_qc_glmnet import glmnet_qc_bench
from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.problems import get_all_problems

from .util import get_obj_val
from .zeros_benchmark import zeros_bench

try:
    from glm_benchmarks.bench_glmnet_python import glmnet_python_bench  # isort:skip

    GLMNET_PYTHON_INSTALLED = True
except ImportError:
    GLMNET_PYTHON_INSTALLED = False

try:
    from glm_benchmarks.bench_h2o import h2o_bench  # isort:skip

    H20_INSTALLED = True
except ImportError:
    H20_INSTALLED = False


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
    "--storage",
    type=str,
    default="dense",
    help="Specify the storage format. Currently supported: dense, sparse. Leaving this black will default to dense.",
)
@click.option(
    "--threads",
    type=int,
    help="Specify the number of threads. If not set, it will use OMP_NUM_THREADS. If that's not set either, it will default to os.cpu_count().",
)
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory to store benchmarking output.",
)
def cli_run(
    problem_names: str,
    library_names: str,
    num_rows: int,
    storage: str,
    threads: int,
    output_dir: str,
):
    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    for Pn, P in problems.items():
        for Ln, L in libraries.items():
            print(f"running problem={Pn} library={Ln}")
            result = execute_problem_library(P, L, num_rows, storage, threads)
            save_benchmark_results(
                output_dir, Pn, Ln, num_rows, storage, threads, result
            )
            print("ran")


def execute_problem_library(P, L, num_rows=None, storage="dense", threads=None):
    dat = P.data_loader(num_rows=num_rows)
    if threads is None:
        threads = os.environ.get("OMP_NUM_THREADS", os.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(threads)
    if storage == "sparse":
        dat["X"] = scipy.sparse.csc_matrix(dat["X"])
    result = L(dat, P.distribution, P.regularization_strength, P.l1_ratio)
    return result


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
    "--num_rows",
    type=str,
    help="The number of rows that the GLM models were run with.",
)
@click.option(
    "--storage", type=str, help="Can be dense or sparse. Leave blank to analyze all.",
)
@click.option(
    "--threads",
    type=int,
    help="Specify the number of threads. Leave blank to analyze all.",
)
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory where we load benchmarking output.",
)
def cli_analyze(
    problem_names: str,
    library_names: str,
    num_rows: str,
    storage: str,
    threads: str,
    output_dir: str,
):
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)

    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    results: Dict[str, Dict[str, Dict[str, Any]]] = dict()
    for Pn in problems:
        results[Pn] = dict()

        # Find the row counts that have been used on this problem
        num_rows_used = (
            get_num_rows_used_to_solve_this_problem(output_dir, Pn)
            if num_rows is None
            else [str(num_rows)]
        )

        for num_rows_ in num_rows_used:
            results[Pn][num_rows_] = dict()

            # Find the storage formats that have been used on this problem
            storage_used = (
                get_storage_used_to_solve_this_problem(output_dir, Pn, num_rows_)
                if storage is None
                else [storage]
            )

            for storage_ in storage_used:

                results[Pn][num_rows_][storage_] = dict()

                # Find the storage formats that have been used on this problem
                threads_used = (
                    get_threads_used_to_solve_this_problem(
                        output_dir, Pn, num_rows_, storage_
                    )
                    if threads is None
                    else [str(threads)]
                )

                for threads_ in threads_used:

                    results[Pn][num_rows_][storage_][threads_] = dict()
                    for Ln in libraries:
                        try:
                            res = load_benchmark_results(
                                output_dir, Pn, Ln, num_rows_, storage_, threads_
                            )
                        except FileNotFoundError:
                            continue
                        if len(res) > 0:
                            results[Pn][num_rows_][storage_][threads_][Ln] = res

    formatted_results = (
        extract_dict_results_to_pd_series(
            prob_name, lib_name, num_rows, storage, threads, res
        )
        for prob_name in results.keys()
        for num_rows in results[prob_name].keys()
        for storage in results[prob_name][num_rows].keys()
        for threads in results[prob_name][num_rows][storage].keys()
        for lib_name, res in results[prob_name][num_rows][storage][threads].items()
    )
    res_df = (
        pd.concat(formatted_results, axis=1)
        .T.set_index(["problem", "num_rows", "storage", "threads", "library"])
        .sort_index()
    )

    res_df["n_iter"] = res_df["n_iter"].astype(int)
    for col in ["runtime", "runtime per iter", "intercept", "l1", "l2"]:
        res_df[col] = res_df[col].astype(float)

    for col in ["obj_val"]:
        res_df["rel_" + col] = (
            res_df[col]
            - res_df.groupby(["problem", "num_rows", "storage", "threads"])[col].min()
        )

    problems = res_df.index.get_level_values("problem").values
    # keeps = ["sparse" not in x and "no_weights" in x for x in problems]
    keeps = [x in x for x in problems]
    # res_df.loc[keeps, :].reset_index().to_csv("results.csv")
    with pd.option_context("display.expand_frame_repr", False, "max_columns", 10):
        print(
            res_df.loc[
                keeps, ["n_iter", "runtime", "intercept", "obj_val", "rel_obj_val"],
            ]
        )


def extract_dict_results_to_pd_series(
    prob_name: str,
    lib_name: str,
    num_rows: str,
    storage: str,
    threads: str,
    results: Dict[str, Any],
) -> pd.Series:
    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs ** 2)

    problem = get_all_problems()[prob_name]
    dat = problem.data_loader(None if num_rows == "None" else int(num_rows))
    try:
        obj_val = get_obj_val(
            dat,
            problem.distribution,
            problem.regularization_strength,
            problem.l1_ratio,
            results["intercept"],
            coefs,
        )

    except NotImplementedError:
        obj_val = 0
        print(
            "skipping objective calculation because this distribution is not implemented"
        )

    formatted = {
        "problem": prob_name,
        "library": lib_name,
        "threads": "None" if threads == "None" else int(threads),
        "storage": storage,
        "num_rows": dat["y"].shape[0] if num_rows == "None" else int(num_rows),
        "n_iter": results["n_iter"],
        "runtime": results["runtime"],
        "runtime per iter": runtime_per_iter,
        "intercept": results["intercept"],
        "l1": l1_norm,
        "l2": l2_norm,
        "obj_val": obj_val,
    }
    return pd.Series(formatted)


def get_num_rows_used_to_solve_this_problem(
    output_dir: str, prob_name: str
) -> List[str]:
    prob_dir = os.path.join(output_dir, prob_name)
    num_rows_used = os.listdir(prob_dir)
    if not all(os.path.isdir(os.path.join(prob_dir, x)) for x in num_rows_used):
        raise RuntimeError(
            f"""
            Everything in {prob_dir} should be a directory, but this is not the
            case. This likely happened because you have benchmarks generated
            under an older storage scheme. Please delete them.
            """
        )
    return num_rows_used


def get_storage_used_to_solve_this_problem(
    output_dir: str, prob_name: str, num_rows: str
) -> List[str]:
    prob_dir = os.path.join(output_dir, prob_name, num_rows)
    storage_used = os.listdir(prob_dir)
    if not all(os.path.isdir(os.path.join(prob_dir, x)) for x in storage_used):
        raise RuntimeError(
            f"""
            Everything in {prob_dir} should be a directory, but this is not the
            case. This likely happened because you have benchmarks generated
            under an older storage scheme. Please delete them.
            """
        )
    return storage_used


def get_threads_used_to_solve_this_problem(
    output_dir: str, prob_name: str, num_rows: str, storage: str
) -> List[str]:
    prob_dir = os.path.join(output_dir, prob_name, num_rows, storage)
    threads_used = os.listdir(prob_dir)
    if not all(os.path.isdir(os.path.join(prob_dir, x)) for x in threads_used):
        raise RuntimeError(
            f"""
            Everything in {prob_dir} should be a directory, but this is not the
            case. This likely happened because you have benchmarks generated
            under an older storage scheme. Please delete them.
            """
        )
    return threads_used


def get_limited_problems_libraries(
    problem_names: str, library_names: str
) -> Tuple[Dict, Dict]:
    all_libraries = dict(
        sklearn_fork=sklearn_fork_bench, glmnet_qc=glmnet_qc_bench, zeros=zeros_bench,
    )

    if GLMNET_PYTHON_INSTALLED:
        all_libraries["glmnet_python"] = glmnet_python_bench

    if H20_INSTALLED:
        all_libraries["h2o"] = h2o_bench

    if len(library_names) > 0:
        library_names_split = get_comma_sep_names(library_names)
        libraries = {k: all_libraries[k] for k in library_names_split}
    else:
        libraries = all_libraries
    return get_limited_problems(problem_names), libraries


def get_limited_problems(problem_names):
    all_problems = get_all_problems()

    if len(problem_names) > 0:
        problem_names_split = get_comma_sep_names(problem_names)
        problems = {k: all_problems[k] for k in problem_names_split}
    else:
        problems = all_problems
    return problems


def get_comma_sep_names(xs: str) -> List[str]:
    return [x.strip() for x in xs.split(",")]


def save_benchmark_results(
    output_dir: str,
    problem_name: str,
    library_name: str,
    num_rows: int,
    storage: str,
    threads: int,
    result,
) -> None:
    problem_dir = os.path.join(output_dir, problem_name)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    results_dir = os.path.join(problem_dir, str(num_rows), storage, str(threads))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, library_name + "-results.pkl"), "wb") as f:
        pickle.dump(result, f)


def load_benchmark_results(
    output_dir: str,
    problem_name: str,
    library_name: str,
    num_rows: str,
    storage: str,
    threads: str,
):
    results_dir = os.path.join(output_dir, problem_name, num_rows, storage, threads)
    with open(os.path.join(results_dir, library_name + "-results.pkl"), "rb") as f:
        return pickle.load(f)
