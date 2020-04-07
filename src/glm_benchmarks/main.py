import os
import pickle
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import pandas as pd

from glm_benchmarks.bench_glmnet_python import glmnet_python_bench
from glm_benchmarks.bench_h2o import h2o_bench
from glm_benchmarks.bench_qc_glmnet import glmnet_qc_bench
from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.bench_tensorflow import tensorflow_bench
from glm_benchmarks.problems import get_all_problems

from .bench_pyglmnet import pyglmnet_bench
from .util import get_obj_val
from .zeros_benchmark import zeros_bench


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
def cli_run(problem_names: str, library_names: str, num_rows: int, output_dir: str):
    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    for Pn, P in problems.items():
        for Ln, L in libraries.items():
            print(f"running problem={Pn} library={Ln}")
            dat = P.data_loader(num_rows=num_rows)
            result = L(dat, P.distribution, P.regularization_strength, P.l1_ratio)
            save_benchmark_results(output_dir, Pn, Ln, num_rows, result)
            print("ran")


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
    "--output_dir",
    default="benchmark_output",
    help="The directory where we load benchmarking output.",
)
def cli_analyze(problem_names: str, library_names: str, num_rows: str, output_dir: str):
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)

    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    results: Dict[str, Dict[str, Dict[str, Any]]] = dict()
    for Pn in problems:
        results[Pn] = dict()

        # Find the row counts that have been used on this problem
        n_rows_used = (
            get_n_rows_used_to_solve_this_problem(output_dir, Pn)
            if num_rows is None
            else [str(num_rows)]
        )

        for n_rows in n_rows_used:
            results[Pn][n_rows] = dict()
            for Ln in libraries:
                try:
                    res = load_benchmark_results(output_dir, Pn, Ln, n_rows)
                except FileNotFoundError:
                    continue
                if len(res) > 0:
                    results[Pn][n_rows][Ln] = res

    formatted_results = (
        extract_dict_results_to_pd_series(prob_name, lib_name, n_rows, res)
        for prob_name in results.keys()
        for n_rows in results[prob_name].keys()
        for lib_name, res in results[prob_name][n_rows].items()
    )
    res_df = (
        pd.concat(formatted_results, axis=1)
        .T.set_index(["problem", "n_rows", "library"])
        .sort_index()
    )

    res_df["n_iter"] = res_df["n_iter"].astype(int)
    for col in ["runtime", "runtime per iter", "intercept", "l1", "l2"]:
        res_df[col] = res_df[col].astype(float)

    for col in ["obj_val", "obj_val_2"]:
        res_df["rel_" + col] = res_df[col] - res_df.groupby(level=[0, 1])[col].min()

    problems = res_df.index.get_level_values("problem").values
    # keeps = ["sparse" not in x and "no_weights" in x for x in problems]
    keeps = [x in x for x in problems]
    res_df.loc[keeps, :].reset_index().to_csv("results.csv")
    print(
        res_df.loc[
            keeps,
            [
                "n_iter",
                "runtime",
                "intercept",
                "obj_val",
                "rel_obj_val",
                "rel_obj_val_2",
            ],
        ]
    )


def extract_dict_results_to_pd_series(
    prob_name: str, lib_name: str, n_rows: str, results: Dict[str, Any]
) -> pd.Series:
    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs ** 2)

    problem = get_all_problems()[prob_name]
    dat = problem.data_loader(None if n_rows == "None" else int(n_rows))
    obj_val = get_obj_val(
        dat,
        problem.distribution,
        problem.regularization_strength,
        problem.l1_ratio,
        results["intercept"],
        coefs,
    )
    obj_2 = get_obj_val(
        dat,
        problem.distribution,
        problem.regularization_strength,
        problem.l1_ratio,
        results["intercept"],
        coefs,
        True,
    ) * len(dat["y"])

    formatted = {
        "problem": prob_name,
        "library": lib_name,
        "n_rows": dat["y"].shape[0] if n_rows == "None" else int(n_rows),
        "n_iter": results["n_iter"],
        "runtime": results["runtime"],
        "runtime per iter": runtime_per_iter,
        "intercept": results["intercept"],
        "l1": l1_norm,
        "l2": l2_norm,
        "obj_val": obj_val,
        "obj_val_2": obj_2,
    }
    return pd.Series(formatted)


def get_n_rows_used_to_solve_this_problem(output_dir: str, prob_name: str) -> List[str]:
    prob_dir = os.path.join(output_dir, prob_name)
    n_rows_used = os.listdir(prob_dir)
    if not all(os.path.isdir(os.path.join(prob_dir, x)) for x in n_rows_used):
        raise RuntimeError(
            f"""
            Everything in {prob_dir} should be a directory, but this is not the
            case. This likely happened because you have benchmarks generated
            under an older storage scheme. Please delete them.
            """
        )
    return n_rows_used


def get_limited_problems_libraries(
    problem_names: str, library_names: str
) -> Tuple[Dict, Dict]:
    all_problems = get_all_problems()
    all_libraries = dict(
        sklearn_fork=sklearn_fork_bench,
        glmnet_python=glmnet_python_bench,
        tensorflow=tensorflow_bench,
        h2o=h2o_bench,
        glmnet_qc=glmnet_qc_bench,
        zeros=zeros_bench,
        pyglmnet=pyglmnet_bench,
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


def save_benchmark_results(
    output_dir: str, problem_name: str, library_name: str, n_rows: int, result
) -> None:
    problem_dir = os.path.join(output_dir, problem_name)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    problem_nrow_dir = os.path.join(problem_dir, str(n_rows))
    if not os.path.exists(problem_nrow_dir):
        os.makedirs(problem_nrow_dir)
    with open(os.path.join(problem_nrow_dir, library_name + "-results.pkl"), "wb") as f:
        pickle.dump(result, f)


def load_benchmark_results(
    output_dir: str, problem_name: str, library_name: str, n_rows: str
):
    problem_nrow_dir = os.path.join(output_dir, problem_name, n_rows)
    with open(os.path.join(problem_nrow_dir, library_name + "-results.pkl"), "rb") as f:
        return pickle.load(f)
