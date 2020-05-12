import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import scipy.sparse

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
@click.option("--single_precision", is_flag=True, help="Whether to use 32-bit data")
# TODO: where it calls data loader in main.py, convert x to the correct dtype
def cli_run(
    problem_names: str,
    library_names: str,
    num_rows: int,
    storage: str,
    threads: int,
    output_dir: str,
    single_precision: bool,
):
    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    for Pn, P in problems.items():
        for Ln, L in libraries.items():
            print(f"running problem={Pn} library={Ln}")
            result = execute_problem_library(
                P, L, num_rows, storage, threads, single_precision
            )
            save_benchmark_results(
                output_dir, Pn, Ln, num_rows, storage, threads, single_precision, result
            )
            print("ran")


def execute_problem_library(
    P,
    L,
    num_rows=None,
    storage="dense",
    threads=None,
    single_precision: bool = False,
    print_diagnostics: bool = True,
    **kwargs,
):
    dat = P.data_loader(num_rows=num_rows)
    if threads is None:
        threads = os.environ.get("OMP_NUM_THREADS", os.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(threads)
    if storage == "sparse":
        dat["X"] = scipy.sparse.csc_matrix(dat["X"])
    if single_precision:
        for k, v in dat.items():
            dat[k] = v.astype(np.float32)
    result = L(
        dat,
        P.distribution,
        P.regularization_strength,
        P.l1_ratio,
        print_diagnostics,
        **kwargs,
    )
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
    "--single_precision",
    type=bool,
    help="please help me i have been in a hole for 273 hours",
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
    single_precision: str,
    output_dir: str,
):
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)

    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    formatted_results = []
    for Pn in problems:

        # Find the row counts that have been used on this problem
        dir_ = os.path.join(output_dir, Pn)
        param_values = identify_parameter_directories(
            dir_, [num_rows, storage, threads, single_precision]
        )

        for pv in param_values:
            num_rows_, storage_, threads_, single_precision_ = pv
            for Ln in libraries:
                try:
                    res = load_benchmark_results(
                        output_dir,
                        Pn,
                        Ln,
                        num_rows_,
                        storage_,
                        threads_,
                        single_precision_,
                    )
                except FileNotFoundError:
                    continue

                if len(res) > 0:
                    formatted_results.append(
                        extract_dict_results_to_pd_series(
                            Pn,
                            Ln,
                            num_rows_,
                            storage_,
                            threads_,
                            single_precision_,
                            res,
                        )
                    )

    res_df = pd.concat(formatted_results, axis=1).T
    res_df["offset"] = res_df["problem"].apply(lambda x: "offset" in x)
    res_df["problem"] = ["weights".join(x.split("offset")) for x in res_df["problem"]]
    res_df = res_df.set_index(["problem", "num_rows", "library"]).sort_index()
    problems = res_df.index.get_level_values("problem").values

    res_df["n_iter"] = res_df["n_iter"].astype(int)
    for col in ["runtime", "runtime per iter", "intercept", "l1", "l2"]:
        res_df[col] = res_df[col].astype(float)

    for col in ["obj_val"]:
        res_df["rel_" + col] = (
            res_df[col] - res_df.groupby(["problem", "num_rows"])[col].min()
        )

    with pd.option_context(
        "display.expand_frame_repr", False, "max_columns", 10, "max_rows", None
    ):
        cols_to_show = [
            "n_iter",
            "runtime",
            "intercept",
            "obj_val",
            "rel_obj_val",
        ]
        print(res_df[cols_to_show])


def extract_dict_results_to_pd_series(
    prob_name: str,
    lib_name: str,
    num_rows: str,
    storage: str,
    threads: str,
    single_precision: str,
    results: Dict[str, Any],
) -> pd.Series:
    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs ** 2)

    # weights and offsets are solving the same problem, but the objective is set up to
    # deal with weights, so load the data for the weights problem rather than the
    # offset problem
    prob_name_weights = "weights".join(prob_name.split("offset"))
    problem = get_all_problems()[prob_name_weights]
    dat = problem.data_loader(None if num_rows == "None" else int(num_rows))
    tweedie = "tweedie" in prob_name
    if tweedie:
        tweedie_p = float(prob_name.split("=")[-1])

    obj_val = get_obj_val(
        dat,
        problem.distribution,
        problem.regularization_strength,
        problem.l1_ratio,
        results["intercept"],
        coefs,
        tweedie_p=tweedie_p if tweedie else None,
    )

    formatted = {
        "problem": prob_name,
        "library": lib_name,
        "threads": "None" if threads == "None" else int(threads),
        "storage": storage,
        "single_precision": single_precision,
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


def identify_parameter_directories(
    root_dir: str, constraints: List[str]
) -> List[List[str]]:
    if constraints[0] is None:
        try:
            param_values = os.listdir(root_dir)
        except FileNotFoundError:
            return []
        if not all(os.path.isdir(os.path.join(root_dir, x)) for x in param_values):
            raise RuntimeError(
                f"""
                Everything in {root_dir} should be a directory, but this is not the
                case. This likely happened because you have benchmarks generated
                under an older storage scheme. Please delete them.
                """
            )
    else:
        param_values = [str(constraints[0])]
    if len(constraints) == 1:
        return [[pv] for pv in param_values]
    else:
        out = []
        for pv in param_values:
            sub_root_dir = os.path.join(root_dir, pv)
            sub_param_values = identify_parameter_directories(
                sub_root_dir, constraints[1:]
            )
            for spv in sub_param_values:
                out.append([pv] + spv)
        return out


def get_limited_problems_libraries(
    problem_names: str, library_names: str
) -> Tuple[Dict, Dict]:
    all_libraries = dict(sklearn_fork=sklearn_fork_bench, zeros=zeros_bench,)

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


def get_path(
    output_dir: str,
    problem_name: str,
    num_rows: Union[int, str] = None,
    storage: str = None,
    threads: Union[int, str] = None,
    single_precision: bool = False,
):
    return os.path.join(
        output_dir,
        problem_name,
        str(num_rows),
        str(storage),
        str(threads),
        "single" if single_precision else "double",
    )


def save_benchmark_results(
    output_dir: str,
    problem_name: str,
    library_name: str,
    num_rows: int,
    storage: str,
    threads: int,
    single_precision: bool,
    result,
) -> None:
    problem_dir = os.path.join(output_dir, problem_name)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    results_dir = get_path(
        output_dir, problem_name, num_rows, storage, threads, single_precision
    )
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
    single_precision: str,
):
    results_dir = os.path.join(
        output_dir, problem_name, num_rows, storage, threads, single_precision
    )
    with open(os.path.join(results_dir, library_name + "-results.pkl"), "rb") as f:
        return pickle.load(f)
