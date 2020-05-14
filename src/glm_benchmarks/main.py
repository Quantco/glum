import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import scipy.sparse

from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.problems import get_all_problems
from glm_benchmarks.util import BenchmarkParams, get_obj_val
from glm_benchmarks.zeros_benchmark import zeros_bench

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

benchmark_params = [
    "problem-name",
    "library-name",
    "num-rows",
    "storage",
    "threads",
    "single-precision",
    "regularization-strength",
    "cv",
]


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
@click.option("--cv", is_flag=True, help="Cross-validation")
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory to store benchmarking output.",
)
@click.option("--single_precision", is_flag=True, help="Whether to use 32-bit data")
@click.option(
    "--iterations",
    default=1,
    type=int,
    help="Number of times to re-run the benchmark. This can be useful for avoid performance noise.",
)
@click.option(
    "--regularization_strength",
    default=None,
    type=float,
    help="Regularization strength. Set to None to use the default value of the problem.",
)
# TODO: where it calls data loader in main.py, convert x to the correct dtype
def cli_run(
    problem_names: str,
    library_names: str,
    num_rows: int,
    storage: str,
    threads: int,
    cv: bool,
    output_dir: str,
    single_precision: bool,
    iterations: int,
    regularization_strength: Optional[float],
):
    problems, libraries = get_limited_problems_libraries(problem_names, library_names)

    params = BenchmarkParams(
        problem_names,
        library_names,
        num_rows,
        storage,
        threads,
        single_precision,
        regularization_strength,
        cv,
    )

    for Pn, P in problems.items():
        for Ln, L in libraries.items():
            print(f"running problem={Pn} library={Ln}")
            result, regularization_strength_ = execute_problem_library(
                P, L, params, iterations,
            )
            save_benchmark_results(
                output_dir,
                params.update_params(problem_name=Pn, library_name=Ln),
                result,
            )
            print(f"ran problem {Pn} with libray {Ln}")
            print(f"ran in {result['runtime']}")


def execute_problem_library(
    P,
    L: Callable,
    params: BenchmarkParams,
    iterations: int = 1,
    print_diagnostics: bool = False,
    **kwargs,
):
    dat = P.data_loader(num_rows=params.num_rows)
    if params.threads is None:
        threads = os.environ.get("OMP_NUM_THREADS", os.cpu_count())
    else:
        threads = params.threads

    os.environ["OMP_NUM_THREADS"] = str(threads)
    if params.single_precision:
        for k, v in dat.items():
            dat[k] = v.astype(np.float32)

    if params.storage == "sparse":
        dat["X"] = scipy.sparse.csc_matrix(dat["X"])
    elif params.storage.startswith("split"):
        from glm_benchmarks.scaled_spmat.split_matrix import SplitMatrix

        threshold = float(params.storage.split("split")[1])
        dat["X"] = SplitMatrix(scipy.sparse.csc_matrix(dat["X"]), threshold)

    if params.regularization_strength is None:
        regularization_strength = P.regularization_strength
    else:
        regularization_strength = params.regularization_strength

    result = L(
        dat,
        P.distribution,
        regularization_strength,
        P.l1_ratio,
        iterations,
        params.cv,
        print_diagnostics,
        **kwargs,
    )
    return result, regularization_strength


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
    type=int,
    help="The number of rows that the GLM models were run with.",
)
@click.option(
    "--storage",
    type=str,
    default="dense",
    help="Can be dense or sparse. Leave blank to analyze all.",
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
@click.option("--cv", is_flag=True, help="cross-validation")
@click.option(
    "--regularization_strength",
    type=float,
    default=None,
    help="Specify regularization strength. Leave blank to analyze all.",
)
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory where we load benchmarking output.",
)
def cli_analyze(
    problem_names: str,
    library_names: str,
    num_rows: int,
    storage: str,
    threads: int,
    single_precision: bool,
    cv: bool,
    regularization_strength: Optional[float],
    output_dir: str,
):
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)
    params = BenchmarkParams(
        problem_names,
        library_names,
        num_rows,
        storage,
        threads,
        single_precision,
        regularization_strength,
        cv,
    )

    file_names = identify_parameter_directories(output_dir, params)

    raw_results = {
        fname: load_benchmark_results(output_dir, fname) for fname in file_names
    }
    formatted_results = [
        extract_dict_results_to_pd_series(name, res)
        for name, res in raw_results.items()
        if len(res) > 0
    ]

    res_df = pd.concat(formatted_results, axis=1).T
    res_df["offset"] = res_df["problem-name"].apply(lambda x: "offset" in x)
    res_df["problem-name"] = [
        "weights".join(x.split("offset")) for x in res_df["problem-name"]
    ]
    res_df = res_df.set_index(["problem-name", "num-rows", "library-name"]).sort_index()
    res_df["n_iter"] = res_df["n_iter"].astype(int)
    for col in ["runtime", "runtime per iter", "intercept", "l1", "l2"]:
        res_df[col] = res_df[col].astype(float)

    for col in ["obj_val"]:
        res_df["rel_" + col] = (
            res_df[col]
            - res_df.groupby(["problem-name", "num-rows", "regularization_strength"])[
                col
            ].min()
        )

    with pd.option_context(
        "display.expand_frame_repr", False, "max_columns", None, "max_rows", None
    ):
        cols_to_show = [
            "storage",
            "threads",
            "single_precision",
            "n_iter",
            "runtime",
            "offset",
        ]
        if res_df["cv"].any():
            cols_to_show += ["n_alphas", "max_alpha", "min_alpha", "best_alpha"]
        else:
            cols_to_show += ["intercept", "obj_val", "rel_obj_val"]

        print(res_df[cols_to_show])


def extract_dict_results_to_pd_series(
    fname: str, results: Dict[str, Any],
) -> pd.Series:
    assert "coef" in results.keys()
    (
        prob_name,
        lib_name,
        num_rows,
        storage,
        threads,
        single_precision,
        cv,
        regularization_strength,
    ) = fname.split("_")
    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs ** 2)

    # weights and offsets are solving the same problem, but the objective is set up to
    # deal with weights, so load the data for the weights problem rather than the
    # offset problem
    prob_name_weights = "weights".join(prob_name.split("offset"))
    problem = get_all_problems()[prob_name_weights]
    possible_distributions = ["gaussian", "poisson", "tweedie", "gamma"]

    distribution = None
    for elt in possible_distributions:
        if elt in prob_name:
            distribution = elt
    assert distribution is not None
    dat = problem.data_loader(
        None if num_rows == "None" else int(num_rows), None, distribution
    )
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

    formatted: Dict[str, Any] = dict(
        zip(
            benchmark_params,
            [prob_name, lib_name, num_rows, storage, threads, single_precision, cv],
        )
    )
    items_to_use_from_results = ["n_iter", "runtime", "intercept"]
    if cv:
        items_to_use_from_results += [
            "n_alphas",
            "max_alpha",
            "min_alpha",
            "best_alpha",
        ]
    formatted.update(
        {k: v for k, v in results.items() if k in items_to_use_from_results}
    )

    formatted.update(
        {
            "threads": "None" if threads == "None" else int(threads),
            "num_rows": dat["y"].shape[0] if num_rows == "None" else int(num_rows),
            "runtime per iter": runtime_per_iter,
            "l1": l1_norm,
            "l2": l2_norm,
            "obj_val": obj_val,
            "offset": "offset" in prob_name,
            "regularization_strength": float(regularization_strength),
        }
    )

    return pd.Series(formatted)


def identify_parameter_directories(
    root_dir: str, constraint_params: BenchmarkParams
) -> List[str]:
    def _to_dict(name):
        return dict(zip(benchmark_params, name.split("_")))

    results_to_keep = []
    for result_name in os.listdir(root_dir):
        this_file_params = _to_dict(result_name.strip(".pkl"))
        keep_this_problem = {
            k: getattr(constraint_params, k) is None
            or getattr(constraint_params, k) == ""
            # e.g. this_file_params['library_name'] is 'sklearn-fork'
            # and constraint_params.library_name is 'sklearn-fork,h2o'
            or this_file_params[k] in str(getattr(constraint_params, k))
            for k in constraint_params.param_names
        }
        if all(keep_this_problem.values()):
            results_to_keep.append(result_name)
    return results_to_keep


def get_limited_problems_libraries(
    problem_names: str, library_names: str
) -> Tuple[Dict, Dict]:
    all_libraries = {"sklearn-fork": sklearn_fork_bench, "zeros": zeros_bench}

    if GLMNET_PYTHON_INSTALLED:
        all_libraries["glmnet-python"] = glmnet_python_bench

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


def save_benchmark_results(output_dir: str, params: BenchmarkParams, result,) -> None:

    results_path = output_dir + "/" + params.get_result_fname()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(results_path + ".pkl", "wb") as f:
        pickle.dump(result, f)


def load_benchmark_results(output_dir: str, fname: str):
    results_path = os.path.join(output_dir, fname)
    with open(results_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    cli_run()
