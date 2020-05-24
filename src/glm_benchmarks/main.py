import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

from glm_benchmarks.bench_admm import admm_bench
from glm_benchmarks.bench_orig_sklearn_fork import orig_sklearn_fork_bench
from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.problems import Problem, get_all_problems
from glm_benchmarks.util import (
    BenchmarkParams,
    benchmark_params_cli,
    clear_cache,
    get_obj_val,
    get_params_from_fname,
)
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


@click.command()
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory to store benchmarking output.",
)
@click.option(
    "--iterations",
    default=1,
    type=int,
    help="Number of times to re-run the benchmark. This can be useful for avoid performance noise.",
)
@benchmark_params_cli
def cli_run(
    params: BenchmarkParams, output_dir: str, iterations: int,
):
    clear_cache()
    problems, libraries = get_limited_problems_libraries(
        params.problem_name, params.library_name
    )

    for Pn, P in problems.items():
        for Ln, L in libraries.items():
            click.echo(f"running problem={Pn} library={Ln}")
            new_params = params.update_params(problem_name=Pn, library_name=Ln)
            result, regularization_strength_ = execute_problem_library(
                new_params, iterations
            )
            save_benchmark_results(
                output_dir, new_params, result,
            )
            if len(result) > 0:
                click.echo(f"ran problem {Pn} with library {Ln}")
                click.echo(f"ran in {result.get('runtime')}")


def execute_problem_library(
    params: BenchmarkParams,
    iterations: int = 1,
    print_diagnostics: bool = True,
    **kwargs,
):
    assert params.problem_name is not None
    assert params.library_name is not None
    P = get_all_problems()[params.problem_name]
    L = get_all_libraries()[params.library_name]

    for k in params.param_names:
        if getattr(params, k) is None:
            params.update_params(**{k: get_default_val(k)})

    dat = P.data_loader(
        num_rows=params.num_rows,
        storage=params.storage,
        single_precision=params.single_precision,
    )
    os.environ["OMP_NUM_THREADS"] = str(params.threads)

    if params.regularization_strength is None:
        params.regularization_strength = P.regularization_strength

    result = L(
        dat,
        P.distribution,
        params.regularization_strength,
        P.l1_ratio,
        iterations,
        params.cv,
        print_diagnostics,
        **kwargs,
    )
    return result, params.regularization_strength


@click.command()
@click.option(
    "--output_dir",
    default="benchmark_output",
    help="The directory where we load benchmarking output.",
)
@click.option(
    "--export",
    default=None,
    type=str,
    help="File name or path to export the results to CSV or Pickle.",
)
@benchmark_params_cli
def cli_analyze(params: BenchmarkParams, output_dir: str, export: Optional[str]):

    clear_cache()
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)

    file_names = identify_parameter_fnames(output_dir, params)

    raw_results = {
        fname: load_benchmark_results(output_dir, fname) for fname in file_names
    }
    formatted_results = [
        extract_dict_results_to_pd_series(name, res)
        for name, res in raw_results.items()
        if len(res) > 0
    ]

    if not formatted_results:
        return

    res_df = pd.DataFrame.from_records(formatted_results)
    res_df["offset"] = res_df["problem_name"].apply(lambda x: "offset" in x)
    res_df["problem_name"] = [
        "weights".join(x.split("offset")) for x in res_df["problem_name"]
    ]
    problem_id_cols = ["problem_name", "num_rows", "regularization_strength"]
    res_df = res_df.set_index(problem_id_cols).sort_values("library_name").sort_index()
    if params.cv:
        for col in ["max_alpha", "min_alpha"]:
            res_df[col] = res_df[col].astype(float)

    res_df["rel_obj_val"] = (
        res_df[["obj_val"]] - res_df.groupby(level=[0, 1, 2])[["obj_val"]].min()
    )

    with pd.option_context(
        "display.expand_frame_repr", False, "max_columns", None, "max_rows", None
    ):
        cols_to_show = [
            "library_name",
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
            cols_to_show += ["intercept", "num_nonzero_coef", "obj_val", "rel_obj_val"]

        print(res_df[cols_to_show])

    if export:
        if export.endswith(".pkl"):
            res_df.to_pickle(export)
        else:
            res_df.to_csv(export)

    return res_df


def extract_dict_results_to_pd_series(fname: str, results: Dict[str, Any],) -> Dict:
    assert "coef" in results.keys()
    params = get_params_from_fname(fname)
    assert params.problem_name is not None

    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs ** 2)
    num_nonzero_coef = np.sum(np.abs(coefs) > 1e-8)

    # weights and offsets are solving the same problem, but the objective is set up to
    # deal with weights, so load the data for the weights problem rather than the
    # offset problem
    prob_name_weights = "weights".join(params.problem_name.split("offset"))
    problem = get_all_problems()[prob_name_weights]

    dat = problem.data_loader(params.num_rows)
    tweedie = "tweedie" in params.problem_name
    if tweedie:
        tweedie_p = float(params.problem_name.split("=")[-1])

    obj_val = get_obj_val(
        dat,
        problem.distribution,
        problem.regularization_strength,
        problem.l1_ratio,
        results["intercept"],
        coefs,
        tweedie_p=tweedie_p if tweedie else None,
    )

    formatted: Dict[str, Any] = params.__dict__
    items_to_use_from_results = ["n_iter", "runtime", "intercept"]
    if params.cv:
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
            "num_rows": dat["y"].shape[0]
            if params.num_rows is None
            else params.num_rows,
            "regularization_strength": (
                problem.regularization_strength
                if params.regularization_strength is None
                else params.regularization_strength
            ),
            "runtime per iter": runtime_per_iter,
            "l1": l1_norm,
            "l2": l2_norm,
            "num_nonzero_coef": num_nonzero_coef,
            "obj_val": obj_val,
            "offset": "offset" in params.problem_name,
        }
    )

    return formatted


def identify_parameter_fnames(
    root_dir: str, constraint_params: BenchmarkParams
) -> List[str]:
    def _satisfies_constraint(params: BenchmarkParams, k: str) -> bool:
        constraint = getattr(constraint_params, k)
        param = getattr(params, k)
        return (
            constraint is None
            or param == constraint
            # e.g. this_file_params['library_name'] is 'sklearn-fork'
            # and constraint_params.library_name is 'sklearn-fork,h2o'
            or (isinstance(constraint, str) and param in constraint.split(","))
        )

    results_to_keep = []
    for fname in os.listdir(root_dir):
        this_file_params = get_params_from_fname(fname)

        keep_this_problem = {
            k: _satisfies_constraint(this_file_params, k)
            for k in constraint_params.param_names
        }

        if all(keep_this_problem.values()):
            results_to_keep.append(fname)
    return results_to_keep


def get_all_libraries() -> Dict[str, Any]:
    all_libraries = {
        "sklearn-fork": sklearn_fork_bench,
        "orig-sklearn-fork": orig_sklearn_fork_bench,
        "zeros": zeros_bench,
        "admm": admm_bench,
    }

    if GLMNET_PYTHON_INSTALLED:
        all_libraries["glmnet-python"] = glmnet_python_bench

    if H20_INSTALLED:
        all_libraries["h2o"] = h2o_bench
    return all_libraries


def get_limited_problems_libraries(
    problem_names: Optional[str], library_names: Optional[str]
) -> Tuple[Dict, Dict]:
    all_libraries = get_all_libraries()

    if library_names is not None:
        library_names_split = get_comma_sep_names(library_names)
        libraries = {k: all_libraries[k] for k in library_names_split}
    else:
        libraries = all_libraries
    return get_limited_problems(problem_names), libraries


def get_limited_problems(problem_names: Optional[str]) -> Dict[str, Problem]:
    all_problems = get_all_problems()

    if problem_names is not None:
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


def get_default_val(k: str) -> Any:
    """

    Parameters
    ----------
    k: An element of BenchmarkParams.param_names

    Returns
    -------
        Default value of parameter.
    """
    if k == "threads":
        return os.environ.get("OMP_NUM_THREADS", os.cpu_count())
    # For these parameters, value is fixed downstream,
    # e.g. threads depends on hardware in cli_run and is 'all' for cli_analyze
    if k in ["problem_name", "library_name", "num_rows", "regularization_strength"]:
        return None
    if k == "storage":
        return "dense"
    if k == "cv":
        return False
    if k == "single_precision":
        return False
    raise KeyError(f"Key {k} not found")


if __name__ == "__main__":
    cli_run()
