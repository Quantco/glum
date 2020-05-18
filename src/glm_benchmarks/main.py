import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

from glm_benchmarks.bench_admm import admm_bench
from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.problems import get_all_problems
from glm_benchmarks.util import (
    BenchmarkParams,
    benchmark_params_cli,
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
# TODO: where it calls data loader in main.py, convert x to the correct dtype
def cli_run(
    params: BenchmarkParams, output_dir: str, iterations: int,
):
    problems, libraries = get_limited_problems_libraries(
        params.problem_name, params.library_name
    )

    for Pn, P in problems.items():
        dat = P.data_loader(
            num_rows=params.num_rows,
            storage=params.storage,
            single_precision=params.single_precision,
        )

        for Ln, L in libraries.items():
            print(f"running problem={Pn} library={Ln}")
            new_params = params.update_params(problem_name=Pn, library_name=Ln)
            result, regularization_strength_ = execute_problem_library(
                new_params, iterations, dat=dat
            )
            save_benchmark_results(
                output_dir, new_params, result,
            )
            if len(result) > 0:
                print(f"ran problem {Pn} with libray {Ln}")
                print(f"ran in {result['runtime']}")
        del dat


def execute_problem_library(
    params: BenchmarkParams,
    iterations: int = 1,
    print_diagnostics: bool = True,
    dat: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    P = get_all_problems()[params.problem_name]
    L = get_all_libraries()[params.library_name]

    if dat is None:
        dat = P.data_loader(
            num_rows=params.num_rows,
            storage=params.storage,
            single_precision=params.single_precision,
        )

    if params.threads is None:
        threads = os.environ.get("OMP_NUM_THREADS", os.cpu_count())
    else:
        threads = params.threads

    os.environ["OMP_NUM_THREADS"] = str(threads)

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
@benchmark_params_cli
def cli_analyze(
    params: BenchmarkParams, output_dir: str,
):
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)

    file_names = identify_parameter_fnames(output_dir, params)

    raw_results = sorted(
        filter(
            lambda x: len(x["res"]) > 0,
            [
                {
                    "fname": fname,
                    "res": load_benchmark_results(output_dir, fname),
                    "problem_name": get_params_from_fname(fname).problem_name,
                }
                for fname in file_names
            ],
        ),
        key=lambda x: x["problem_name"],
    )

    formatted_results = []
    current_problem = ""
    dat: Dict[str, Any] = {}
    for elt in raw_results:
        problem_name = elt["problem_name"]
        if elt["problem_name"] != current_problem:
            current_problem = elt["problem_name"]
            del dat
            dat = get_all_problems()[problem_name].data_loader(
                num_rows=params.num_rows,
                storage=params.storage,
                single_precision=params.single_precision,
            )

        formatted_results.append(
            extract_dict_results_to_pd_series(elt["fname"], elt["res"], dat)
        )

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
            cols_to_show += ["intercept", "obj_val", "rel_obj_val"]

        print(res_df[cols_to_show])


def extract_dict_results_to_pd_series(
    fname: str, results: Dict[str, Any], dat: Dict[str, Any] = None
) -> Dict:
    assert "coef" in results.keys()
    params = get_params_from_fname(fname)
    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs ** 2)

    # weights and offsets are solving the same problem, but the objective is set up to
    # deal with weights, so load the data for the weights problem rather than the
    # offset problem
    prob_name_weights = "weights".join(params.problem_name.split("offset"))
    problem = get_all_problems()[prob_name_weights]

    if dat is None:
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
                else problem.regularization_strength
            ),
            "runtime per iter": runtime_per_iter,
            "l1": l1_norm,
            "l2": l2_norm,
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
            # TODO: no more ""
            constraint is None
            or constraint == ""
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
        "zeros": zeros_bench,
        "admm": admm_bench,
    }

    if GLMNET_PYTHON_INSTALLED:
        all_libraries["glmnet-python"] = glmnet_python_bench

    if H20_INSTALLED:
        all_libraries["h2o"] = h2o_bench
    return all_libraries


def get_limited_problems_libraries(
    problem_names: str, library_names: str
) -> Tuple[Dict, Dict]:
    all_libraries = get_all_libraries()

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
