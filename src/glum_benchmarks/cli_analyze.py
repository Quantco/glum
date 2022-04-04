import os
import pickle
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd

from glum_benchmarks.problems import get_all_problems
from glum_benchmarks.util import (
    BenchmarkParams,
    benchmark_params_cli,
    clear_cache,
    get_params_from_fname,
)


def _get_comma_sep_names(xs: str) -> List[str]:
    return [x.strip() for x in xs.split(",")]


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
@click.option(
    "--cols", default=None, type=str, help="Which output analysis columns to display?"
)
@benchmark_params_cli
def cli_analyze(
    params: BenchmarkParams, output_dir: str, export: Optional[str], cols: str
):
    """
    Describe runtime, objective function values, and other statistics on the \
    already-run problems specified by the command line options.

    Parameters
    ----------
    params: BenchmarkParams
    output_dir: str
    export: string or None
    cols: str
    """
    clear_cache()
    display_precision = 4
    np.set_printoptions(precision=display_precision, suppress=True)
    pd.set_option("precision", display_precision)

    file_names = _identify_parameter_fnames(output_dir, params)

    raw_results = {
        fname: _load_benchmark_results(output_dir, fname) for fname in file_names
    }
    formatted_results = [
        _extract_dict_results_to_pd_series(name, res)
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
    problem_id_cols = ["problem_name", "num_rows", "regularization_strength", "offset"]
    res_df = res_df.set_index(problem_id_cols).sort_values("library_name").sort_index()
    if params.cv:
        for col in ["max_alpha", "min_alpha"]:
            res_df[col] = res_df[col].astype(float)

    res_df["rel_obj_val"] = (
        res_df[["obj_val"]] - res_df.groupby(level=[0, 1, 2, 3])[["obj_val"]].min()
    )

    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):
        if cols is not None:
            cols_to_show = _get_comma_sep_names(cols)
        else:
            cols_to_show = [
                "library_name",
                "storage",
                "threads",
                "single_precision",
                "n_iter",
                "runtime",
            ]
            if res_df["cv"].any():
                cols_to_show += ["n_alphas", "max_alpha", "min_alpha", "best_alpha"]
            else:
                cols_to_show += [
                    "intercept",
                    "num_nonzero_coef",
                    "obj_val",
                    "rel_obj_val",
                ]
        if "library_name" not in cols_to_show:
            cols_to_show.insert(0, "library_name")
        print(res_df[cols_to_show])

    if export:
        if export.endswith(".pkl"):
            res_df.to_pickle(export)
        else:
            res_df.to_csv(export)

    return res_df


def _extract_dict_results_to_pd_series(
    fname: str,
    results: Dict[str, Any],
) -> Dict:
    assert "coef" in results.keys()
    params = get_params_from_fname(fname)
    assert params.problem_name is not None

    coefs = results["coef"]
    runtime_per_iter = results["runtime"] / results["n_iter"]
    l1_norm = np.sum(np.abs(coefs))
    l2_norm = np.sum(coefs**2)
    num_nonzero_coef = np.sum(np.abs(coefs) > 1e-8)

    # weights and offsets are solving the same problem, but the objective is set up to
    # deal with weights, so load the data for the weights problem rather than the
    # offset problem
    if "housing" not in params.problem_name:
        prob_name_weights = "weights".join(params.problem_name.split("offset"))
    else:
        prob_name_weights = params.problem_name
    problem = get_all_problems()[prob_name_weights]

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
            "num_rows": results["num_rows"],
            "regularization_strength": (
                problem.regularization_strength
                if params.regularization_strength is None
                else params.regularization_strength
            ),
            "runtime per iter": runtime_per_iter,
            "l1": l1_norm,
            "l2": l2_norm,
            "num_nonzero_coef": num_nonzero_coef,
            "obj_val": results["obj_val"],
            "offset": "offset" in params.problem_name,
        }
    )

    return formatted


def _identify_parameter_fnames(
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


def _load_benchmark_results(output_dir: str, fname: str):
    results_path = os.path.join(output_dir, fname)
    with open(results_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    cli_analyze()
