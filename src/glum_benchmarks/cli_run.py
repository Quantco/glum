import os
import pickle
from typing import Any, Optional

import click

from glum_benchmarks.bench_glum import glum_bench
from glum_benchmarks.bench_orig_sklearn_fork import orig_sklearn_fork_bench
from glum_benchmarks.problems import Problem, get_all_problems
from glum_benchmarks.util import (
    BenchmarkParams,
    benchmark_params_cli,
    clear_cache,
    defaults,
    get_obj_val,
)
from glum_benchmarks.zeros_benchmark import zeros_bench

try:
    from .bench_h2o import h2o_bench  # isort:skip

    H20_INSTALLED = True
except ImportError:
    H20_INSTALLED = False

try:
    from .bench_liblinear import liblinear_bench  # isort:skip

    LIBLINEAR_INSTALLED = True
except ImportError:
    LIBLINEAR_INSTALLED = False

try:
    from .bench_r_glmnet import r_glmnet_bench  # isort:skip

    R_GLMNET_INSTALLED = True
except ImportError:
    R_GLMNET_INSTALLED = False


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
    help="Number of times to re-run the benchmark. This can be useful for avoid "
    "performance noise.",
)
@benchmark_params_cli
def cli_run(
    params: BenchmarkParams,
    output_dir: str,
    iterations: int,
):
    """
    Run benchmark problems through the command line.

    See the README for more info.

    Parameters
    ----------
    params: BenchmarkParams, parsed by benchmark_params_cli decorator
    output_dir
    iterations

    Returns
    -------
    None
    """
    clear_cache()
    problems, libraries = get_limited_problems_libraries(
        params.problem_name, params.library_name
    )

    for Pn in problems.keys():
        for Ln in libraries.keys():
            click.echo(f"running problem={Pn} library={Ln}")
            new_params = params.update_params(problem_name=Pn, library_name=Ln)
            result, _ = execute_problem_library(
                new_params,
                iterations,
                (
                    defaults["diagnostics_level"]  # type: ignore
                    if params.diagnostics_level is None  # type: ignore
                    else params.diagnostics_level  # type: ignore
                ),
            )
            _save_benchmark_results(
                output_dir,
                new_params,
                result,
            )
            if len(result) > 0:
                click.echo(f"ran problem {Pn} with library {Ln}")
                click.echo(f"ran in {result.get('runtime')}")


def execute_problem_library(
    params: BenchmarkParams,
    iterations: int = 1,
    diagnostics_level: str = "basic",
    **kwargs,
):
    """
    Run the benchmark problem specified by 'params', 'iterations' times.

    Parameters
    ----------
    params
    iterations
    diagnostics_level
    kwargs

    Returns
    -------
    Tuple: Result data on this run, and the regularization applied
    """
    assert params.problem_name is not None
    assert params.library_name is not None
    P = get_all_problems()[params.problem_name]
    L = get_all_libraries()[params.library_name]

    for k in params.param_names:
        if getattr(params, k) is None:
            params.update_params(**{k: defaults[k]})

    dat = P.data_loader(
        num_rows=params.num_rows,
        storage=params.storage,
        single_precision=params.single_precision,
    )
    os.environ["OMP_NUM_THREADS"] = str(params.threads)

    if params.regularization_strength is None:
        params.regularization_strength = P.regularization_strength
    # Weights have been multiplied by exposure. The new sum of weights
    # should influence the objective function (in order to keep everything comparable
    # to the "weights instead of offset" setup), but this will get undone by weight
    # normalization. So instead divide the penalty by the new weight sum divided by
    # the old weight sum
    reg_multiplier = (
        1 / dat["sample_weight"].mean() if "sample_weight" in dat.keys() else None
    )
    result = L(
        dat,
        distribution=P.distribution,
        alpha=params.regularization_strength,
        l1_ratio=P.l1_ratio,
        iterations=iterations,
        cv=params.cv,
        diagnostics_level=diagnostics_level,
        reg_multiplier=reg_multiplier,
        hessian_approx=params.hessian_approx,
        **kwargs,
    )
    if len(result) > 0:
        result["num_rows"] = dat["y"].shape[0]
        obj_val = get_obj_val(
            dat,
            P.distribution,
            P.regularization_strength,
            P.l1_ratio,
            result["intercept"],
            result["coef"],
        )

        result["obj_val"] = obj_val
        result["num_rows"] = dat["y"].shape[0]

    return result, params.regularization_strength


def get_all_libraries() -> dict[str, Any]:
    """
    Get the names of all available libraries and the functions to benchmark them.

    Returns
    -------
    dict
    """
    all_libraries = {
        "glum": glum_bench,
        "orig-sklearn-fork": orig_sklearn_fork_bench,
        "zeros": zeros_bench,
    }

    if H20_INSTALLED:
        all_libraries["h2o"] = h2o_bench

    if LIBLINEAR_INSTALLED:
        all_libraries["liblinear"] = liblinear_bench

    if R_GLMNET_INSTALLED:
        all_libraries["r-glmnet"] = r_glmnet_bench
    return all_libraries


def get_limited_problems_libraries(
    problem_names: Optional[str], library_names: Optional[str]
) -> tuple[dict, dict]:
    """
    Get only the problems and libraries specified by problem_names and library_names.

    Parameters
    ----------
    problem_names
    library_names

    Returns
    -------
    tuple: dict of problems, dict of libraries
    """
    all_libraries = get_all_libraries()

    if library_names is not None:
        library_names_split = _get_comma_sep_names(library_names)
        libraries = {k: all_libraries[k] for k in library_names_split}
    else:
        libraries = all_libraries
    return get_limited_problems(problem_names), libraries


def get_limited_problems(problem_names: Optional[str]) -> dict[str, Problem]:
    """
    Get the names of problems and problems specified by problem_names.

    Parameters
    ----------
    problem_names

    Returns
    -------
    dict mapping problem name to Problem
    """
    all_problems = get_all_problems()

    if problem_names is not None:
        problem_names_split = _get_comma_sep_names(problem_names)
        problems = {k: all_problems[k] for k in problem_names_split}
    else:
        problems = all_problems
    return problems


def _get_comma_sep_names(xs: str) -> list[str]:
    return [x.strip() for x in xs.split(",")]


def _save_benchmark_results(
    output_dir: str,
    params: BenchmarkParams,
    result,
) -> None:
    results_path = output_dir + "/" + params.get_result_fname()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(results_path + ".pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    cli_run()
