import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import click

from glm_benchmarks.bench_admm import admm_bench
from glm_benchmarks.bench_orig_sklearn_fork import orig_sklearn_fork_bench
from glm_benchmarks.bench_sklearn_fork import sklearn_fork_bench
from glm_benchmarks.problems import Problem, get_all_problems
from glm_benchmarks.util import (
    BenchmarkParams,
    benchmark_params_cli,
    clear_cache,
    get_default_val,
    get_obj_val,
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
    result["num_rows"] = dat["y"].shape[0]

    obj_val = get_obj_val(
        dat,
        P.distribution,
        P.regularization_strength,
        P.l1_ratio,
        result["intercept"],
        result["coef"],
        tweedie_p=get_tweedie_p(params.problem_name),
    )
    result["obj_val"] = obj_val

    return result, params.regularization_strength


def get_tweedie_p(problem_name):
    tweedie = "tweedie" in problem_name
    if tweedie:
        return float(problem_name.split("=")[-1])
    else:
        return None


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


if __name__ == "__main__":
    cli_run()
