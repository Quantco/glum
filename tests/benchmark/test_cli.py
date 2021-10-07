import os
import tempfile
import warnings
from typing import Any, Dict, List

import click
import pytest
from click.testing import CliRunner

from glum_benchmarks.cli_analyze import _identify_parameter_fnames
from glum_benchmarks.cli_run import cli_run
from glum_benchmarks.util import BenchmarkParams, benchmark_params_cli, defaults


@pytest.mark.parametrize(
    "cli_options, expected_params",
    [
        ([], {}),
        (["--num_rows", "1000", "--cv", "True"], {"num_rows": 1000, "cv": True}),
    ],
)
def test_make_params(cli_options: List[str], expected_params: Dict[str, Any]):
    """
    Test that the basic command line interface runs and that the benchmark_params_cli \
    decorator works.

    Parameters
    ----------
    cli_options: List of strings
    expected_params
    """

    @click.command()
    @benchmark_params_cli
    def _params_test(params: BenchmarkParams):
        for k in params.param_names:
            as_expected = getattr(params, k) == expected_params.get(k)
            if not as_expected:
                click.echo(
                    f"""
                    For parameter {k} expected {expected_params.get(k)} but
                    got {getattr(params, k)}."""
                )

    runner = CliRunner()
    result = runner.invoke(_params_test, cli_options)
    if not result.exit_code == 0:
        raise ValueError(result.output)


def test_correct_problems_run():
    """Test that the correct problems are run given certain command-line inputs."""
    output_dir = "test_output_tmp"

    problem_names = [
        "narrow-insurance-weights-l2-gamma",
        "wide-insurance-no-weights-net-poisson",
    ]
    library_names = ["zeros", "glum"]
    num_rows = 20
    regularization_strength = 1000.0

    assert output_dir not in os.listdir()
    with tempfile.TemporaryDirectory() as d:
        args = [
            "--problem_name",
            ",".join(problem_names),
            "--library_name",
            ",".join(library_names),
            "--num_rows",
            str(num_rows),
            "--regularization_strength",
            str(regularization_strength),
            "--output_dir",
            d,
        ]
        runner = CliRunner()
        result = runner.invoke(cli_run, args)
        if not result.exit_code == 0:
            problem_name_str = " ".join(args)
            raise ValueError(
                f"""Failed on problem {problem_name_str} with output: \n {result.output}"""
            )
        problems_run = os.listdir(d)

    expected_problems_run = [
        BenchmarkParams(
            pn,
            ln,
            num_rows=num_rows,
            regularization_strength=regularization_strength,
            **{
                k: defaults[k]
                for k in [
                    "storage",
                    "threads",
                    "single_precision",
                    "cv",
                    "hessian_approx",
                    "diagnostics_level",
                ]
            },
        ).get_result_fname()
        + ".pkl"
        for pn in problem_names
        for ln in library_names
    ]

    n_threads = os.environ.get("OMP_NUM_THREADS", os.cpu_count())

    expected_problems_run_2 = [
        f"narrow-insurance-weights-l2-gamma_zeros_20_dense_{n_threads}_False_1000.0_Fals"
        "e_0.0_basic.pkl",
        f"narrow-insurance-weights-l2-gamma_glum_20_dense_{n_threads}_False_10"
        "00.0_False_0.0_basic.pkl",
        f"wide-insurance-no-weights-net-poisson_zeros_20_dense_{n_threads}_False_1000.0"
        "_False_0.0_basic.pkl",
        f"wide-insurance-no-weights-net-poisson_glum_20_dense_{n_threads}_False"
        "_1000.0_False_0.0_basic.pkl",
    ]

    assert sorted(problems_run) == sorted(expected_problems_run)
    assert sorted(problems_run) == sorted(expected_problems_run_2)


def test_correct_problems_analyzed():
    """
    Test that cli_analyze runs on the correct problems.

    Everything should be analyzed when cli_analyze is not given any parameters.
    This test checks that everything in benchmark_output/ is run, so if that directory
    is empty, the test will not be meaningful.
    """
    output_dir = "benchmark_output"
    if output_dir not in os.listdir():
        warnings.warn("Output directory not found")
        return

    to_analyze = _identify_parameter_fnames(output_dir, BenchmarkParams())
    assert sorted(to_analyze) == sorted(os.listdir(output_dir))
