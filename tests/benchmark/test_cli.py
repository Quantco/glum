import os
import tempfile
from typing import Any, Dict, List

import click
import pytest
from click.testing import CliRunner

from glm_benchmarks.main import cli_run, get_default_val, identify_parameter_fnames
from glm_benchmarks.util import BenchmarkParams, benchmark_params_cli


@pytest.mark.parametrize(
    "cli_options, expected_params",
    [
        ([], {}),
        (["--num_rows", "1000", "--cv", "True"], {"num_rows": 1000, "cv": True}),
    ],
)
def test_make_params(cli_options: List[str], expected_params: Dict[str, Any]):
    @click.command()
    @benchmark_params_cli
    def params_test(params: BenchmarkParams):
        for k in params.param_names:
            as_expected = getattr(params, k) == expected_params.get(k)
            if not as_expected:
                click.echo(
                    f"""
                    For parameter {k} expected {expected_params.get(k)} but
                    got {getattr(params, k)}."""
                )

    runner = CliRunner()
    result = runner.invoke(params_test, cli_options)
    if not result.exit_code == 0:
        raise ValueError(result.output)


def test_correct_problems_run():
    output_dir = "test_output_tmp"

    problem_names = [
        "narrow-insurance-weights-l2-gamma",
        "wide-insurance-no-weights-net-poisson",
    ]
    library_names = ["zeros", "sklearn-fork"]
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
            raise ValueError(result.output)
        problems_run = os.listdir(d)

    expected_problems_run = [
        BenchmarkParams(
            pn,
            ln,
            num_rows=num_rows,
            regularization_strength=regularization_strength,
            **{
                k: get_default_val(k)
                for k in ["storage", "threads", "single_precision", "cv"]
            },
        ).get_result_fname()
        + ".pkl"
        for pn in problem_names
        for ln in library_names
    ]

    expected_problems_run_2 = [
        "narrow-insurance-weights-l2-gamma_zeros_20_dense_4_False_1000.0_False.pkl",
        "narrow-insurance-weights-l2-gamma_sklearn-fork_20_dense_4_False_1000.0_False.pkl",
        "wide-insurance-no-weights-net-poisson_zeros_20_dense_4_False_1000.0_False.pkl",
        "wide-insurance-no-weights-net-poisson_sklearn-fork_20_dense_4_False_1000.0_False.pkl",
    ]

    assert sorted(problems_run) == sorted(expected_problems_run)
    assert sorted(problems_run) == sorted(expected_problems_run_2)


def test_correct_problems_analyzed():
    """
    Everything should be analyzed when cli_analyze is not given any parameters.
    This test checks that everything in benchmark_output/ is run, so if that directory
    is empty, the test will not be meaningful.
    """
    output_dir = "benchmark_output"
    to_analyze = identify_parameter_fnames(output_dir, BenchmarkParams())
    assert sorted(to_analyze) == sorted(os.listdir(output_dir))
