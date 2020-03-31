from click.testing import CliRunner

from glm_benchmarks.main import cli_analyze, cli_run


def test_cli_run__runs_10_rows():
    runner = CliRunner()
    runner.invoke(cli_run, ["--num_rows", "10"])


def test_cli_analyze__runs_default_kws():
    runner = CliRunner()
    runner.invoke(cli_analyze, ["--num_rows", "10"])
