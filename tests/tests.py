import numpy as np
import tensorflow as tf
from click.testing import CliRunner
from scipy import sparse as sps

from glm_benchmarks.bench_tensorflow import sps_to_tf_sparse
from glm_benchmarks.main import cli_analyze, cli_run, get_limited_problems_libraries

# Minimum number of rows for which all benchmarks will run

num_rows = 800


def test_cli_run__runs_10_rows():
    runner = CliRunner()
    result = runner.invoke(cli_run, ["--num_rows", str(num_rows)])
    assert result.exit_code == 0
    problems, libraries = get_limited_problems_libraries("", "")
    # Make sure every problem and library ran
    for prob in problems.keys():
        for lib in libraries.keys():
            string = f"running problem={prob} library={lib}"
            assert string in result.output


def test_cli_analyze__runs_default_kws():
    runner = CliRunner()
    result = runner.invoke(cli_analyze, [])
    assert result.exit_code == 0


def test_sps_to_tf_sparse():
    x = sps.eye(4)
    tensor = sps_to_tf_sparse(x)
    orig_as_dense = x.A
    new_as_dense = tf.sparse.to_dense(tensor).numpy()
    np.testing.assert_almost_equal(orig_as_dense, new_as_dense)
