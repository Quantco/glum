"""
Basic tests for the benchmark infrastructure.
These tests verify the benchmark pipeline works, not specific problem results.
"""

import numpy as np
import pytest

from glum_benchmarks.problems import get_all_problems
from glum_benchmarks.util import (
    BenchmarkParams,
    execute_problem_library,
    get_all_libraries,
    get_params_from_fname,
    get_tweedie_p,
)


class TestProblems:
    """Tests for problem definitions."""

    def test_get_all_problems_returns_dict(self):
        problems = get_all_problems()
        assert isinstance(problems, dict)
        assert len(problems) > 0

    def test_all_problems_have_required_attributes(self):
        problems = get_all_problems()
        for name, problem in problems.items():
            assert hasattr(problem, "data_loader")
            assert hasattr(problem, "distribution")
            assert hasattr(problem, "alpha")


class TestLibraries:
    """Tests for library availability."""

    def test_get_all_libraries_returns_dict(self):
        libraries = get_all_libraries()
        assert isinstance(libraries, dict)

    def test_glum_always_available(self):
        libraries = get_all_libraries()
        assert "glum" in libraries


class TestBenchmarkParams:
    """Tests for BenchmarkParams."""

    def test_params_creation(self):
        params = BenchmarkParams(
            problem_name="test-problem",
            library_name="glum",
            num_rows=100,
            storage="dense",
            threads=1,
            alpha=0.01,
        )
        assert params.problem_name == "test-problem"
        assert params.library_name == "glum"

    def test_get_result_fname(self):
        params = BenchmarkParams(
            problem_name="test-problem",
            library_name="glum",
            num_rows=100,
            storage="dense",
            threads=1,
            alpha=0.01,
        )
        fname = params.get_result_fname()
        assert "test-problem" in fname
        assert "glum" in fname

    def test_params_roundtrip(self):
        """Test that params can be serialized to filename and back."""
        params = BenchmarkParams(
            problem_name="intermediate-housing-no-weights-lasso-gaussian",
            library_name="glum",
            num_rows=1000,
            storage="dense",
            threads=4,
            alpha=0.001,
        )
        fname = params.get_result_fname() + ".pkl"
        recovered = get_params_from_fname(fname)
        assert recovered.problem_name == params.problem_name
        assert recovered.library_name == params.library_name


class TestExecuteProblem:
    """Tests for execute_problem_library."""

    def test_execute_simple_problem(self):
        """Test that a simple problem runs without error."""
        # Use a small, fast problem
        problems = get_all_problems()
        # Find a gaussian problem (fastest)
        problem_name = next(
            (n for n in problems if "gaussian" in n and "narrow" not in n), None
        )
        if problem_name is None:
            pytest.skip("No suitable test problem found")

        params = BenchmarkParams(
            problem_name=problem_name,
            library_name="glum",
            num_rows=100,  # Small for speed
            storage="dense",
            threads=1,
            alpha=0.1,
        )

        result, _ = execute_problem_library(params, iterations=1)

        assert "coef" in result
        assert "intercept" in result
        assert "runtime" in result
        assert isinstance(result["coef"], np.ndarray)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_tweedie_p_poisson(self):
        assert get_tweedie_p("poisson") == 1

    def test_get_tweedie_p_gamma(self):
        assert get_tweedie_p("gamma") == 2

    def test_get_tweedie_p_gaussian(self):
        assert get_tweedie_p("gaussian") == 0

    def test_get_tweedie_p_explicit(self):
        assert get_tweedie_p("tweedie-p=1.5") == 1.5
