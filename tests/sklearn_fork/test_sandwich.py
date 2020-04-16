import numpy as np
import pytest
from scipy import sparse as sps

from glm_benchmarks.scaled_spmat import ColScaledSpMat
from glm_benchmarks.sklearn_fork._glm import _safe_sandwich_dot

n_rows = 5


def get_x_array() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, (n_rows, 2))


@pytest.fixture
def x_array() -> np.ndarray:
    return get_x_array()


@pytest.fixture
def d() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, n_rows)


def get_expected_answer(x_array: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (x_array.T * d) @ x_array


def test_array(x_array: np.ndarray, d: np.ndarray):
    answer = _safe_sandwich_dot(x_array, d)
    expected_answer = get_expected_answer(x_array, d)
    np.testing.assert_almost_equal(answer, expected_answer)


def test_sparse(x_array: np.ndarray, d: np.ndarray):
    answer = _safe_sandwich_dot(sps.csc_matrix(x_array), d)
    expected_answer = get_expected_answer(x_array, d)
    np.testing.assert_almost_equal(answer, expected_answer)


def test_scaled(x_array: np.ndarray, d: np.ndarray):
    scaled_mat = ColScaledSpMat(sps.csc_matrix(x_array), np.zeros(x_array.shape[1]))
    answer = _safe_sandwich_dot(scaled_mat, d)
    expected_answer = get_expected_answer(x_array, d)
    np.testing.assert_almost_equal(answer, expected_answer)
