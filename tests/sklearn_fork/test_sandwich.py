import numpy as np
import pytest
from scipy import sparse as sps

from glm_benchmarks.fast_sandwich_dot import sandwich_cython
from glm_benchmarks.fast_sandwich_dot_py import sandwich_dot_whiten
from glm_benchmarks.scaled_spmat import ColScaledSpMat
from glm_benchmarks.sklearn_fork._glm import _safe_sandwich_dot

n_rows = 5


def x_array() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, (n_rows, 2))


@pytest.fixture
def x_array_fixture() -> np.ndarray:
    return x_array()


def d() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(0, 1, n_rows)


@pytest.fixture
def d_fixture() -> np.ndarray:
    return d()


def x_sparse() -> sps.csc_matrix:
    return sps.csc_matrix(x_array())


def x_scaled():
    return ColScaledSpMat(x_sparse(), np.zeros(x_array().shape[1]))


def get_expected_answer() -> np.ndarray:
    x = x_array()
    return (x.T * d()) @ x


@pytest.mark.parametrize("arr_func", [x_array, x_sparse, x_scaled])
def test_sklearn_sandwich(arr_func, d_fixture: np.ndarray):
    arr = arr_func()
    answer = _safe_sandwich_dot(arr, d_fixture)
    np.testing.assert_almost_equal(answer, get_expected_answer())


def test_sandwich_dot_whiten(x_array_fixture: np.ndarray, d_fixture: np.ndarray):
    answer = sandwich_dot_whiten(x_array_fixture, d_fixture)
    np.testing.assert_almost_equal(answer, get_expected_answer())


def test_cython_sandwich(x_array_fixture: np.ndarray, d_fixture: np.ndarray):
    res = sandwich_cython(x_array_fixture, d_fixture)
    np.testing.assert_almost_equal(res, get_expected_answer())
