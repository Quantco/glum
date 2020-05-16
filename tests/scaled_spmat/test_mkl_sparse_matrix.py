import numpy as np
import pytest
from scipy import sparse as sps

from glm_benchmarks.matrix import MKLSparseMatrix


@pytest.fixture
def x() -> sps.csc_matrix:
    np.random.seed(0)
    return sps.random(10, 3).tocsc()


def test_mkl_sparse_init(x: sps.csc_matrix):
    one = MKLSparseMatrix(x)
    two = MKLSparseMatrix((x.data, x.indices, x.indptr), shape=x.shape)
    three = MKLSparseMatrix(x.A)
    np.testing.assert_allclose(one.A, two.A)
    np.testing.assert_allclose(one.A, three.A)


def test_to_csc(x: sps.csc_matrix):
    result = x.tocsc()
    assert isinstance(result, sps.csc_matrix)


@pytest.mark.parametrize("matrix_shape", [(3,), (3, 1), (3, 2)])
def test_dot(x: sps.csc_matrix, matrix_shape):
    v = np.ones(matrix_shape)
    result = x.dot(v)
    expected = x.tocsc().dot(v)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("matrix_shape", [(10,), (1, 10), (2, 10)])
def test_r_matmul(x: sps.csc_matrix, matrix_shape):
    v = np.ones(matrix_shape)
    result = v @ x
    expected = v @ x.tocsc()
    np.testing.assert_allclose(result, expected)
