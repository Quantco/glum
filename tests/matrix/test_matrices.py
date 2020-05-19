import numpy as np
import pytest
from scipy import sparse as sps

import glm_benchmarks.matrix as mx


def base_array() -> np.ndarray:
    return np.array([[0, 0, 0], [0.0, -1.0, 2.0]]).T


def dense_glm_data_matrix() -> mx.DenseGLMDataMatrix:
    return mx.DenseGLMDataMatrix(base_array())


def col_scaled_sp_mat() -> mx.ColScaledSpMat:
    return mx.ColScaledSpMat(sps.csc_matrix(base_array()), [0.0, 1.0])


def row_scaled_sp_mat() -> mx.RowScaledSpMat:
    return mx.RowScaledSpMat(sps.csc_matrix(base_array()), [0.0, 1.0, -0.1])


def split_matrix() -> mx.SplitMatrix:
    return mx.SplitMatrix(sps.csc_matrix(base_array()), threshold=0.1)


def mkl_sparse_matrix() -> mx.MKLSparseMatrix:
    return mx.MKLSparseMatrix(sps.csc_matrix(base_array()))


matrices = [
    dense_glm_data_matrix(),
    col_scaled_sp_mat(),
    row_scaled_sp_mat(),
    split_matrix(),
    mkl_sparse_matrix(),
]


@pytest.mark.parametrize(
    "mat", [dense_glm_data_matrix(), split_matrix(), mkl_sparse_matrix()]
)
def test_get_col(mat):
    i = 1
    col = mat.getcol(i)
    np.testing.assert_almost_equal(col.A, base_array()[:, [i]])


@pytest.mark.parametrize(
    "mat", [dense_glm_data_matrix(), split_matrix(), mkl_sparse_matrix()]
)
def test_to_array(mat):
    assert isinstance(mat.A, np.ndarray)
    np.testing.assert_allclose(mat.A, base_array())


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
def test_dot_vector(mat: mx.MatrixBase, vec_type):
    vec_as_list = [3.0, -0.1]
    vec = vec_type(vec_as_list)
    res = mat.dot(vec)
    expected = mat.A.dot(vec_as_list)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", matrices)
def test_dot_dense_matrix(mat: mx.MatrixBase):
    vec = [[3.0], [-0.1]]
    res = mat.dot(vec)
    expected = mat.A.dot(vec)
    np.testing.assert_allclose(res, expected)
