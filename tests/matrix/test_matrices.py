import numpy as np
import pytest
from scipy import sparse as sps

import glm_benchmarks.matrix as mx
from glm_benchmarks.matrix.sandwich.sandwich import csr_dense_sandwich


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


@pytest.mark.parametrize("mat", matrices)
def test_get_col(mat):
    i = 1
    col = mat.getcol(i)
    if not isinstance(col, np.ndarray):
        col = col.A
    np.testing.assert_almost_equal(col, mat.A[:, [i]])


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
    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
def test_dot_vector_matmul(mat: mx.MatrixBase, vec_type):
    vec_as_list = [3.0, -0.1]
    vec = vec_type(vec_as_list)
    res = mat @ vec
    expected = mat.A @ vec_as_list
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
def test_dot_dense_matrix(mat: mx.MatrixBase, vec_type):
    vec_as_list = [[3.0], [-0.1]]
    vec = vec_type(vec_as_list)
    res = mat.dot(vec)
    expected = mat.A.dot(vec_as_list)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
def test_dot_dense_matrix_matmul(mat: mx.MatrixBase, vec_type):
    vec_as_list = [[3.0], [-0.1]]
    vec = vec_type(vec_as_list)
    res = mat @ vec
    expected = mat.A @ vec_as_list
    np.testing.assert_allclose(res, expected)


def test_dense_sandwich():
    sp_mat = sps.csr_matrix(sps.eye(3))
    d = np.arange(3).astype(float)
    B = np.ones((3, 2))
    result = csr_dense_sandwich(sp_mat, B, d)
    expected = sp_mat.A @ np.diag(d) @ B
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "mat",
    [
        dense_glm_data_matrix(),
        mkl_sparse_matrix(),
        col_scaled_sp_mat(),
        split_matrix(),
    ],
)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
def test_sandwich(mat: mx.MatrixBase, vec_type):
    vec_as_list = [3, 0.1, 1]
    vec = vec_type(vec_as_list)
    res = mat.sandwich(vec)
    expected = mat.A.T @ np.diag(vec_as_list) @ mat.A
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize(
    "mat",
    [
        dense_glm_data_matrix(),
        col_scaled_sp_mat(),
        row_scaled_sp_mat(),
        mkl_sparse_matrix(),
    ],
)
def test_transpose(mat: mx.MatrixBase):
    res = mat.T.A
    expected = mat.A.T
    assert res.shape == (mat.shape[1], mat.shape[0])
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("matrix_shape", [(3,), (1, 3), (2, 3)])
@pytest.mark.parametrize(
    "mat",
    [
        dense_glm_data_matrix(),
        col_scaled_sp_mat(),
        split_matrix(),
        mkl_sparse_matrix(),
    ],
)
def test_r_matmul(mat, matrix_shape):
    v = np.ones(matrix_shape)
    result = v @ mat
    expected = v @ mat.A
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mat", matrices)
def test_dot_raises(mat):
    with pytest.raises(ValueError):
        mat.dot(np.ones((10, 1)))


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_astype(mat, dtype):
    new_mat = mat.astype(dtype)
    assert np.issubdtype(new_mat.dtype, dtype)
    vec = np.zeros(mat.shape[1], dtype=dtype)
    res = new_mat.dot(vec)
    assert res.dtype == new_mat.dtype
