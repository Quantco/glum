import numpy as np
import pytest
from scipy import sparse as sps

import glm_benchmarks.matrix as mx
from glm_benchmarks.matrix.sandwich.sandwich import csr_dense_sandwich


def base_array(order="F") -> np.ndarray:
    return np.array([[0, 0], [0, -1.0], [0, 2.0]], order=order)


def dense_glm_data_matrix(order="F") -> mx.DenseGLMDataMatrix:
    return mx.DenseGLMDataMatrix(base_array(order))


def split_matrix(order="F") -> mx.SplitMatrix:
    return mx.SplitMatrix(sps.csc_matrix(base_array(order)), threshold=0.1)


def mkl_sparse_matrix(order="F") -> mx.MKLSparseMatrix:
    return mx.MKLSparseMatrix(sps.csc_matrix(base_array(order)))


def col_scaled_dense(order="F") -> mx.ColScaledMat:
    return mx.ColScaledMat(dense_glm_data_matrix(order), [0.0, 1.0])


def col_scaled_sparse(order="F") -> mx.ColScaledMat:
    return mx.ColScaledMat(mkl_sparse_matrix(order), [0.0, 1.0])


def col_scaled_split(order="F") -> mx.ColScaledMat:
    return mx.ColScaledMat(split_matrix(order), [0.0, 1.0])


unscaled_matrices = [
    dense_glm_data_matrix,
    split_matrix,
    mkl_sparse_matrix,
]

scaled_matrices = [col_scaled_dense, col_scaled_sparse, col_scaled_split]

matrices = unscaled_matrices + scaled_matrices  # type: ignore


@pytest.mark.parametrize("mat", matrices)
def test_get_col(mat):
    i = 1
    mat_ = mat()
    col = mat_.getcol(i)
    if not isinstance(col, np.ndarray):
        col = col.A
    np.testing.assert_almost_equal(col, mat_.A[:, [i]])


@pytest.mark.parametrize("mat", unscaled_matrices)
@pytest.mark.parametrize("order", ["F", "C"])
def test_to_array(mat, order):
    mat_ = mat(order)
    assert isinstance(mat_.A, np.ndarray)
    np.testing.assert_allclose(mat_.A, base_array(order))


@pytest.mark.parametrize("mat", scaled_matrices)
@pytest.mark.parametrize("order", ["F", "C"])
def test_to_array_scaled(mat, order):
    mat_ = mat(order)
    assert isinstance(mat_.A, np.ndarray)
    np.testing.assert_allclose(mat_.A, base_array(order) + np.array([[0, 1]]))


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
def test_dot_vector(mat: type, vec_type):
    vec_as_list = [3.0, -0.1]
    vec = vec_type(vec_as_list)
    mat_ = mat()
    res = mat_.dot(vec)
    expected = mat_.A.dot(vec_as_list)
    np.testing.assert_allclose(res, expected)
    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_dot_vector_matmul(mat: type, vec_type, order):
    vec_as_list = [3.0, -0.1]
    vec = vec_type(vec_as_list)
    mat_ = mat(order)
    res = mat_ @ vec
    expected = mat_.A @ vec_as_list
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_dot_dense_matrix(mat: type, vec_type, order):
    vec_as_list = [[3.0], [-0.1]]
    vec = vec_type(vec_as_list)
    mat_ = mat(order)
    res = mat_.dot(vec)
    expected = mat_.A.dot(vec_as_list)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_dot_dense_matrix_matmul(mat: type, vec_type, order):
    vec_as_list = [[3.0], [-0.1]]
    vec = vec_type(vec_as_list)
    mat_ = mat(order)
    res = mat_ @ vec
    expected = mat_.A @ vec_as_list
    np.testing.assert_allclose(res, expected)


def test_dense_sandwich():
    sp_mat = sps.csr_matrix(sps.eye(3))
    d = np.arange(3).astype(float)
    B = np.ones((3, 2))
    result = csr_dense_sandwich(sp_mat, B, d)
    expected = sp_mat.A @ np.diag(d) @ B
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_sandwich(mat: type, vec_type, order):
    vec_as_list = [3, 0.1, 1]
    vec = vec_type(vec_as_list)
    mat_ = mat(order)
    res = mat_.sandwich(vec)
    expected = mat_.A.T @ np.diag(vec_as_list) @ mat_.A
    np.testing.assert_allclose(res, expected)


# TODO: make this work for SplitMatrix too, or figure out a way to not need transpose
# TODO: make sure we have sklearn tests for each matrix setup
@pytest.mark.parametrize(
    "mat",
    [dense_glm_data_matrix, mkl_sparse_matrix, col_scaled_dense, col_scaled_sparse],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_transpose(mat: type, order):
    mat_ = mat(order)
    res = mat_.T.A
    expected = mat_.A.T
    assert res.shape == (mat_.shape[1], mat_.shape[0])
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("matrix_shape", [(3,), (1, 3), (2, 3)])
@pytest.mark.parametrize(
    "mat",
    [dense_glm_data_matrix, mkl_sparse_matrix, col_scaled_dense, col_scaled_sparse],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_r_matmul(mat, matrix_shape, order):
    v = np.ones(matrix_shape)
    mat_ = mat(order)
    result = v @ mat_
    expected = v @ mat_.A
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize("order", ["F", "C"])
def test_dot_raises(mat, order):
    mat_ = mat(order)
    with pytest.raises(ValueError):
        mat_.dot(np.ones((10, 1)))


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("order", ["F", "C"])
def test_astype(mat, dtype, order):
    mat_ = mat(order)
    new_mat = mat_.astype(dtype)
    assert np.issubdtype(new_mat.dtype, dtype)
    vec = np.zeros(mat_.shape[1], dtype=dtype)
    res = new_mat.dot(vec)
    assert res.dtype == new_mat.dtype
