import warnings

import numpy as np
import pytest
from scipy import sparse as sps

import quantcore.glm.matrix as mx
from quantcore.glm.matrix.sandwich.sandwich import csr_dense_sandwich


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
@pytest.mark.parametrize("i", [1, -2])
def test_getcol(mat, i):
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
    "other_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize(
    "other_as_list", [[3.0, -0.1], [[3.0], [-0.1]], [[0.0, 2], [-1, 0]]]
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_dot(mat: type, other_type, other_as_list, order: str):
    other = other_type(other_as_list)
    mat_ = mat(order)
    res = mat_.dot(other)
    res2 = mat_ @ other
    expected = mat_.A.dot(other_as_list)
    np.testing.assert_allclose(res, expected)
    np.testing.assert_allclose(res2, expected)
    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "other_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize(
    "other_as_list",
    # shapes (3,); (3,1); (3, 2)
    [[3.0, -0.1, 0], [[3.0], [-0.1], [0]], [[0, 1.0], [-0.1, 0], [0, 3.0]]],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_transpose_dot(mat: type, other_type, other_as_list, order: str):
    other = other_type(other_as_list)
    mat_ = mat(order)
    assert np.shape(other)[0] == mat_.shape[0]
    res = mat_.transpose_dot(other)
    expected = mat_.A.T.dot(other_as_list)
    np.testing.assert_allclose(res, expected)
    assert isinstance(res, np.ndarray)


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


# TODO: make sure we have sklearn tests for each matrix setup
@pytest.mark.parametrize("mat", [dense_glm_data_matrix, mkl_sparse_matrix])
@pytest.mark.parametrize("order", ["F", "C"])
def test_transpose(mat: type, order):
    mat_ = mat(order)
    res = mat_.T.A
    expected = mat_.A.T
    assert res.shape == (mat_.shape[1], mat_.shape[0])
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize(
    "vec_as_list",
    # shapes (3,); (1,3); (2, 3)
    [[3.0, -0.1, 0], [[3.0, -0.1, 0]], [[0, -0.1, 1.0], [-0.1, 0, 3]]],
)
@pytest.mark.parametrize("order", ["F", "C"])
def test_rmatmul(mat: type, vec_type, vec_as_list, order: str):
    vec = vec_type(vec_as_list)
    mat_ = mat(order)
    res = mat_.__rmatmul__(vec)
    res2 = vec @ mat_
    expected = vec_as_list @ mat_.A
    np.testing.assert_allclose(res, expected)
    np.testing.assert_allclose(res2, expected)
    assert isinstance(res, np.ndarray)


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


@pytest.mark.parametrize("mat", unscaled_matrices)
@pytest.mark.parametrize("scale_predictors", [False, True])
def test_standardize(mat, scale_predictors: bool):
    mat_: mx.MatrixBase = mat()
    asarray = mat_.A.copy()
    weights = np.random.rand(mat_.shape[0])
    weights /= weights.sum()

    true_means = asarray.T.dot(weights)
    true_sds = np.sqrt((asarray ** 2).T.dot(weights) - true_means ** 2)

    standardized, means, stds = mat_.standardize(weights, scale_predictors)
    assert isinstance(standardized, mx.ColScaledMat)
    assert isinstance(standardized.mat, type(mat_))

    np.testing.assert_allclose(means, asarray.T.dot(weights))
    if scale_predictors:
        np.testing.assert_allclose(stds, true_sds)
    else:
        assert stds is None

    expected_sds = true_sds if scale_predictors else np.ones_like(true_sds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        one_over_sds = np.nan_to_num(1 / expected_sds)

    np.testing.assert_allclose(standardized.A, (asarray - true_means) * one_over_sds)

    unstandardized = standardized.unstandardize(stds)
    assert isinstance(unstandardized, type(mat_))
    np.testing.assert_allclose(unstandardized.A, asarray)


@pytest.mark.parametrize("mat", matrices)
def test_indexing_int_row(mat):
    mat_ = mat()
    res = mat_[0, :]
    if not isinstance(res, np.ndarray):
        res = res.A
    expected = mat_.A[0, :]
    np.testing.assert_allclose(np.squeeze(res), expected)


@pytest.mark.parametrize("mat", matrices)
def test_indexing_range_row(mat):
    mat_ = mat()
    res = mat_[0:2, :]
    if not isinstance(res, np.ndarray):
        res = res.A
    expected = mat_.A[0:2, :]
    np.testing.assert_allclose(np.squeeze(res), expected)
