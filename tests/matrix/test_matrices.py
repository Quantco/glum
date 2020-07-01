import warnings

import numpy as np
import pytest
from scipy import sparse as sps

import quantcore.glm.matrix as mx


def base_array(order="F") -> np.ndarray:
    return np.array([[0, 0], [0, -1.0], [0, 2.0]], order=order)


def dense_glm_data_matrix(order="F") -> mx.DenseGLMDataMatrix:
    return mx.DenseGLMDataMatrix(base_array(order))


def split_matrix(order="F") -> mx.SplitMatrix:
    return mx.csc_to_split(sps.csc_matrix(base_array(order)))


def mkl_sparse_matrix(order="F") -> mx.MKLSparseMatrix:
    return mx.MKLSparseMatrix(sps.csc_matrix(base_array(order)))


def categorical_matrix(order="F"):
    vec = [1, 0, 1]
    return mx.CategoricalMatrix(vec)


def categorical_matrix_col_mult(order="F"):
    vec = [1, 0, 1]
    return mx.CategoricalMatrix(vec, [0.5, 3])


def standardized_dense_shifted(order="F") -> mx.StandardizedMat:
    return mx.StandardizedMat(dense_glm_data_matrix(order), [0.0, 1.0])


def standardized_dense_scaled_shifted(order="F") -> mx.StandardizedMat:
    return mx.StandardizedMat(dense_glm_data_matrix(order), [0.0, 1.0], [0.6, 1.67])


def standardized_sparse(order="F") -> mx.StandardizedMat:
    return mx.StandardizedMat(mkl_sparse_matrix(order), [0.0, 1.0])


def standardized_split(order="F") -> mx.StandardizedMat:
    return mx.StandardizedMat(split_matrix(order), [0.0, 1.0])


unscaled_matrices = [
    dense_glm_data_matrix,
    split_matrix,
    mkl_sparse_matrix,
    categorical_matrix,
    categorical_matrix_col_mult,
]

scaled_matrices = [
    standardized_dense_shifted,
    standardized_dense_scaled_shifted,
    standardized_sparse,
    standardized_split,
]

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
    if isinstance(mat_, mx.CategoricalMatrix):
        expected = np.array([[0, 1], [1, 0], [0, 1]])
        if mat_.col_mult is not None:
            expected = expected * mat_.col_mult[None, :]
    else:
        expected = base_array(order)
    np.testing.assert_allclose(mat_.A, expected)


@pytest.mark.parametrize("mat", scaled_matrices)
@pytest.mark.parametrize("order", ["F", "C"])
def test_to_array_scaled(mat, order):
    mat_ = mat(order)
    assert isinstance(mat_.A, np.ndarray)
    true_mat_part = mat_.mat.A
    if mat_.mult is not None:
        true_mat_part = mat_.mult[None, :] * mat_.mat.A
    np.testing.assert_allclose(mat_.A, true_mat_part + mat_.shift)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "other_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize(
    "other_as_list", [[3.0, -0.1], [[3.0], [-0.1]], [[0.0, 2], [-1, 0]]]
)
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("rows", [None, np.arange(2, dtype=np.int32)])
@pytest.mark.parametrize("cols", [None, np.arange(1, dtype=np.int32)])
def test_dot(mat: type, other_type, other_as_list, order: str, rows, cols):
    other = other_type(other_as_list)
    mat_ = mat(order)
    res = mat_.dot(other, rows, cols)

    mat_subset, vec_subset = process_mat_vec_subsets(
        mat_, other_as_list, rows, cols, cols
    )
    expected = mat_subset.dot(vec_subset)

    np.testing.assert_allclose(res, expected)
    assert isinstance(res, np.ndarray)

    if rows is None and cols is None:
        res2 = mat_ @ other
        np.testing.assert_allclose(res2, expected)


def process_mat_vec_subsets(mat, vec, mat_rows, mat_cols, vec_idxs):
    mat_subset = mat.A
    vec_subset = vec
    if mat_rows is not None:
        mat_subset = mat_subset[mat_rows, :]
    if mat_cols is not None:
        mat_subset = mat_subset[:, mat_cols]
    if vec_idxs is not None:
        vec_subset = np.array(vec_subset)[vec_idxs]
    return mat_subset, vec_subset


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "other_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize(
    "other_as_list",
    # shapes (3,); (3,1), (3, 2);
    [[3.0, -0.1, 0], [[3.0], [-0.1], [0]], [[0, 1.0], [-0.1, 0], [0, 3.0]]],
)
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("rows", [None, np.arange(2, dtype=np.int32)])
@pytest.mark.parametrize("cols", [None, np.arange(1, dtype=np.int32)])
def test_transpose_dot(mat: type, other_type, other_as_list, order: str, rows, cols):
    other = other_type(other_as_list)
    mat_ = mat(order)
    assert np.shape(other)[0] == mat_.shape[0]
    res = mat_.transpose_dot(other, rows, cols)

    mat_subset, vec_subset = process_mat_vec_subsets(
        mat_, other_as_list, rows, cols, rows
    )
    expected = mat_subset.T.dot(vec_subset)
    np.testing.assert_allclose(res, expected)
    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("mat", matrices)
@pytest.mark.parametrize(
    "vec_type", [lambda x: x, np.array, mx.DenseGLMDataMatrix],
)
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("rows", [None, np.arange(2, dtype=np.int32)])
@pytest.mark.parametrize("cols", [None, np.arange(1, dtype=np.int32)])
def test_sandwich(mat: type, vec_type, order, rows, cols):
    vec_as_list = [3, 0.1, 1]
    vec = vec_type(vec_as_list)
    mat_ = mat(order)
    res = mat_.sandwich(vec, rows, cols)

    mat_subset, vec_subset = process_mat_vec_subsets(
        mat_, vec_as_list, rows, cols, rows
    )
    expected = mat_subset.T @ np.diag(vec_subset) @ mat_subset
    if sps.issparse(res):
        res = res.A
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


@pytest.mark.parametrize(
    "mat", matrices,
)
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
def test_get_col_means(mat):
    mat_ = mat()
    weights = np.random.random(mat_.shape[0])
    # TODO: make weights sum to 1 within functions
    weights /= weights.sum()
    means = mat_.get_col_means(weights)
    expected = mat_.A.T.dot(weights)
    np.testing.assert_allclose(means, expected)


@pytest.mark.parametrize("mat", unscaled_matrices)
def test_get_col_means_unweighted(mat):
    mat_ = mat()
    weights = np.ones(mat_.shape[0])
    # TODO: make weights sum to 1 within functions
    weights /= weights.sum()
    means = mat_.get_col_means(weights)
    expected = mat_.A.mean(0)
    np.testing.assert_allclose(means, expected)


@pytest.mark.parametrize("mat", unscaled_matrices)
def test_get_col_stds(mat):
    mat_ = mat()
    weights = np.random.random(mat_.shape[0])
    # TODO: make weights sum to 1
    weights /= weights.sum()
    means = mat_.get_col_means(weights)
    expected = np.sqrt((mat_.A ** 2).T.dot(weights) - means ** 2)
    stds = mat_.get_col_stds(weights, means)
    np.testing.assert_allclose(stds, expected)


@pytest.mark.parametrize("mat", unscaled_matrices)
def test_get_col_stds_unweighted(mat):
    mat_ = mat()
    weights = np.ones(mat_.shape[0])
    # TODO: make weights sum to 1
    weights /= weights.sum()
    means = mat_.get_col_means(weights)
    expected = mat_.A.std(0)
    stds = mat_.get_col_stds(weights, means)
    np.testing.assert_allclose(stds, expected)


@pytest.mark.parametrize("mat", unscaled_matrices)
@pytest.mark.parametrize("center_predictors", [False, True])
@pytest.mark.parametrize("scale_predictors", [False, True])
def test_standardize(mat, center_predictors: bool, scale_predictors: bool):
    mat_: mx.MatrixBase = mat()
    asarray = mat_.A.copy()
    weights = np.random.rand(mat_.shape[0])
    weights /= weights.sum()

    true_means = asarray.T.dot(weights)
    true_sds = np.sqrt((asarray ** 2).T.dot(weights) - true_means ** 2)

    standardized, means, stds = mat_.standardize(
        weights, center_predictors, scale_predictors
    )
    assert isinstance(standardized, mx.StandardizedMat)
    assert isinstance(standardized.mat, type(mat_))
    if center_predictors:
        np.testing.assert_allclose(standardized.transpose_dot(weights), 0, atol=1e-11)
        np.testing.assert_allclose(means, asarray.T.dot(weights))
    else:
        np.testing.assert_almost_equal(means, 0)

    if scale_predictors:
        np.testing.assert_allclose(stds, true_sds)
    else:
        assert stds is None

    expected_sds = true_sds if scale_predictors else np.ones_like(true_sds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        one_over_sds = np.nan_to_num(1 / expected_sds)

    expected_mat = asarray * one_over_sds
    if center_predictors:
        expected_mat -= true_means * one_over_sds
    np.testing.assert_allclose(standardized.A, expected_mat)

    unstandardized = standardized.unstandardize()
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
