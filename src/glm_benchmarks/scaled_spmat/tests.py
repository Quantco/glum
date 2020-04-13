import numpy as np
import pytest
from scipy import sparse as sps

from glm_benchmarks.scaled_spmat.scaled_spmat import ColScaledSpMat, RowScaledSpMat


def row_scaled_mat() -> RowScaledSpMat:
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    shift = np.random.uniform(0, 1, n_rows)
    return RowScaledSpMat(sp_mat, shift)


def col_scaled_mat() -> ColScaledSpMat:
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    col_shift = np.random.uniform(0, 1, n_cols)
    return ColScaledSpMat(sp_mat, col_shift)


@pytest.fixture
def col_scaled_mat_fixture():
    return col_scaled_mat()


@pytest.fixture
def row_scaled_mat_fixture():
    return row_scaled_mat()


def test_setup_and_densify_col():

    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    col_shift = np.random.uniform(0, 1, n_cols)
    col_scaled_mat = ColScaledSpMat(sp_mat, col_shift)
    expected = sp_mat.A + col_shift[None, :]
    assert col_scaled_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(col_scaled_mat.A, expected)


def test_setup_and_densify_row():

    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    shift = np.random.uniform(0, 1, n_rows)
    scaled_mat = RowScaledSpMat(sp_mat, shift)
    expected = sp_mat.A + shift[:, None]
    assert scaled_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(scaled_mat.A, expected)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_col_dot_dense_vec(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    mat = np.random.uniform(0, 2, scaled_mat.shape[1])
    result = scaled_mat.dot(mat)
    np.testing.assert_almost_equal(result, scaled_mat.A.dot(mat))


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_col_dot_dense_mat(scaled_mat_builder):

    scaled_mat = scaled_mat_builder()
    mat_shape = (scaled_mat.shape[1], 2)
    mat = np.random.uniform(0, 2, mat_shape)
    result = scaled_mat.dot(mat)
    expected = scaled_mat.A.dot(mat)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_col_dot_sparse_mat(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    mat_shape = (scaled_mat.shape[1], 2)

    mat = sps.csc_matrix(np.random.uniform(0, 2, mat_shape))
    result = scaled_mat.dot(mat)
    expected = scaled_mat.A.dot(mat.A)
    np.testing.assert_almost_equal(result.A, expected)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_power(scaled_mat_builder):
    p = 3
    scaled_mat = scaled_mat_builder()
    result = scaled_mat.power(p)
    assert isinstance(result, type(scaled_mat))
    assert result.shape == scaled_mat.shape
    np.testing.assert_almost_equal(result.A, scaled_mat.A ** p)


def test_transpose_type_col(col_scaled_mat_fixture: ColScaledSpMat):
    t = col_scaled_mat_fixture.T
    assert isinstance(t, RowScaledSpMat)
    assert t.shape == (col_scaled_mat_fixture.shape[1], col_scaled_mat_fixture.shape[0])


def test_transpose_type_row(row_scaled_mat_fixture: RowScaledSpMat):
    t = row_scaled_mat_fixture.T
    assert isinstance(t, ColScaledSpMat)
    assert t.shape == (row_scaled_mat_fixture.shape[1], row_scaled_mat_fixture.shape[0])


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_transpose_against_dense(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    np.testing.assert_almost_equal(scaled_mat.T.A, scaled_mat.A.T)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_transpose_reversible(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    two_trans = scaled_mat.T.T
    assert (two_trans.mat != scaled_mat.mat).sum() == 0
    assert (two_trans.shift == scaled_mat.shift).all()


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_multiply(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    other = np.random.normal(0, 1, (scaled_mat.shape[scaled_mat.scale_axis]))
    result = scaled_mat.multiply(other)
    np.testing.assert_almost_equal(
        result.A, scaled_mat.A * np.expand_dims(other, 1 - scaled_mat.scale_axis)
    )


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_multiply_scalar(scaled_mat_builder):
    other = 4.3
    scaled_mat = scaled_mat_builder()
    result = scaled_mat.multiply(other)
    np.testing.assert_almost_equal(result.A, scaled_mat.A * other)
