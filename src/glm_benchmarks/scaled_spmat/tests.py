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
    mult = np.random.uniform(0, 1, n_rows)
    return RowScaledSpMat(sp_mat, shift, mult)


def col_scaled_mat() -> ColScaledSpMat:
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    col_shift = np.random.uniform(0, 1, n_cols)
    col_mult = np.random.uniform(0, 1, n_cols)
    return ColScaledSpMat(sp_mat, col_shift, col_mult)


@pytest.fixture
def col_scaled_mat_fixture():
    return col_scaled_mat()


def test_setup_and_densify():

    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    col_shift = np.random.uniform(0, 1, n_cols)
    col_mult = np.random.uniform(0, 1, n_cols)
    col_scaled_mat = ColScaledSpMat(sp_mat, col_shift, col_mult)
    expected = (sp_mat.A + col_shift[None, :]) * col_mult[None, :]
    assert col_scaled_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(col_scaled_mat.A, expected)


def test_setup_and_densify_row():

    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    shift = np.random.uniform(0, 1, n_rows)
    mult = np.random.uniform(0, 1, n_rows)
    scaled_mat = RowScaledSpMat(sp_mat, shift, mult)
    expected = (sp_mat.A + shift[:, None]) * mult[:, None]
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


def test_power(col_scaled_mat_fixture: ColScaledSpMat):
    p = 3
    result = col_scaled_mat_fixture.power(p)
    assert isinstance(result, ColScaledSpMat)
    assert result.shape == col_scaled_mat_fixture.shape
    np.testing.assert_almost_equal(result.A, col_scaled_mat_fixture.A ** p)


def test_transpose_smoke_test(col_scaled_mat_fixture: ColScaledSpMat):
    t = col_scaled_mat_fixture.T
    assert isinstance(t, RowScaledSpMat)
    assert t.shape == (col_scaled_mat_fixture.shape[1], col_scaled_mat_fixture.shape[0])


def test_transpose_against_dense(col_scaled_mat_fixture: ColScaledSpMat):
    np.testing.assert_almost_equal(
        col_scaled_mat_fixture.T.A, col_scaled_mat_fixture.A.T
    )


def test_transpose_reversible(col_scaled_mat_fixture: ColScaledSpMat):
    two_trans = col_scaled_mat_fixture.T.T
    assert (two_trans.mat != col_scaled_mat_fixture.mat).sum() == 0
    assert (two_trans.shift == col_scaled_mat_fixture.shift).all()
