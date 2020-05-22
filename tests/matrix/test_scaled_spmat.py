from typing import Union

import numpy as np
import pytest
from scipy import sparse as sps

from glm_benchmarks.matrix.scaled_spmat import ColScaledSpMat, RowScaledSpMat


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


@pytest.mark.parametrize("mat_type", [ColScaledSpMat, RowScaledSpMat])
def test_setup_and_densify_row(mat_type):
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    shift = np.random.uniform(0, 1, sp_mat.shape[mat_type.scale_axis()])
    scaled_mat = mat_type(sp_mat, shift)
    expected = sp_mat.A + np.expand_dims(shift, 1 - scaled_mat.scale_axis())
    assert scaled_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(scaled_mat.A, expected)


def as_sparse(x: Union[ColScaledSpMat, RowScaledSpMat]) -> sps.csc_matrix:
    return sps.csc_matrix(x.A)


def test_row_scaled_get_row():
    scaled_mat = row_scaled_mat()
    i = 1
    result = scaled_mat.getrow(i)
    expected = scaled_mat.A[[i], :]
    np.testing.assert_allclose(result.toarray(), expected)


def test_col_scaled_mat_get_row():
    scaled_mat = col_scaled_mat()
    i = 1
    result = scaled_mat.getrow(i)
    expected = scaled_mat.A[i, :]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_power(scaled_mat_builder):
    p = 3
    scaled_mat = scaled_mat_builder()
    result = scaled_mat.power(p)
    assert isinstance(result, type(scaled_mat))
    assert result.shape == scaled_mat.shape
    np.testing.assert_almost_equal(result.A, scaled_mat.A ** p)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_transpose_reversible(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    two_trans = scaled_mat.T.T
    assert (two_trans.mat != scaled_mat.mat).sum() == 0
    assert (two_trans.shift == scaled_mat.shift).all()


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_multiply(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    other = np.random.normal(0, 1, scaled_mat.shape[scaled_mat.scale_axis()])
    other = np.expand_dims(other, 1 - scaled_mat.scale_axis())

    expected = as_sparse(scaled_mat).multiply(other)
    result = scaled_mat.multiply(other)
    np.testing.assert_almost_equal(result.A, expected.A)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_multiply_scalar(scaled_mat_builder):
    other = 4.3
    scaled_mat = scaled_mat_builder()
    expected = as_sparse(scaled_mat).multiply(other)
    result = scaled_mat.multiply(other)
    np.testing.assert_almost_equal(result.A, expected.A)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
@pytest.mark.parametrize("axis", [0, 1])
def test_sum_axis(scaled_mat_builder, axis):
    scaled_mat = scaled_mat_builder()
    expected = as_sparse(scaled_mat).sum(axis)
    result = scaled_mat.sum(axis)
    assert isinstance(result, np.ndarray)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_sum_none(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    result = scaled_mat.sum(None)
    assert np.isscalar(result)
    np.testing.assert_almost_equal(result, scaled_mat.A.sum(None))


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
@pytest.mark.parametrize("axis", [0, 1])
def test_mean_axis(scaled_mat_builder, axis):
    scaled_mat = scaled_mat_builder()
    expected = as_sparse(scaled_mat).mean(axis)
    result = scaled_mat.mean(axis)
    assert isinstance(result, np.ndarray)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_mean_none(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    result = scaled_mat.mean(None)
    assert np.isscalar(result)
    np.testing.assert_almost_equal(result, scaled_mat.A.mean(None))
