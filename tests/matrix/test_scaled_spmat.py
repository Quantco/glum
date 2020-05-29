from typing import Union

import numpy as np
import pytest
from scipy import sparse as sps

from glm_benchmarks.matrix.scaled_mat import ColScaledMat, RowScaledMat


def row_scaled_mat() -> RowScaledMat:
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    shift = np.random.uniform(0, 1, n_rows)
    return RowScaledMat(sp_mat, shift)


def col_scaled_mat() -> ColScaledMat:
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    col_shift = np.random.uniform(0, 1, n_cols)
    return ColScaledMat(sp_mat, col_shift)


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
    col_scaled_mat = ColScaledMat(sp_mat, col_shift)
    expected = sp_mat.A + col_shift[None, :]
    assert col_scaled_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(col_scaled_mat.A, expected)


@pytest.mark.parametrize("mat_type", [ColScaledMat, RowScaledMat])
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


def as_sparse(x: Union[ColScaledMat, RowScaledMat]) -> sps.csc_matrix:
    return sps.csc_matrix(x.A)


@pytest.mark.parametrize("scaled_mat_builder", [col_scaled_mat, row_scaled_mat])
def test_transpose_reversible(scaled_mat_builder):
    scaled_mat = scaled_mat_builder()
    two_trans = scaled_mat.T.T
    assert (two_trans.mat != scaled_mat.mat).sum() == 0
    assert (two_trans.shift == scaled_mat.shift).all()
