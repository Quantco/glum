import numpy as np
import pytest
from scipy import sparse as sps

from quantcore.glm.matrix import ColScaledMat


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


def test_setup_and_densify_row():
    n_rows = 4
    n_cols = 3

    np.random.seed(0)
    sp_mat = sps.random(n_rows, n_cols, density=0.8)
    shift = np.random.uniform(0, 1, sp_mat.shape[1])
    scaled_mat = ColScaledMat(sp_mat, shift)
    expected = sp_mat.A + np.expand_dims(shift, 0)
    assert scaled_mat.A.shape == (n_rows, n_cols)
    np.testing.assert_almost_equal(scaled_mat.A, expected)


def as_sparse(x: ColScaledMat) -> sps.csc_matrix:
    return sps.csc_matrix(x.A)
