import numpy as np
import pytest
from scipy import sparse as sps
from sklearn.preprocessing import OneHotEncoder

from quantcore.glm.matrix.categorical_matrix import CategoricalMatrix


@pytest.fixture
def cat_vec():
    m = 10
    seed = 0
    np.random.seed(seed)
    return np.random.choice(np.arange(4, dtype=int), m)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("col_mult", [None, [0, -0.1, 2]])
def test_csr_dot_categorical(cat_vec, vec_dtype, col_mult):
    mat = OneHotEncoder().fit_transform(cat_vec[:, None])
    cat_mat = CategoricalMatrix(cat_vec, col_mult)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.dot(vec)
    np.testing.assert_allclose(res, cat_mat.A.dot(vec))


@pytest.mark.parametrize("col_mult", [None, [0, -0.1, 2]])
def test_tocsr(cat_vec, col_mult):
    cat_mat = CategoricalMatrix(cat_vec, col_mult)
    res = cat_mat.tocsr().A
    expected = OneHotEncoder().fit_transform(cat_vec[:, None]).A
    if col_mult is not None:
        expected *= np.array(col_mult)[None, :]
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("col_mult", [None, [0, -0.1, 2]])
def test_check_csc(cat_vec, col_mult):
    cat_mat = CategoricalMatrix(cat_vec, col_mult)
    data, indices, indptr = cat_mat._check_csc()
    data = np.ones(cat_mat.shape[0], dtype=int) if data is None else data
    res = sps.csc_matrix((data, indices, indptr), shape=cat_mat.shape)
    expected = cat_mat.tocsr().tocsc()
    np.testing.assert_allclose(res.indices, expected.indices)
    np.testing.assert_allclose(res.indptr, expected.indptr)


@pytest.mark.parametrize("col_mult", [None, [0, -0.1, 2]])
def test_transpose_dot(cat_vec, col_mult):
    cat_mat = CategoricalMatrix(cat_vec, col_mult)
    other = np.random.random(cat_mat.shape[0])
    res = cat_mat.transpose_dot(other)
    expected = cat_mat.A.T.dot(other)
    np.testing.assert_allclose(res, expected)
