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
def test_recover_orig(cat_vec, vec_dtype):
    orig_recovered = CategoricalMatrix(cat_vec).recover_orig()
    np.testing.assert_equal(orig_recovered, cat_vec)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_csr_dot_categorical(cat_vec, vec_dtype):
    mat = OneHotEncoder().fit_transform(cat_vec[:, None])
    cat_mat = CategoricalMatrix(cat_vec)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.dot(vec)
    np.testing.assert_allclose(res, cat_mat.A.dot(vec))


def test_tocsr(cat_vec):
    cat_mat = CategoricalMatrix(cat_vec)
    res = cat_mat.tocsr().A
    expected = OneHotEncoder().fit_transform(cat_vec[:, None]).A
    np.testing.assert_allclose(res, expected)


def test_check_csc(cat_vec):
    cat_mat = CategoricalMatrix(cat_vec)
    data, indices, indptr = cat_mat._check_csc()
    data = np.ones(cat_mat.shape[0], dtype=int) if data is None else data
    res = sps.csc_matrix((data, indices, indptr), shape=cat_mat.shape)
    expected = cat_mat.tocsr().tocsc()
    np.testing.assert_allclose(res.indices, expected.indices)
    np.testing.assert_allclose(res.indptr, expected.indptr)


def test_to_csc(cat_vec):
    cat_mat = CategoricalMatrix(cat_vec)
    res = cat_mat.tocsc()
    expected = cat_mat.tocsr().tocsc()
    np.testing.assert_allclose(res.indices, expected.indices)
    np.testing.assert_allclose(res.indptr, expected.indptr)


def test_transpose_dot(cat_vec):
    cat_mat = CategoricalMatrix(cat_vec)
    other = np.random.random(cat_mat.shape[0])
    res = cat_mat.transpose_dot(other)
    expected = cat_mat.A.T.dot(other)
    np.testing.assert_allclose(res, expected)
