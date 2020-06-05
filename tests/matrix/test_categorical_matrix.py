import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from quantcore.glm.matrix.categorical_matrix import CategoricalCSRMatrix


@pytest.fixture
def cat_vec():
    m = 10
    seed = 0
    np.random.seed(seed)
    return np.random.choice(np.arange(4, dtype=int), m)


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_csr_dot_categorical(cat_vec, vec_dtype):
    mat = OneHotEncoder().fit_transform(cat_vec[:, None])
    cat_mat = CategoricalCSRMatrix(cat_vec)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.dot(vec)
    np.testing.assert_allclose(res, mat.dot(vec))


def test_tocsr(cat_vec):
    cat_mat = CategoricalCSRMatrix(cat_vec)
    res = cat_mat.tocsr().A
    expected = OneHotEncoder().fit_transform(cat_vec[:, None]).A
    np.testing.assert_allclose(res, expected)


def test_transpose_dot(cat_vec):
    cat_mat = CategoricalCSRMatrix(cat_vec)
    other = np.random.random(cat_mat.shape[0])
    res = cat_mat.transpose_dot(other)
    expected = OneHotEncoder().fit_transform(cat_vec[:, None]).T.dot(other)
    np.testing.assert_allclose(res, expected)
