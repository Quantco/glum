import numpy as np
import pytest
from scipy import sparse as sps
from sklearn.preprocessing import OneHotEncoder

from glm_benchmarks.matrix.categorical_matrix import csr_dot, csr_dot_categorical


@pytest.mark.parametrize("mat_dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_csr_dot(mat_dtype, vec_dtype):
    m = 10
    n = 5
    seed = 0
    np.random.seed(seed)
    mat = (4 * sps.rand(m, n, density=0.3, random_state=seed)).tocsr().astype(mat_dtype)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), n)
    res = csr_dot(mat, vec)
    expected = mat.dot(vec)
    assert res.dtype == expected.dtype
    np.testing.assert_allclose(res, mat.dot(vec))


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_csr_dot_categorical(vec_dtype):
    m = 10
    seed = 0
    np.random.seed(seed)
    cat = np.random.choice(np.arange(4, dtype=int), m)[:, None]
    mat = OneHotEncoder().fit_transform(cat)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = csr_dot_categorical(mat.indices, vec)
    np.testing.assert_allclose(res, mat.dot(vec))
