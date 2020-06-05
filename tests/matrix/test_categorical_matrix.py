import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from quantcore.glm.matrix.categorical_matrix import CategoricalCSRMatrix


@pytest.mark.parametrize("vec_dtype", [np.float64, np.float32, np.int64, np.int32])
def test_csr_dot_categorical(vec_dtype):
    m = 10
    seed = 0
    np.random.seed(seed)
    cat = np.random.choice(np.arange(4, dtype=int), m)
    mat = OneHotEncoder().fit_transform(cat[:, None])
    cat_mat = CategoricalCSRMatrix(cat)
    vec = np.random.choice(np.arange(4, dtype=vec_dtype), mat.shape[1])
    res = cat_mat.dot(vec)
    np.testing.assert_allclose(res, mat.dot(vec))
