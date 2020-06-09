import numpy as np
import pytest
from scipy import sparse as sps

from quantcore.glm.matrix import MKLSparseMatrix


@pytest.fixture
def x() -> sps.csc_matrix:
    np.random.seed(0)
    return sps.random(10, 3, density=0.1).tocsc()


def test_mkl_sparse_init(x: sps.csc_matrix):
    one = MKLSparseMatrix(x)
    two = MKLSparseMatrix((x.data, x.indices, x.indptr), shape=x.shape)
    three = MKLSparseMatrix(x.A)
    np.testing.assert_allclose(one.A, two.A)
    np.testing.assert_allclose(one.A, three.A)


def test_to_csc(x: sps.csc_matrix):
    one = MKLSparseMatrix(x).tocsc()
    two = MKLSparseMatrix((x.data, x.indices, x.indptr), shape=x.shape).tocsc()
    three = MKLSparseMatrix(x.A).tocsc()

    assert isinstance(one, sps.csc_matrix)
    assert isinstance(two, sps.csc_matrix)
    assert isinstance(three, sps.csc_matrix)

    def _test_same(mat1, mat2):
        assert mat1.nnz == mat2.nnz
        np.testing.assert_allclose(mat1.data, mat2.data)
        assert (mat1.indices == mat2.indices).all()
        assert (mat1.indptr == mat2.indptr).all()

    _test_same(one, two)
    _test_same(one, three)
