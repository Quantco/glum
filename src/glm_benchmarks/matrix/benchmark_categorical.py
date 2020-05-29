import time

import numpy as np
from scipy import sparse as sps

from glm_benchmarks.matrix.categorical_matrix import CategoricalCSRMatrix
from glm_benchmarks.matrix.mkl_sparse_matrix import MKLSparseMatrix
from glm_benchmarks.sklearn_fork import GeneralizedLinearRegressor


def _timeit(func, n_iters=1):
    start = time.time()
    for _ in range(n_iters):
        res = func()
    elapsed = time.time() - start
    return res, elapsed


def gen_data(n_categories: int, n_rows: int, max_category: int) -> np.ndarray:
    assert max_category > n_categories
    np.random.seed(0)
    cats = np.random.choice(
        np.arange(max_category, dtype=int), n_categories, replace=False
    )
    return np.random.choice(cats, n_rows, replace=True)


def sandwich_dot_bench():
    n_categories = int(1e3)
    n_rows = int(1e6)
    max_category = int(2e5)
    cat_vec = gen_data(n_categories, n_rows, max_category)

    n_categories_used = len(np.unique(cat_vec))
    vec = np.random.random(n_categories_used)

    cat_mat, elapsed = _timeit(lambda: CategoricalCSRMatrix(cat_vec))
    print("Set-up time CategoricalMatrix: ", elapsed)

    # sandwich
    vec = np.random.random(n_rows)
    res_old, time_old = _timeit(lambda: cat_mat.sandwich_old(vec))
    cat_mat.x_csc = None
    res_cython, time_cython = _timeit(lambda: cat_mat.sandwich(vec))

    np.testing.assert_allclose(res_cython.data, res_old.data, atol=1e-11)
    print("old time: ", time_old)
    print("cython time: ", time_cython)


def do_sklearn_bench():
    n_categories = int(1e3)
    n_rows = int(1e6)
    max_category = int(2e5)
    cat_vec = gen_data(n_categories, n_rows, max_category)

    cat_mat = CategoricalCSRMatrix(cat_vec)
    coefs = sps.random(cat_mat.shape[1], 1).A
    mu = cat_mat.dot(coefs)
    print(mu.min(), mu.mean(), mu.max())
    y = np.random.poisson(mu)

    est = GeneralizedLinearRegressor(family="poisson")
    cat_res, cat_time = _timeit(lambda: est.fit(cat_mat, y))

    mkl_mat = MKLSparseMatrix(cat_mat._check_csc())
    mkl_res, mkl_time = _timeit(lambda: est.fit(mkl_mat, y))

    np.testing.assert_allclose(cat_res.coef_, mkl_res.coef_)
    print("cat time", cat_time)
    print("mkl time", mkl_time)


if __name__ == "__main__":
    do_sklearn_bench()
