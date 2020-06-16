import time

import numpy as np
import pandas as pd
from scipy import sparse as sps

from quantcore.glm.matrix import CategoricalMatrix, MKLSparseMatrix
from quantcore.glm.sklearn_fork import GeneralizedLinearRegressor


def main():
    n_rows = int(1e6)
    n_possible_cats = int(3e3)
    n_cats = int(2e3)
    cats = np.random.choice(np.arange(n_possible_cats), n_cats, replace=False)

    np.random.seed(0)
    cats = np.random.choice(cats, n_rows)

    cat_mat = CategoricalMatrix(cats)
    csr_mat = MKLSparseMatrix(cat_mat.tocsr()).astype(np.float64)
    coefs = np.random.normal(0, 0.1, cat_mat.shape[1])

    times = {}
    start = time.time()
    lin_pred_cat = cat_mat.dot(coefs)
    times["cat"] = time.time() - start

    start = time.time()
    lin_pred_old = csr_mat.dot(coefs)
    times["old"] = time.time() - start
    np.testing.assert_allclose(lin_pred_old, lin_pred_cat)
    print("dot times")
    times = pd.Series(times)
    print(times["old"] / times)

    # TODO: check dtype promotion
    y = np.random.normal(cat_mat.dot(coefs))
    times = {}

    model = GeneralizedLinearRegressor(alpha=1e-5, l1_ratio=0.5)
    start = time.time()
    model.fit(cat_mat, y)
    times["cat_time"] = time.time() - start

    model2 = GeneralizedLinearRegressor(alpha=1e-5, l1_ratio=0.5)
    start = time.time()
    model2.fit(csr_mat, y)
    times["csr_time"] = time.time() - start
    times = pd.Series(times)
    print("Fit times")
    print(times["csr_time"] / times)

    np.testing.assert_almost_equal(model2.coef_, model.coef_)

    # Try out sandwich
    d = np.random.random(n_rows)
    times = {}

    start = time.time()
    cython_res = cat_mat.sandwich(d)
    times["cython_sand"] = time.time() - start

    start = time.time()
    csr_res = csr_mat.sandwich(d)
    times["csr_time"] = time.time() - start
    assert csr_res.shape == cython_res.shape
    as_sparse = sps.dia_matrix(csr_res)
    assert (as_sparse != cython_res).sum() == 0
    times = pd.Series(times)
    print(times["csr_time"] / times)


if __name__ == "__main__":
    main()
