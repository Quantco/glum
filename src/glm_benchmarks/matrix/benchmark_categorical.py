import time

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.preprocessing import OneHotEncoder

from glm_benchmarks.matrix.categorical_matrix import CategoricalCSRMatrix
from glm_benchmarks.matrix.mkl_sparse_matrix import MKLSparseMatrix


def _timeit(func, n_iters=1):
    start = time.time()
    for _ in range(n_iters):
        res = func()
    elapsed = time.time() - start
    return res, elapsed


def main():
    n_categories = int(1e3)
    n_rows = int(1e6)
    max_category = int(2e5)
    assert max_category > n_categories
    np.random.seed(0)
    cats = np.random.choice(
        np.arange(max_category, dtype=int), n_categories, replace=False
    )
    cat_vec = np.random.choice(cats, n_rows, replace=True)
    n_categories_used = len(np.unique(cat_vec))
    vec = np.random.random(n_categories_used)

    cat_mat, elapsed = _timeit(lambda: CategoricalCSRMatrix(cat_vec))
    print("Set-up time CategoricalMatrix: ", elapsed)

    csr, elapsed = _timeit(lambda: OneHotEncoder().fit_transform(cat_vec[:, None]))
    print("Set-up time scipy csr: ", elapsed)

    csc = csr.tocsc()

    mkl, elapsed = _timeit(lambda: MKLSparseMatrix(csc))
    mkl._check_csr()
    print("Additional set-up time MKLSparseMatrix: ", elapsed)

    n_iters = 100

    matrices = {"cat": cat_mat, "csr": csr, "csc": csc, "mkl": mkl}
    elapsed = {
        k: _timeit(lambda: mat.dot(vec), n_iters)[1] for k, mat in matrices.items()
    }

    print("\nMatrix-vector product times:")
    print(pd.Series(elapsed))

    # sandwich
    vec = np.random.random(n_rows)
    sandwich_times = {
        k: _timeit(lambda: matrices[k].sandwich(vec))[1] for k in ["cat", "mkl"]
    }
    sandwich_times["csc"] = _timeit(lambda: csc.T @ sps.diags(vec) @ csc)[1]
    sandwich_times["csr"] = _timeit(lambda: csr.T @ sps.diags(vec) @ csr)[1]
    print("\nsandwich times")
    print(pd.Series(sandwich_times))


if __name__ == "__main__":
    main()
