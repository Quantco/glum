import time

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from glm_benchmarks.matrix.categorical_matrix import CategoricalMatrix


def main():
    n_categories = int(1e5)
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

    start = time.time()
    cat_mat = CategoricalMatrix(cat_vec)
    elapsed = time.time() - start
    print("Set-up time new: ", elapsed)
    assert cat_mat.shape == (n_rows, n_categories_used)

    start = time.time()
    csr = OneHotEncoder().fit_transform(cat_vec[:, None])
    elapsed = time.time() - start
    assert csr.shape == (n_rows, n_categories_used)
    print("Set-up time old: ", elapsed)

    n_iters = 100

    start = time.time()
    for _ in range(n_iters):
        res1 = cat_mat.dot(vec)
    elapsed1 = time.time() - start

    start = time.time()
    for _ in range(n_iters):
        res2 = csr.dot(vec)
    elapsed2 = time.time() - start

    print("Computation new", elapsed1)
    print("Computation old", elapsed2)
    print("Result diff", np.max(np.abs(res2 - res1)))


if __name__ == "__main__":
    main()
