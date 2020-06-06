import time

import numpy as np

from quantcore.glm.matrix import CategoricalCSRMatrix, MKLSparseMatrix
from quantcore.glm.sklearn_fork import GeneralizedLinearRegressor


def main():
    n_rows = int(1e6)
    n_possible_cats = int(1e4)
    n_cats = int(2e3)
    cats = np.random.choice(np.arange(n_possible_cats), n_cats, replace=False)

    np.random.seed(0)
    cats = np.random.choice(cats, n_rows)

    cat_mat = CategoricalCSRMatrix(cats)
    coefs = np.random.normal(0, 0.1, cat_mat.shape[1])
    # TODO: check dtype promotion
    y = np.random.normal(cat_mat.dot(coefs))
    print(y[:10])

    model = GeneralizedLinearRegressor(alpha=1e-5, l1_ratio=0.5)
    start = time.time()
    model.fit(cat_mat, y)
    cat_time = time.time() - start
    print("cat time: ", cat_time)

    model2 = GeneralizedLinearRegressor(alpha=1e-5, l1_ratio=0.5)
    csr_mat = MKLSparseMatrix(cat_mat.tocsr())
    start = time.time()
    model2.fit(csr_mat, y)
    csr_time = time.time() - start

    print("csr time: ", csr_time)
    np.testing.assert_almost_equal(model2.coef_, model.coef_)


if __name__ == "__main__":
    main()
