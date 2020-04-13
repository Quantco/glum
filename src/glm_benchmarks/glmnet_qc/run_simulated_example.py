import time
from typing import Any, Dict

import numpy as np
from scipy import sparse as sps

from glm_benchmarks.glmnet_qc.glmnet_qc import fit_pathwise


def sim_data(n_rows: int, n_cols: int, sparse: bool) -> Dict[str, Any]:
    intercept = -3
    np.random.seed(0)
    x = sps.random(n_rows, n_cols, format="csc")
    if not sparse:
        x = x.A
    true_coefs = np.random.normal(0, 1, n_cols)
    y = x.dot(true_coefs)
    y = y + intercept + np.random.normal(0, 1, n_rows)
    return {"y": y, "x": x, "coefs": true_coefs}


def run(sparse: bool):
    data = sim_data(10000, 1000, sparse)
    y = data["y"]
    print("\n\n")
    solver = "sparse" if sparse else "naive"
    print(solver)
    x = data["x"]

    start = time.time()
    model = fit_pathwise(y, x, 0.5, n_iters=40)
    end = time.time()
    print("time", end - start)
    print("r2", model.get_r2(y))
    print("frac of coefs zero", (model.params == 0).mean())


if __name__ == "__main__":
    run(sparse=True)
