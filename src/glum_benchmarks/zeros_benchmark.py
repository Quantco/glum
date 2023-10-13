from typing import Any, Union

import numpy as np
from scipy import sparse as sps


def zeros_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]], *, cv: bool = False, **kwargs
) -> dict[str, Any]:
    """
    Run a "benchmark" of how long it takes to return all zero coefficients.

    Parameters
    ----------
    dat
    cv
    kwargs
    """
    result = {
        "runtime": 0,
        "model_obj": None,
        "intercept": 0,
        "coef": np.zeros(dat["X"].shape[1]),
        "n_iter": 1,
    }
    if cv:
        result["alpha"] = 0
    return result
