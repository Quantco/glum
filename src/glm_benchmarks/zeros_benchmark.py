from typing import Any, Dict, Union

import numpy as np
from scipy import sparse as sps


def zeros_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    print_diagnostics,
) -> Dict[str, Any]:

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
