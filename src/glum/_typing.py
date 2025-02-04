from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import scipy.sparse
import tabmat as tm

VectorLike = Union[np.ndarray, pd.api.extensions.ExtensionArray, pd.Index, pd.Series]

ArrayLike = Union[
    list,
    tm.MatrixBase,
    tm.StandardizedMatrix,
    pd.DataFrame,
    scipy.sparse.spmatrix,
    VectorLike,
]

ShapedArrayLike = Union[
    tm.MatrixBase,
    tm.StandardizedMatrix,
    pd.DataFrame,
    scipy.sparse.spmatrix,
    VectorLike,
]


class WaldTestResult(NamedTuple):
    test_statistic: float
    p_value: float
    df: int
