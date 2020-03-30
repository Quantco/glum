import time
from typing import Any, Callable, Dict, Union

import git_root
import pandas as pd


def runtime(f: Callable) -> Callable:
    """
    >>> def original_func(x):
    ...     return {'x': x}
    """

    def g(*args, **kwargs) -> Dict[str, Any]:
        start = time.time()
        results = f(*args, **kwargs)
        end = time.time()
        results["runtime"] = end - start
        return results

    return g


def load_data(nrows: int = None) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    df = pd.read_parquet(git_root.git_root("data/data.parquet"))
    if nrows is not None:
        df = df.iloc[:nrows]
    X = df[[col for col in df.columns if col not in ["y", "exposure"]]]
    y = df["y"]
    exposure = df["exposure"]
    return dict(X=X, y=y, exposure=exposure)
