from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import tensorflow_probability as tfp

from glm_benchmarks.utils import load_data, runtime


def tensorflow_bench(dat: Dict[str, Union[pd.Series, pd.DataFrame]]) -> Dict[str, Any]:
    results = {}
    model = tfp.glm.fit(
        model_matrix=dat["X"].values,
        response=dat['y'].values,
        model=tfp.glm.Normal(),
        l2_regularizer=1
    )
    return results


def main():
    dat = load_data(nrows=1000)
    x = dat['X']
    print(x.shape)
    tensorflow_bench(dat)


if __name__ == "__main__":
    main()
