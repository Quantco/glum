import os
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import attr
import numpy as np
import pandas as pd
import tabmat as tm
from dask_ml.preprocessing import DummyEncoder
from git_root import git_root
from joblib import Memory
from scipy.sparse import csc_matrix

from .data import (
    generate_housing_dataset,
    generate_intermediate_insurance_dataset,
    generate_narrow_insurance_dataset,
    generate_real_insurance_dataset,
    generate_wide_insurance_dataset,
)
from .util import cache_location, exposure_and_offset_to_weights, get_tweedie_p

joblib_memory = Memory(cache_location, verbose=0)


@attr.s
class Problem:
    """Store metadata about which problem we should run."""

    data_loader = attr.ib(type=Callable)
    distribution = attr.ib(type=str)
    regularization_strength = attr.ib(type=float)
    l1_ratio = attr.ib(type=float)


@joblib_memory.cache
def load_data(
    loader_func: Callable[
        [Optional[int], Optional[float], Optional[str]],
        Tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ],
    num_rows: Optional[int] = None,
    storage: str = "dense",
    single_precision: bool = False,
    noise: Optional[float] = None,
    distribution: str = "poisson",
    data_setup: str = "weights",
) -> Dict[str, np.ndarray]:
    """
    Load the data.

    A note about weights and exposures: Due to the way we have set up this problem, by
    rescaling the target variable, it is appropriate to pass what is modeled as an
    'exposure' as a weight. Everywhere else, exposures will be referred to as weights.
    """
    # TODO: add a weights_and_offset option
    # Step 1) Load the data.
    if data_setup not in ["weights", "offset", "no-weights"]:
        raise NotImplementedError
    X_in, y, exposure = loader_func(num_rows, noise, distribution)

    # Step 2) Convert to needed precision level.
    if single_precision:
        X_in = X_in.astype(np.float32)
        y = y.astype(np.float32)
        if exposure is not None:
            exposure = exposure.astype(np.float32)

    # Step 3) One hot encode columns if we are not using CategoricalMatrix
    def transform_col(i: int, dtype) -> Union[pd.DataFrame, tm.CategoricalMatrix]:
        if dtype.name == "category":
            if storage == "cat":
                return tm.CategoricalMatrix(X_in.iloc[:, i])
            return DummyEncoder().fit_transform(X_in.iloc[:, [i]])
        return X_in.iloc[:, [i]]

    mat_parts = [transform_col(i, dtype) for i, dtype in enumerate(X_in.dtypes)]
    # TODO: add a threshold for the number of categories needed to make a categorical
    #  matrix

    # Step 4) Convert the matrix to the appopriate storage type.
    if storage == "auto":
        dtype = np.float32 if single_precision else np.float64
        X = tm.from_pandas(X_in, dtype, sparse_threshold=0.1, cat_threshold=3)
    elif storage == "cat":
        cat_indices_in_expanded_arr: List[np.ndarray] = []
        dense_indices_in_expanded_arr: List[int] = []
        i = 0
        for elt in mat_parts:
            assert elt.ndim == 2
            if isinstance(elt, tm.CategoricalMatrix):
                ncol = elt.shape[1]
                cat_indices_in_expanded_arr.append(np.arange(i, i + ncol))
                i += ncol
            else:
                dense_indices_in_expanded_arr.append(i)
                i += 1

        non_cat_part = tm.DenseMatrix(
            np.hstack(
                [
                    elt.values
                    for elt in mat_parts
                    if not isinstance(elt, tm.CategoricalMatrix)
                ]
            )
        )
        X = tm.SplitMatrix(
            matrices=[non_cat_part]
            + [elt for elt in mat_parts if isinstance(elt, tm.CategoricalMatrix)],
            indices=[np.array(dense_indices_in_expanded_arr)]
            + cat_indices_in_expanded_arr,
        )
    elif storage == "sparse":
        X = csc_matrix(pd.concat(mat_parts, axis=1))
    elif storage.startswith("split"):
        threshold = float(storage.split("split")[1])
        X = tm.from_csc(csc_matrix(pd.concat(mat_parts, axis=1)), threshold)
    else:  # Fall back to using a dense matrix.
        X = pd.concat(mat_parts, axis=1)

    # Step 5) Handle weights or offsets if needed.
    if data_setup == "weights":
        # The exposure correction doesn't make sense for these distributions since
        # they don't use a log link (plus binomial isn't in the tweedie family),
        # but this is what we were doing before.
        if distribution in ["gaussian", "binomial"]:
            return dict(X=X, y=y, sample_weight=exposure)
        # when poisson, should be y=y, sample_weight=exposure
        # instead have y = y / exposure, weight = exposure
        y, sample_weight = exposure_and_offset_to_weights(
            get_tweedie_p(distribution), y, exposure
        )
        return dict(X=X, y=y * exposure, sample_weight=sample_weight)

    if data_setup == "offset":
        log_exposure = np.log(exposure)
        assert np.all(np.isfinite(log_exposure))
        # y has already been divided by exposure loader_func, so undo it here
        return dict(X=X, y=y * exposure, offset=log_exposure)
    # data_setup = "no_weights"
    return dict(X=X, y=y)


def get_all_problems() -> Dict[str, Problem]:
    """
    Return the name of all possible problems.

    Returns
    -------
    Dict mapping problem names to Problem instances.

    """
    regularization_strength = 0.001

    housing_distributions = ["gaussian", "gamma", "binomial"]
    housing_load_funcs = {
        "intermediate-housing": generate_housing_dataset,
    }

    insurance_distributions = [
        "gaussian",
        "poisson",
        "gamma",
        "tweedie-p=1.5",
        "binomial",
    ]
    insurance_load_funcs = {
        "intermediate-insurance": generate_intermediate_insurance_dataset,
        "narrow-insurance": generate_narrow_insurance_dataset,
        "wide-insurance": generate_wide_insurance_dataset,
    }
    if os.path.isfile(git_root("data", "X.parquet")):
        insurance_load_funcs["real-insurance"] = generate_real_insurance_dataset

    problems = {}
    for penalty_str, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        # Add housing problems
        for distribution in housing_distributions:
            suffix = penalty_str + "-" + distribution
            dist = distribution
            for problem_name, load_fn in housing_load_funcs.items():
                for data_setup in ["no-weights", "offset"]:
                    problems["-".join((problem_name, data_setup, suffix))] = Problem(
                        data_loader=partial(
                            load_data, load_fn, distribution=dist, data_setup=data_setup
                        ),
                        distribution=distribution,
                        regularization_strength=regularization_strength,
                        l1_ratio=l1_ratio,
                    )
        # Add insurance problems
        for distribution in insurance_distributions:
            suffix = penalty_str + "-" + distribution
            dist = distribution
            for problem_name, load_fn in insurance_load_funcs.items():
                for data_setup in ["weights", "no-weights", "offset"]:
                    problems["-".join((problem_name, data_setup, suffix))] = Problem(
                        data_loader=partial(
                            load_data, load_fn, distribution=dist, data_setup=data_setup
                        ),
                        distribution=distribution,
                        regularization_strength=regularization_strength,
                        l1_ratio=l1_ratio,
                    )

    return problems
