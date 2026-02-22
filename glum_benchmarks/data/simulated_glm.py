from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sps

from glum._distribution import TweedieDistribution
from glum._glm import get_family, get_link


def _resolve_family(family: str):
    """Convert benchmark family strings like 'tweedie_p=1.5' to glum family objects."""
    if "tweedie" in family and "=" in family:
        p = float(family.split("=")[1])
        return TweedieDistribution(p)
    return get_family(family)


def tweedie_rv(mu, sigma2=1, p=1.5):
    """Generate draws from a tweedie distribution with power p.

    mu is the location parameter and sigma2 is the dispersion coefficient.
    """
    n = len(mu)
    rand = np.random.default_rng(1)

    # transform tweedie parameters into poisson and gamma
    lambda_ = (mu ** (2 - p)) / ((2 - p) * sigma2)
    alpha_ = (2 - p) / (p - 1)
    beta_ = (mu ** (1 - p)) / ((p - 1) * sigma2)

    arr_N = rand.poisson(lambda_)
    out: np.ndarray = np.empty(n, dtype=np.float64)
    for i, N in enumerate(arr_N):  # type: ignore
        out[i] = np.sum(rand.gamma(alpha_, 1 / beta_[i], size=N))

    return out


def _gamma_rv(mu, rand: np.random._generator.Generator, shape: float = 2.0):
    """Generate gamma random variates with specified mean.

    Parameters
    ----------
    mu : array-like
        The desired mean values. Must be positive.
    rand : np.random.Generator
        Random number generator.
    shape : float
        Shape parameter (k). Higher values give less variance. Default 2.0.

    Returns
    -------
    array
        Gamma random variates with E[y] = mu.
    """
    mu = np.asarray(mu)
    scale = mu / shape
    y = rand.gamma(shape, scale)
    # Ensure strictly positive (gamma GLM requires y > 0)
    return np.maximum(y, np.finfo(float).eps)


def _get_family_rv(family, rand: np.random._generator.Generator):
    family_rv = {
        "poisson": rand.poisson,
        "gamma": rand.gamma,
        "normal": rand.normal,
        "binomial": partial(rand.binomial, 1),
    }

    if family in family_rv.keys():
        return family_rv[family]
    elif "tweedie" in family:
        p = float(family.split("=")[1])
        return partial(tweedie_rv, p=p)
    else:
        raise ValueError(
            'family must take the value "poisson", "gamma", "normal", "binomial", or '
            '"tweedie_p=XX". '
            f"Currently {family}."
        )


def simulate_mixed_data(
    family: str = "poisson",
    link: str = "auto",
    n_rows: int = 5000,
    dense_features: int = 10,
    sparse_features: int = 0,
    sparse_density: float = 0.05,
    categorical_features: int = 2,
    categorical_levels: int = 10,
    ohe_categorical: bool = True,
    intercept: float = 0.2,
    drop_first: bool = False,
    random_seed: int = 1,
):
    """
    Simulate the data we will use for benchmarks.

    Parameters
    ----------
    family
    link
    n_rows
    dense_features
    sparse_features
    sparse_density
    categorical_features
    categorical_levels
    ohe_categorical
    intercept
    drop_first
    random_seed

    Returns
    -------
    dict
    """
    rand = np.random.default_rng(random_seed)

    # Creating dense component
    if dense_features > 0:
        dense_feature_names = [f"dense{i}" for i in range(dense_features)]
        X_dense = rand.normal(
            rand.integers(-2, 2, size=dense_features), size=(n_rows, dense_features)
        )
        X_dense = pd.DataFrame(data=X_dense, columns=dense_feature_names)
        coefs_dense = np.concatenate(
            [
                [1, 0.5, 0.1, -0.1, -0.5, -1, 0, 0, 0, 0],
                rand.choice([0, 1, -1], size=dense_features),
            ]
        )[:dense_features]
        coefs_dense = pd.Series(data=coefs_dense, index=dense_feature_names)

    # Creating sparse component
    sparse_feature_names = [f"sparse{i}" for i in range(sparse_features)]
    X_sparse = sps.random(n_rows, sparse_features, density=sparse_density).toarray()
    X_sparse = pd.DataFrame(data=X_sparse, columns=sparse_feature_names)
    coefs_sparse = rand.choice([0, 1, -1], size=sparse_features)
    coefs_sparse = pd.Series(data=coefs_sparse, index=sparse_feature_names)

    # Creating categorical component
    cat_feature_names = [f"cat{i}" for i in range(categorical_features)]
    fixed_effects = rand.choice(
        np.arange(categorical_levels), size=(n_rows, categorical_features)
    )
    X_cat = pd.DataFrame(data=fixed_effects, columns=cat_feature_names)
    # Convert categorical columns to dtype 'category'
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].astype("category")
    X_cat_ohe = pd.get_dummies(
        X_cat, columns=cat_feature_names, drop_first=drop_first, dtype=float
    )

    coefs_cat = pd.Series(
        data=rand.uniform(size=len(X_cat_ohe.columns)), index=X_cat_ohe.columns
    )

    # Merging
    X = pd.concat([X_dense, X_sparse, X_cat_ohe], axis=1)
    coefs = pd.concat([coefs_dense, coefs_sparse, coefs_cat])

    intercept = intercept

    link_inst = get_link(link=link, family=_resolve_family(family))
    family_rv = _get_family_rv(family, rand)

    y = family_rv(link_inst.inverse(intercept + X.to_numpy() @ coefs.to_numpy()))

    weights = rand.uniform(size=n_rows)
    offset = np.log(rand.uniform(size=n_rows))

    if not ohe_categorical:
        X = pd.concat([X_dense, X_sparse, X_cat], axis=1)

    data = {
        "X": X,
        "y": y,
        "sample_weight": weights,
        "offset": offset,
        "intercept": intercept,
        "coefs": coefs,
    }
    return data


def simulate_glm_dataset(
    num_rows: Optional[int] = None,
    noise: Optional[float] = None,  # unused, required by load_data signature
    distribution: str = "poisson",
    k_over_n_ratio: Optional[float] = 1.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a simulated GLM dataset with configurable K/N ratio.

    Parameters
    ----------
    num_rows
        Number of rows. Defaults to 1000.
    noise
        Unused, present to match load_data signature.
    distribution
        The GLM family: "gaussian", "poisson", "gamma", or "binomial".
    k_over_n_ratio
        Ratio of number of features to number of rows (K/N).
        - 1.0 gives a square design matrix
        - > 1.0 gives high-dimensional data (K > N)
        - < 1.0 gives low-dimensional data (K < N)

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        (X, y, exposure).
    """
    n_rows = num_rows if num_rows is not None else 1000
    ratio = 1.0 if k_over_n_ratio is None else float(k_over_n_ratio)
    if ratio <= 0:
        raise ValueError("k_over_n_ratio must be > 0.")
    n_features = max(1, int(round(n_rows * ratio)))
    rand = np.random.default_rng(42)

    # Generate standardized features
    X = pd.DataFrame(
        data=rand.normal(0, 1, size=(n_rows, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
    )

    # Sparse coefficients (~10% non-zero)
    coefs = rand.choice([0] * 9 + [1], size=n_features) * rand.normal(
        0, 1, size=n_features
    )

    # Linear predictor
    eta = X.to_numpy() @ coefs

    # Map distribution names (_get_family_rv uses "normal" not "gaussian")
    family = "normal" if distribution == "gaussian" else distribution

    # Get the link function for the distribution
    family_inst = _resolve_family(family)
    link_inst = get_link(link="auto", family=family_inst)

    # Compute mu using the inverse link
    mu = link_inst.inverse(np.clip(eta, -5, 5))

    # Generate y based on distribution using the family's random variate generator.
    # Use improved gamma parameterization for benchmarks.
    if family == "gamma":
        y = _gamma_rv(mu, rand=rand)
    else:
        family_rv = _get_family_rv(family, rand)
        y = family_rv(mu)

    exposure = np.ones(n_rows)
    return X, y, exposure


def simulate_categorical_dataset(
    num_rows: Optional[int] = None,
    noise: Optional[float] = None,  # unused, required by load_data signature
    distribution: str = "gaussian",
    categorical_ratio: float = 0.9,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a dataset with a high-cardinality categorical feature.

    This creates a dataset with a single high-cardinality categorical
    feature (stored as categorical values, not one-hot encoded).

    Parameters
    ----------
    num_rows
        Number of rows. Defaults to 1000.
    noise
        Unused, present to match load_data signature.
    distribution
        The GLM family: "gaussian", "poisson", "gamma", or "binomial".
    categorical_ratio
        Controls the number of categorical levels relative to rows.
        Default 0.9 means ~0.9 * n levels for the categorical feature.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        (X, y, exposure) where X has dense columns and one categorical column.
    """
    n = num_rows if num_rows is not None else 1000

    # Number of categorical levels (creates this many one-hot columns)
    n_cat_levels = int(n * categorical_ratio)

    # Map distribution names (simulate_mixed_data uses "normal" not "gaussian")
    family = "normal" if distribution == "gaussian" else distribution

    data = simulate_mixed_data(
        family=family,
        link="auto",
        n_rows=n,
        dense_features=5,
        sparse_features=0,
        categorical_features=1,
        categorical_levels=n_cat_levels,
        ohe_categorical=False,
    )

    exposure = np.ones(n)
    return data["X"], data["y"], exposure
