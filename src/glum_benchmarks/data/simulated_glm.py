from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse as sps

from glum._glm import get_family, get_link


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
    out = np.empty(n, dtype=np.float64)
    for i, N in enumerate(arr_N):
        out[i] = np.sum(rand.gamma(alpha_, 1 / beta_[i], size=N))

    return out


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


def simulate_glm_data(
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

    link_inst = get_link(link=link, family=get_family("poisson"))
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
