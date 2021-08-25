from typing import Optional, Tuple

import numpy as np
import pandas as pd
from git_root import git_root
from sklearn.datasets import load_boston

# taken from https://github.com/lorentzenchr/Tutorial_freMTPL2/blob/master/glm_freMTPL2_example.ipynb  # noqa: B950
# Modified to generate data sets of different sizes


def create_housing_raw_data() -> None:
    """Do some basic processing on the data that we will later transform into our \
    benchmark data sets."""
    # Load the dataset from sklearn
    boston = load_boston()
    df_bos = pd.DataFrame(boston.data, columns=boston.feature_names)

    # Use only select features
    df_bos = df_bos[["CRIM", "ZN", "CHAS", "NOX", "RM", "AGE", "TAX", "B", "LSTAT"]]

    # Targets
    df_bos["PRICE"] = boston.target
    df_bos["ABOVE_MEDIAN_PRICE"] = (df_bos["PRICE"] < df_bos["PRICE"].median()).astype(
        "int"
    )

    # Remove outliers
    df_bos = df_bos[df_bos["PRICE"] <= 40]

    # Save
    df_bos.to_parquet(git_root("data/housing.parquet"))


def add_noise(df: pd.DataFrame, noise: float) -> pd.DataFrame:
    """Add noise by swapping out data points."""
    np.random.seed(43212)
    for col in df.columns:
        if col in ["PRICE", "ABOVE_MEDIAN_PRICE"]:
            continue
        swap = np.random.uniform(size=len(df)) < noise
        shuffle = np.random.choice(df[col], size=len(df))
        df.loc[swap, col] = shuffle[swap]

    return df


def compute_y_exposure(df, distribution):
    """Compute y and exposure depending on distribution.

    (Exposure/weights for boston housing data always all 1).
    """
    if distribution in ["gamma", "gaussian"]:
        y = df["PRICE"].values
    elif distribution == "binomial":
        y = df["ABOVE_MEDIAN_PRICE"].values
    else:
        raise ValueError(
            f"distribution for boston housing problems must be one of"
            f"['gamma', 'gaussian', 'binomial'] not {distribution}."
        )

    return y, np.ones_like(y)


def _read_housing_data(
    num_rows: Optional[int], noise: Optional[float], distribution: str
) -> pd.DataFrame:
    df = pd.read_parquet(git_root("data/housing.parquet"))

    if num_rows is not None:
        # if we're oversampling, set default value for noise to 0.05
        # can be turned off by setting noise to zero
        if noise is None and num_rows > len(df):
            noise = 0.05
        df = df.sample(n=num_rows, replace=True, random_state=12345)

    if noise is not None:
        df = add_noise(df, noise=noise)
    return df


def generate_housing_dataset(
    num_rows=None, noise=None, distribution="poisson"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the sklearn boston housing dataset."""
    df = _read_housing_data(num_rows, noise, distribution)

    y, exposure = compute_y_exposure(df, distribution)

    return df, y, exposure


if __name__ == "__main__":
    create_housing_raw_data()
