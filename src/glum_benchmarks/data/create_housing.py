import os
from typing import Optional

import numpy as np
import pandas as pd
from git_root import git_root
from sklearn.datasets import fetch_openml


def create_housing_raw_data() -> None:
    """Do some basic processing on the data that we will later transform into our \
    benchmark data sets."""
    # Load the dataset from sklearn
    house_data = fetch_openml(name="house_sales", version=3, as_frame=True)

    # Use only select features
    df_house = house_data.data[
        [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "floors",
            "waterfront",
            "view",
            "condition",
            "grade",
            "yr_built",
            "yr_renovated",
        ]
    ].copy()

    # Targets
    df_house["price"] = house_data.target
    df_house["above_median_price"] = (
        df_house["price"] < df_house["price"].median()
    ).astype("int")

    # Save
    out_path = git_root("data/housing.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_house.to_parquet(git_root("data/housing.parquet"))


def add_noise(df: pd.DataFrame, noise: float) -> pd.DataFrame:
    """Add noise by swapping out data points."""
    np.random.seed(43212)
    for col in df.columns:
        if col in ["price", "above_median_price"]:
            continue
        swap = np.random.uniform(size=len(df)) < noise
        shuffle = np.random.choice(df[col], size=len(df))
        df.loc[swap, col] = shuffle[swap]

    return df


def compute_y_exposure(df, distribution):
    """Compute y and exposure depending on distribution.

    (Exposure/weights for housing data always all 1).
    """
    if distribution in ["gamma", "gaussian"]:
        y = df["price"].values
    elif distribution == "binomial":
        y = df["above_median_price"].values
    else:
        raise ValueError(
            f"distribution for housing problems must be one of"
            f"['gamma', 'gaussian', 'binomial'] not {distribution}."
        )

    return y, np.ones_like(y)


def _read_housing_data(
    num_rows: Optional[int], noise: Optional[float], distribution: str
) -> pd.DataFrame:
    path = git_root("data/housing.parquet")
    if not os.path.exists(path):
        create_housing_raw_data()
    df = pd.read_parquet(path)

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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the openml house_sales housing dataset."""
    df = _read_housing_data(num_rows, noise, distribution)

    y, exposure = compute_y_exposure(df, distribution)

    return df, y, exposure


if __name__ == "__main__":
    create_housing_raw_data()
