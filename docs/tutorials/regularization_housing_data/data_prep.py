import pandas as pd


def download_and_transform():
    """Download data from openml and apply basic transformations."""
    df = pd.read_csv("https://www.openml.org/data/get_csv/21578898/house_sales.arff")
    df = df.drop(columns="id")
    df["zipcode"] = pd.Categorical(df["zipcode"].astype(str), ordered=True)
    df = df[[c for c in df.columns if c != "price"] + ["price"]]

    df = df[(df["price"] < 1.5e6) & (df["price"] > 1e5)]

    drop_cols = [
        "date",
        "sqft_lot",
        "grade",
        "sqft_above",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]
    df = df.drop(columns=drop_cols)
    return df
