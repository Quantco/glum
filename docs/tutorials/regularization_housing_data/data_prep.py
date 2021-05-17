import openml


def download_and_transform():
    """Download data from openml and apply basic transformations."""
    dataset = openml.datasets.get_dataset(42092)
    df, y, _, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    df["price"] = y

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
